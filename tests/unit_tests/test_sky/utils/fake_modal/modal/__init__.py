"""A fake Modal SDK: Sandbox.exec runs the command as a local process.

Tests put this directory on PYTHONPATH so the real exec helper, running as its
own process, imports it as `modal`. It follows the signature and behavior of
modal 1.6.3, including the ones a caller can get wrong: text mode by default,
a 2**16 byte limit on arguments, -1 from wait() when the deadline passes,
output that can outlast the command, and errors from from_id() and exec()
rather than from a later call.

The sandbox ID selects a failure:
    sb-gone: from_id() raises NotFoundError.
    sb-unauthorized: from_id() raises AuthError.
    sb-hang: from_id() never returns.
    sb-drop: output and wait() raise after the first chunk of output.

Sandbox.create starts a real sandbox for the end-to-end launch test; see
_sandbox_process.py. On such a sandbox, App.lookup, Sandbox.list, poll and
terminate go to the test's control plane; see _control.py.
"""
import json
import os
import queue
import signal
import subprocess
import threading
import time

from . import _control
from . import _sandbox_process
from . import config
from . import exception

_ARG_MAX_BYTES = 2**16


class _Reader:
    """Output that the server has already buffered, as Modal does.

    A command can exit with its output still undelivered.
    """

    def __init__(self, pipe, text, on_first_chunk=None):
        self._text = text
        self._on_first_chunk = on_first_chunk
        self._chunks = queue.Queue()
        threading.Thread(target=self._buffer, args=(pipe,), daemon=True).start()

    def _buffer(self, pipe):
        while True:
            chunk = pipe.read1(64 * 1024)
            self._chunks.put(chunk)
            if not chunk:
                return

    def __iter__(self):
        while True:
            chunk = self._chunks.get()
            if not chunk:
                return
            yield chunk.decode('utf-8') if self._text else chunk
            if self._on_first_chunk is not None:
                self._on_first_chunk()


class _Writer:

    def __init__(self, pipe):
        self._pipe = pipe
        self._buffer = bytearray()
        self._closed = False

    def write(self, data):
        if self._closed:
            raise ValueError('Stdin is closed. Cannot write to it.')
        self._buffer.extend(data)

    def write_eof(self):
        self._closed = True

    def drain(self):
        self._pipe.write(bytes(self._buffer))
        self._pipe.flush()
        self._buffer.clear()
        if self._closed:
            self._pipe.close()


class _Process:

    def __init__(self, popen, text, timeout, drops):
        self._popen = popen
        self.returncode = None
        self._timed_out = False
        self._dropped = threading.Event()
        on_first_chunk = self._drop if drops else None
        self.stdout = _Reader(popen.stdout, text, on_first_chunk)
        self.stderr = _Reader(popen.stderr, text)
        self.stdin = _Writer(popen.stdin)
        if timeout is not None:
            timer = threading.Timer(timeout, self._expire)
            timer.daemon = True
            timer.start()

    def _drop(self):
        self._dropped.set()
        raise ConnectionError('connection dropped')

    def _expire(self):
        self._timed_out = True
        os.killpg(self._popen.pid, signal.SIGKILL)

    def wait(self):
        while self._popen.poll() is None:
            if self._dropped.is_set():
                raise ConnectionError('connection dropped')
            time.sleep(0.01)
        if self._timed_out:
            self.returncode = -1
        else:
            returncode = self._popen.returncode
            self.returncode = (128 -
                               returncode if returncode < 0 else returncode)
        return self.returncode

    def attach(self):
        for chunk in self.stdout:
            os.write(1, chunk.encode() if isinstance(chunk, str) else chunk)


class _App:

    def __init__(self, name):
        self.name = name
        self.app_id = f'ap-{name}'


class App:

    @staticmethod
    def lookup(name, environment_name=None, create_if_missing=False):
        if _control.is_client():
            _control.call('lookup_app',
                          name=name,
                          environment_name=environment_name,
                          create_if_missing=create_if_missing)
            return _App(name)
        del environment_name
        marker = os.path.join(os.environ['FAKE_MODAL_STATE'], 'apps', name)
        if not os.path.exists(marker):
            if not create_if_missing:
                raise exception.NotFoundError(f'App {name} not found')
            os.makedirs(os.path.dirname(marker), exist_ok=True)
            with open(marker, 'w', encoding='utf-8'):
                pass
        return _App(name)


class Image:

    def __init__(self, name, commands=()):
        self.name = name
        self.commands = list(commands)

    @staticmethod
    def from_registry(name):
        return Image(name)

    def run_commands(self, *commands):
        return Image(self.name, self.commands + list(commands))


class Sandbox:

    def __init__(self, sandbox_id):
        self.object_id = sandbox_id

    @staticmethod
    def from_id(sandbox_id, client=None):
        del client
        if sandbox_id == 'sb-gone':
            raise exception.NotFoundError(f'{sandbox_id} not found')
        if sandbox_id == 'sb-unauthorized':
            raise exception.AuthError('bad token')
        if sandbox_id == 'sb-hang':
            threading.Event().wait()
        return Sandbox(sandbox_id)

    @staticmethod
    def from_name(app_name, name, environment_name=None):
        app = App.lookup(app_name, environment_name=environment_name)
        sandbox_id = _sandbox_process.find_by_name(app.app_id, name)
        if sandbox_id is None:
            raise exception.NotFoundError(f'Sandbox {name} not found')
        return Sandbox(sandbox_id)

    @staticmethod
    def create(*args, app=None, image=None, tags=None, timeout=300, **kwargs):
        name = kwargs.get('name')
        if name is not None and _sandbox_process.find_by_name(app.app_id,
                                                              name) is not None:
            raise exception.AlreadyExistsError(
                f'Sandbox with name {name!r} already exists')
        image_dir = _sandbox_process.build_image(image.commands)
        sandbox_id = _sandbox_process.create(list(args), image_dir, tags or {},
                                             app.app_id, timeout, kwargs)
        return Sandbox(sandbox_id)

    @staticmethod
    def list(app_id=None, tags=None):
        if _control.is_client():
            sandbox_ids = _control.call('list', app_id=app_id, tags=tags)
        else:
            sandbox_ids = _sandbox_process.list_ids(app_id, tags or {})
        for sandbox_id in sandbox_ids:
            yield Sandbox(sandbox_id)

    def poll(self):
        if _control.is_client():
            return _control.call('poll', sandbox_id=self.object_id)
        return _sandbox_process.poll(self.object_id)

    def terminate(self):
        if _control.is_client():
            _control.call('terminate', sandbox_id=self.object_id)
            return
        _sandbox_process.terminate(self.object_id)

    def exec(self,
             *args,
             stdout=subprocess.PIPE,
             stderr=subprocess.PIPE,
             timeout=None,
             workdir=None,
             env=None,
             secrets=None,
             text=True,
             bufsize=-1,
             pty=False):
        del secrets, bufsize
        total = sum(len(arg) for arg in args)
        if total > _ARG_MAX_BYTES:
            raise exception.InvalidError(
                f'Total length of CMD arguments cannot exceed {_ARG_MAX_BYTES} '
                f'bytes (ARG_MAX). Got {total} bytes.')
        with open(os.environ['FAKE_MODAL_LOG'], 'a', encoding='utf-8') as f:
            f.write(
                json.dumps({
                    'sandbox_id': self.object_id,
                    'args': args,
                    'timeout': timeout,
                    'env': env,
                    'text': text,
                    'pty': pty,
                }) + '\n')
        if _sandbox_process.is_process_sandbox(self.object_id):
            try:
                args = _sandbox_process.exec_args(self.object_id, args, workdir,
                                                  env)
            except ProcessLookupError as e:
                raise exception.NotFoundError(str(e)) from e
            workdir = None
            env = None
        popen = subprocess.Popen(  # pylint: disable=consider-using-with
            args,
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            cwd=workdir,
            env=dict(os.environ, **(env or {})),
            start_new_session=True)
        return _Process(popen, text, timeout, drops=self.object_id == 'sb-drop')


def serve_control_plane():
    """Not Modal's: serves SDK calls from sandboxes. See _control.py."""
    return _control.serve({
        'lookup_app': lambda **kwargs: App.lookup(**kwargs).app_id,
        'list': lambda app_id, tags: [
            sandbox.object_id
            for sandbox in Sandbox.list(app_id=app_id, tags=tags)
        ],
        'poll': lambda sandbox_id: Sandbox(sandbox_id).poll(),
        'terminate': lambda sandbox_id: Sandbox(sandbox_id).terminate(),
    })
