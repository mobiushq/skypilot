"""A fake Modal SDK: Sandbox.exec runs the command as a local process.

Tests put this directory on PYTHONPATH so the real exec helper, running as its
own process, imports it as `modal`. It follows the signature and behavior of
modal 1.6.3, including the ones a caller can get wrong: text mode by default,
a 2**16 byte limit on arguments, -1 from wait() when the deadline passes, and
errors from from_id() and exec() rather than from a later call.

The sandbox ID selects a failure:
    sb-gone: from_id() raises NotFoundError.
    sb-unauthorized: from_id() raises AuthError.
    sb-hang: from_id() never returns.
    sb-drop: output and wait() raise after the first chunk of output.
"""
import json
import os
import signal
import subprocess
import threading
import time

from . import exception

_ARG_MAX_BYTES = 2**16


class _Reader:

    def __init__(self, pipe, text, on_first_chunk=None):
        self._pipe = pipe
        self._text = text
        self._on_first_chunk = on_first_chunk

    def __iter__(self):
        while True:
            chunk = self._pipe.read1(64 * 1024)
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
            return -1
        returncode = self._popen.returncode
        return 128 - returncode if returncode < 0 else returncode

    def attach(self):
        for chunk in self.stdout:
            os.write(1, chunk.encode() if isinstance(chunk, str) else chunk)


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
        popen = subprocess.Popen(  # pylint: disable=consider-using-with
            args,
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            cwd=workdir,
            env=dict(os.environ, **(env or {})),
            start_new_session=True)
        return _Process(popen, text, timeout, drops=self.object_id == 'sb-drop')
