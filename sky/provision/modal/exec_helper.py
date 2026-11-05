"""Runs a shell command in a Modal sandbox and pipes stdin, stdout, stderr.

Used by ModalCommandRunner for commands, and as rsync's remote shell, the
same way `ssh` is used for VMs:

    python exec_helper.py [options] <sandbox_id> <command words...>
    python exec_helper.py -l <user> <sandbox_id> <command words...>  # rsync

Options, which come first:
    --connect-timeout <seconds>: give up on reaching the sandbox after this.
    --timeout <seconds>: have the sandbox end the command after this.

As with ssh, the command words are joined with spaces and run by the remote
shell (`/bin/bash -c`). The exit code is the remote command's, or:
    255: the sandbox cannot be reached or the connection was lost. SkyPilot
        retries this, as it does for ssh.
    124: --timeout passed.
    1: a failure that retrying cannot fix, such as bad credentials or a
        command over Modal's argument limit.

This file runs as a script and imports only `modal`, so it stays fast.
"""
import os
import select
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

_UNREACHABLE = 255
_TIMED_OUT = 124
_PERMANENT_FAILURE = 1
_PERMANENT_ERRORS = ('AuthError', 'ConflictError', 'InvalidError',
                     'PermissionDeniedError')
_DEFAULT_CONNECT_TIMEOUT_SECONDS = 60.0

_POLL_SECONDS = 0.05
# A daemon started by the command (e.g. `ray start`) can inherit stdout and keep
# the stream open after the command has exited. Once it has exited, stop reading
# after this many seconds with no new output.
_OUTPUT_IDLE_SECONDS_AFTER_EXIT = 10.0
# How long to wait for the exit code after a stream has failed.
_EXIT_WAIT_SECONDS_AFTER_ERROR = 5.0


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            view = view[os.write(fd, view):]
        except BlockingIOError:
            # rsync gives its remote shell non-blocking pipes.
            select.select([], [fd], [])


def _read_stdin(fd: int) -> bytes:
    """Returns the next bytes from stdin, or b'' at EOF."""
    while True:
        try:
            return os.read(fd, 1 << 16)
        except BlockingIOError:
            # See _write_all. This is not EOF: ending stdin here would end
            # rsync's remote side in the middle of the transfer.
            select.select([fd], [], [])


class _Session:
    """Moves bytes between this process and one remote process."""

    def __init__(self, proc: Any) -> None:
        self._proc = proc
        self._lock = threading.Lock()
        self._last_output = time.monotonic()
        # Output waiting on a slow local reader is not idle, so must not be cut.
        self._num_writing = 0
        self.error: Optional[BaseException] = None
        self.returncode: Optional[int] = None

    def _pump_stdin(self, fd: int) -> None:
        # Reads the raw descriptor: a daemon thread blocked in sys.stdin's
        # buffered reader would hold its lock and crash interpreter shutdown.
        stdin = self._proc.stdin
        try:
            try:
                while True:
                    data = _read_stdin(fd)
                    if not data:
                        break
                    stdin.write(data)
                    stdin.drain()
            except OSError:
                pass  # Our stdin was closed, which is the same as EOF.
            stdin.write_eof()
            stdin.drain()
        except Exception as e:  # pylint: disable=broad-except
            # Writing to a process that just exited is not a failure.
            if self.returncode is None:
                self.error = e

    def _pump_output(self, stream: Any, fd: int) -> None:
        try:
            for chunk in stream:
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                with self._lock:
                    self._num_writing += 1
                try:
                    _write_all(fd, chunk)
                finally:
                    with self._lock:
                        self._num_writing -= 1
                        self._last_output = time.monotonic()
        except BrokenPipeError:
            pass  # Our reader went away, e.g. `| head`.
        except Exception as e:  # pylint: disable=broad-except
            self.error = e

    def _wait(self) -> None:
        try:
            self.returncode = self._proc.wait()
        except Exception as e:  # pylint: disable=broad-except
            self.error = e

    def run(self, stdin_fd: int, stdout_fd: int,
            stderr_fd: int) -> Optional[int]:
        """Returns the remote exit code, or None if it never arrived."""
        outputs = [
            threading.Thread(target=self._pump_output,
                             args=(self._proc.stdout, stdout_fd),
                             daemon=True),
            threading.Thread(target=self._pump_output,
                             args=(self._proc.stderr, stderr_fd),
                             daemon=True),
        ]
        for thread in outputs:
            thread.start()
        threading.Thread(target=self._pump_stdin, args=(stdin_fd,),
                         daemon=True).start()
        waiter = threading.Thread(target=self._wait, daemon=True)
        waiter.start()

        error_deadline = None
        while waiter.is_alive():
            time.sleep(_POLL_SECONDS)
            if self.error is not None:
                if error_deadline is None:
                    error_deadline = (time.monotonic() +
                                      _EXIT_WAIT_SECONDS_AFTER_ERROR)
                elif time.monotonic() > error_deadline:
                    return None
        if self.returncode is None:
            return None

        while any(thread.is_alive() for thread in outputs):
            with self._lock:
                idle = time.monotonic() - self._last_output
                if (not self._num_writing and
                        idle > _OUTPUT_IDLE_SECONDS_AFTER_EXIT):
                    break
            time.sleep(_POLL_SECONDS)
        return self.returncode


def parse_args(argv):
    """Returns (sandbox_id, command) from the helper's arguments."""
    args = list(argv)
    if args[:1] == ['-l']:
        args = args[2:]
    if len(args) < 2:
        raise ValueError('usage: exec_helper.py [-l user] <sandbox_id> '
                         '<command...>')
    sandbox_id = args[0].rsplit('@', 1)[-1]
    return sandbox_id, ' '.join(args[1:])


def _pop_options(argv: List[str]) -> Tuple[Dict[str, float], List[str]]:
    """Returns the leading options, and the arguments after them."""
    names = {'--connect-timeout': 'connect_timeout', '--timeout': 'timeout'}
    options: Dict[str, float] = {}
    args = list(argv)
    while args and args[0] in names:
        options[names[args[0]]] = float(args[1])
        args = args[2:]
    return options, args


def _start(modal_module: Any, sandbox_id: str, command: str,
           options: Dict[str, float]) -> Any:
    """Returns the remote process.

    Raises:
        TimeoutError: The sandbox was not reached in time. from_id and exec
            keep retrying for a long time when the network is down, and callers
            expect a connect timeout to behave as it does with ssh.
    """
    kwargs: Dict[str, Any] = {'text': False}
    if 'timeout' in options:
        kwargs['timeout'] = int(options['timeout'])
    result: Dict[str, Any] = {}

    def _connect() -> None:
        try:
            sandbox = modal_module.Sandbox.from_id(sandbox_id)
            result['proc'] = sandbox.exec('/bin/bash', '-c', command, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            result['error'] = e

    connect_timeout = options.get('connect_timeout',
                                  _DEFAULT_CONNECT_TIMEOUT_SECONDS)
    connector = threading.Thread(target=_connect, daemon=True)
    connector.start()
    connector.join(connect_timeout)
    if connector.is_alive():
        raise TimeoutError(f'timed out after {connect_timeout:g}s')
    if 'error' in result:
        raise result['error']
    return result['proc']


def main(argv, modal_module=None) -> int:
    options, argv = _pop_options(argv)
    try:
        sandbox_id, command = parse_args(argv)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if modal_module is None:
        # Imported here so tests can pass a stand-in without installing it.
        try:
            import modal  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            print(
                f'The Modal SDK is not available to {sys.executable}: {e}. '
                'Install it with: pip install "skypilot[modal]"',
                file=sys.stderr)
            return _PERMANENT_FAILURE
        modal_module = modal
    try:
        proc = _start(modal_module, sandbox_id, command, options)
    except Exception as e:  # pylint: disable=broad-except
        print(f'Cannot reach Modal sandbox {sandbox_id}: {e}', file=sys.stderr)
        if type(e).__name__ in _PERMANENT_ERRORS:
            return _PERMANENT_FAILURE
        return _UNREACHABLE

    session = _Session(proc)
    returncode = session.run(sys.stdin.fileno(), sys.stdout.fileno(),
                             sys.stderr.fileno())
    if returncode is None:
        print(
            f'Lost the connection to Modal sandbox {sandbox_id}: '
            f'{session.error!r}',
            file=sys.stderr)
        return _UNREACHABLE
    # The SDK reports -1 when the exec's deadline has passed.
    if returncode < 0:
        return _TIMED_OUT if 'timeout' in options else _UNREACHABLE
    return returncode


if __name__ == '__main__':
    # Running as a script puts this directory first on sys.path; drop it so
    # its modules (config.py, utils.py, ...) cannot shadow `modal`'s imports.
    if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(
            os.path.abspath(__file__)):
        sys.path.pop(0)
    code = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter shutdown, which would wait on the SDK and the stdin
    # thread.
    os._exit(code)  # pylint: disable=protected-access
