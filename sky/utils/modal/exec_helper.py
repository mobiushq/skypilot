"""Runs one command in a Modal sandbox, behaving like `kubectl exec`.

`ModalCommandRunner` starts this script as a local subprocess so that
`log_lib.run_with_log` and `rsync -e` can treat it like `ssh`: the remote stdout
and stderr arrive on this process's stdout and stderr as raw bytes, stdin is
forwarded, and the exit status is the remote command's.

This script runs as `python <path>`, so it must not import `sky`.

Exit statuses other than the remote command's:
    255: the sandbox could not be reached or the connection was lost. SkyPilot
        treats this the same as ssh's 255, as a retryable connection failure.
    124: --timeout elapsed and the sandbox ended the command.
    1: a failure that retrying cannot fix (no SDK, bad credentials, a command
        over Modal's argument limit).
"""
import argparse
import json
import os
import select
import sys
import threading
import time
from typing import Any, Dict, List, Optional

ENV_VAR = 'SKYPILOT_MODAL_EXEC_ENV'
EXIT_CONNECTION_LOST = 255
EXIT_TIMED_OUT = 124
EXIT_PERMANENT_FAILURE = 1

_CHUNK_BYTES = 64 * 1024
_POLL_SECONDS = 0.05
# A daemon started by the command (e.g. `ray start`) can inherit stdout and keep
# the stream open after the command has exited. Once it has exited, stop reading
# after this many seconds with no new output.
_OUTPUT_IDLE_SECONDS_AFTER_EXIT = 5.0
# How long to wait for the exit status after a stream has failed.
_EXIT_WAIT_SECONDS_AFTER_ERROR = 5.0


def _log(message: str) -> None:
    os.write(2, f'modal exec: {message}\n'.encode('utf-8', 'replace'))


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            view = view[os.write(fd, view):]
        except BlockingIOError:
            # rsync can hand its transport non-blocking pipes.
            select.select([], [fd], [])


def _read_stdin() -> bytes:
    """Returns the next bytes from stdin, or b'' at EOF."""
    while True:
        try:
            return os.read(0, _CHUNK_BYTES)
        except BlockingIOError:
            # See _write_all. This is not EOF: ending stdin here would end
            # rsync's remote side in the middle of the transfer.
            select.select([0], [], [])
        except OSError:
            return b''


def _exit_status_for(error: BaseException) -> int:
    """Maps an SDK error to an exit status."""
    # pylint: disable-next=import-outside-toplevel
    from modal import exception as modal_exception
    permanent = (modal_exception.InvalidError, modal_exception.AuthError,
                 modal_exception.PermissionDeniedError)
    if isinstance(error, permanent):
        return EXIT_PERMANENT_FAILURE
    return EXIT_CONNECTION_LOST


class _Session:
    """Moves bytes between this process and one remote process."""

    def __init__(self, process: Any) -> None:
        self._process = process
        self._last_output = time.monotonic()
        self.error: Optional[BaseException] = None
        self.exit_status: Optional[int] = None

    def _pump_output(self, stream: Any, fd: int) -> None:
        try:
            for chunk in stream:
                self._last_output = time.monotonic()
                if isinstance(chunk, str):
                    chunk = chunk.encode('utf-8')
                _write_all(fd, chunk)
        except BrokenPipeError:
            # Our reader went away, e.g. `| head`.
            pass
        except Exception as e:  # pylint: disable=broad-except
            self.error = e

    def _pump_stdin(self) -> None:
        stdin = self._process.stdin
        try:
            while True:
                data = _read_stdin()
                if not data:
                    stdin.write_eof()
                    stdin.drain()
                    return
                stdin.write(data)
                stdin.drain()
        except Exception as e:  # pylint: disable=broad-except
            # Writing to a process that just exited is not a failure.
            if self.exit_status is None:
                self.error = e

    def _wait(self) -> None:
        try:
            self.exit_status = self._process.wait()
        except Exception as e:  # pylint: disable=broad-except
            self.error = e

    def run(self) -> Optional[int]:
        """Returns the remote exit status, or None if it never arrived."""
        outputs = [
            threading.Thread(target=self._pump_output,
                             args=(self._process.stdout, 1),
                             daemon=True),
            threading.Thread(target=self._pump_output,
                             args=(self._process.stderr, 2),
                             daemon=True),
        ]
        for thread in outputs:
            thread.start()
        threading.Thread(target=self._pump_stdin, daemon=True).start()
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
        if self.exit_status is None:
            return None

        while any(thread.is_alive() for thread in outputs):
            idle = time.monotonic() - self._last_output
            if idle > _OUTPUT_IDLE_SECONDS_AFTER_EXIT:
                break
            time.sleep(_POLL_SECONDS)
        return self.exit_status


def _start_process(args: argparse.Namespace, modal: Any) -> Any:
    """Starts the remote process, or exits if it takes too long."""
    kwargs: Dict[str, Any] = {'text': False}
    if args.timeout is not None:
        kwargs['timeout'] = args.timeout
    env = json.loads(os.environ.get(ENV_VAR, 'null'))
    if env:
        kwargs['env'] = env
    if args.pty:
        kwargs['pty'] = True

    result: Dict[str, Any] = {}

    def _start() -> None:
        try:
            sandbox = modal.Sandbox.from_id(args.sandbox_id)
            result['process'] = sandbox.exec(*args.command, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            result['error'] = e

    # Not abandoned by the SDK on its own: from_id retries for a long time when
    # the network is down, and callers expect ssh's ConnectTimeout behavior.
    starter = threading.Thread(target=_start, daemon=True)
    starter.start()
    starter.join(args.connect_timeout)
    if starter.is_alive():
        _log(f'timed out after {args.connect_timeout}s connecting to sandbox '
             f'{args.sandbox_id}.')
        _exit(EXIT_CONNECTION_LOST)
    if 'error' in result:
        error = result['error']
        _log(f'could not run a command in sandbox {args.sandbox_id}: '
             f'{type(error).__name__}: {error}')
        _exit(_exit_status_for(error))
    return result['process']


def _exit(status: int) -> None:
    # The SDK's background threads must not delay or change the exit status.
    os._exit(status)  # pylint: disable=protected-access


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     usage='%(prog)s [options] -- command...')
    parser.add_argument('--sandbox-id', required=True)
    parser.add_argument('--connect-timeout', type=float, default=30)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--pty', action='store_true')
    # Split by hand: everything after the first `--` belongs to the command,
    # which may itself contain `--`.
    if '--' not in argv or argv[-1] == '--':
        parser.error('no command given')
    separator = argv.index('--')
    args = parser.parse_args(argv[:separator])
    args.command = argv[separator + 1:]
    return args


def main(argv: List[str]) -> None:
    args = _parse_args(argv)
    try:
        # pylint: disable-next=import-outside-toplevel
        import modal
    except ImportError as e:
        _log(f'the Modal SDK is not available to {sys.executable}: {e}. '
             'Install it with: pip install "skypilot[modal]"')
        _exit(EXIT_PERMANENT_FAILURE)

    process = _start_process(args, modal)
    try:
        if args.pty:
            process.attach()
            status = process.wait()
        else:
            session = _Session(process)
            status = session.run()
            if status is None:
                _log(f'lost the connection to sandbox {args.sandbox_id}: '
                     f'{session.error!r}')
                _exit(EXIT_CONNECTION_LOST)
    except KeyboardInterrupt:
        _exit(130)
    except Exception as e:  # pylint: disable=broad-except
        _log(f'lost the connection to sandbox {args.sandbox_id}: {e!r}')
        _exit(EXIT_CONNECTION_LOST)

    # The SDK reports -1 when the exec's deadline passed.
    if status < 0:
        if args.timeout is not None:
            _log(f'command timed out after {args.timeout}s.')
            _exit(EXIT_TIMED_OUT)
        _exit(EXIT_CONNECTION_LOST)
    _exit(status)


if __name__ == '__main__':
    main(sys.argv[1:])
