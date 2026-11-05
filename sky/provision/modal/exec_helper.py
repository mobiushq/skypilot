"""Runs a shell command in a Modal sandbox and pipes stdin, stdout, stderr.

Used by ModalCommandRunner for commands, and as rsync's remote shell, the
same way `ssh` is used for VMs:

    python exec_helper.py <sandbox_id> <command words...>
    python exec_helper.py -l <user> <sandbox_id> <command words...>  # rsync

As with ssh, the command words are joined with spaces and run by the remote
shell (`/bin/bash -c`). The exit code is the remote command's, or 255 if the
sandbox cannot be reached.

This file runs as a script and imports only `modal`, so it stays fast.
"""
import os
import sys
import threading

_UNREACHABLE = 255


def _pump_stdin(proc) -> None:
    # Reads the raw descriptor: a daemon thread blocked in sys.stdin's
    # buffered reader would hold its lock and crash interpreter shutdown.
    fd = sys.stdin.fileno()
    try:
        while True:
            data = os.read(fd, 1 << 16)
            if not data:
                break
            proc.stdin.write(data)
            proc.stdin.drain()
    except (OSError, ValueError):
        pass
    finally:
        try:
            proc.stdin.write_eof()
            proc.stdin.drain()
        except Exception:  # pylint: disable=broad-except
            pass


def _pump_output(stream, out) -> None:
    for chunk in stream:
        if isinstance(chunk, str):
            chunk = chunk.encode()
        out.write(chunk)
        out.flush()


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


def main(argv, modal_module=None) -> int:
    try:
        sandbox_id, command = parse_args(argv)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if modal_module is None:
        # Imported here so tests can pass a stand-in without installing it.
        import modal  # pylint: disable=import-outside-toplevel
        modal_module = modal
    try:
        sandbox = modal_module.Sandbox.from_id(sandbox_id)
        proc = sandbox.exec('/bin/bash', '-c', command, text=False)
    except Exception as e:  # pylint: disable=broad-except
        print(f'Cannot reach Modal sandbox {sandbox_id}: {e}', file=sys.stderr)
        return _UNREACHABLE
    threading.Thread(target=_pump_stdin, args=(proc,), daemon=True).start()
    outputs = [
        threading.Thread(target=_pump_output,
                         args=(proc.stdout, sys.stdout.buffer)),
        threading.Thread(target=_pump_output,
                         args=(proc.stderr, sys.stderr.buffer)),
    ]
    for t in outputs:
        t.start()
    for t in outputs:
        t.join()
    returncode = proc.wait()
    return _UNREACHABLE if returncode is None else returncode


if __name__ == '__main__':
    # Running as a script puts this directory first on sys.path; drop it so
    # its modules (config.py, utils.py, ...) cannot shadow `modal`'s imports.
    if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(
            os.path.abspath(__file__)):
        sys.path.pop(0)
    code = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter shutdown, which would wait on the stdin thread.
    os._exit(code)  # pylint: disable=protected-access
