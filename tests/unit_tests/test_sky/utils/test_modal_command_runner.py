"""Failure handling of ModalCommandRunner and its exec helper.

The helper and rsync run for real as local processes. Only the SDK is fake (see
fake_modal/): its Sandbox.exec runs commands on this machine.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

import pytest

from sky import exceptions
from sky.utils import command_runner

_FAKE_MODAL_DIR = pathlib.Path(__file__).parent / 'fake_modal'

needs_rsync = pytest.mark.skipif(shutil.which('rsync') is None,
                                 reason='rsync is not installed')


@pytest.fixture(name='exec_log')
def fixture_exec_log(tmp_path, monkeypatch) -> pathlib.Path:
    log = tmp_path / 'exec.log'
    log.touch()
    monkeypatch.setenv('PYTHONPATH', str(_FAKE_MODAL_DIR))
    monkeypatch.setenv('FAKE_MODAL_LOG', str(log))
    return log


@pytest.fixture(name='runner')
def fixture_runner(exec_log) -> command_runner.ModalCommandRunner:
    del exec_log
    return command_runner.ModalCommandRunner(('modal', 'sb-live'))


def _run(sandbox_id: str, command: str, **kwargs):
    return command_runner.ModalCommandRunner(
        ('modal', sandbox_id)).run(command,
                                   require_outputs=True,
                                   separate_stderr=True,
                                   stream_logs=False,
                                   **kwargs)


def _helper(*args: str, **kwargs) -> subprocess.Popen:
    # pylint: disable-next=consider-using-with
    return subprocess.Popen(
        [sys.executable, command_runner.MODAL_EXEC_HELPER, *args], **kwargs)


def test_helper_handles_non_blocking_pipes(exec_log):
    # rsync gives its remote shell non-blocking pipes. A full stdout pipe must
    # not lose output, and an empty stdin must not look like EOF.
    del exec_log
    payload = os.urandom(2 * 1024 * 1024)
    stdin_read, stdin_write = os.pipe()
    stdout_read, stdout_write = os.pipe()
    os.set_blocking(stdin_read, False)
    os.set_blocking(stdout_write, False)
    proc = _helper('sb-live',
                   'cat',
                   stdin=stdin_read,
                   stdout=stdout_write,
                   stderr=subprocess.PIPE)
    os.close(stdin_read)
    os.close(stdout_write)

    def _feed():
        time.sleep(1)  # Long enough for the helper to find stdin empty.
        with os.fdopen(stdin_write, 'wb') as f:
            f.write(payload)

    feeder = threading.Thread(target=_feed)
    feeder.start()
    time.sleep(2)  # Let stdout's pipe fill up before reading any of it.
    with os.fdopen(stdout_read, 'rb') as f:
        received = f.read()
    feeder.join()
    assert proc.wait(timeout=30) == 0, proc.stderr.read()
    proc.stderr.close()
    assert received == payload


def test_helper_keeps_output_for_a_slow_reader(exec_log):
    # The command exits with output undelivered, and the local reader then
    # pauses for longer than the helper waits on an idle stream.
    del exec_log
    size = 1024 * 1024
    proc = _helper('sb-live',
                   f'head -c {size} /dev/zero',
                   stdin=subprocess.DEVNULL,
                   stdout=subprocess.PIPE)
    time.sleep(12)
    received = proc.stdout.read()
    proc.stdout.close()
    assert proc.wait(timeout=30) == 0
    assert len(received) == size


def test_daemon_holding_stdout_does_not_hang_the_command(runner):
    # What `ray start` can look like from here: the command exits, but a process
    # it left behind still has stdout open.
    start = time.monotonic()
    returncode, stdout, _ = runner.run('echo started; sleep 60 &',
                                       require_outputs=True,
                                       stream_logs=False)
    assert returncode == 0
    assert 'started' in stdout
    assert time.monotonic() - start < 40


def test_keeps_all_output_and_exit_code(runner):
    returncode, stdout, _ = runner.run(
        'head -c 5000000 /dev/zero | tr "\\0" x; exit 7',
        require_outputs=True,
        stream_logs=False)
    assert returncode == 7
    assert len(stdout) == 5000000


def test_lost_connection_returns_255(exec_log):
    del exec_log
    returncode, _, stderr = _run('sb-drop', 'echo first; sleep 60')
    assert returncode == 255
    assert 'Lost the connection' in stderr


def test_connect_timeout_returns_255_quickly(exec_log):
    del exec_log
    start = time.monotonic()
    returncode, _, stderr = _run('sb-hang', 'true', connect_timeout=1)
    assert returncode == 255
    assert 'timed out' in stderr
    assert time.monotonic() - start < 30


def test_missing_sandbox_returns_255(exec_log):
    del exec_log
    assert _run('sb-gone', 'true')[0] == 255


def test_bad_credentials_are_not_retried_as_connection_loss(exec_log):
    del exec_log
    returncode, _, stderr = _run('sb-unauthorized', 'true')
    assert returncode == 1
    assert 'bad token' in stderr


def test_missing_sdk_says_how_to_install_it(tmp_path, monkeypatch):
    (tmp_path / 'modal.py').write_text('raise ImportError("no SDK here")')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    returncode, _, stderr = _run('sb-live', 'true')
    assert returncode == 1
    assert 'skypilot[modal]' in stderr


def test_timeout_raises_and_sandbox_ends_the_command(runner, exec_log):
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run('sleep 60', timeout=1, stream_logs=False)
    assert time.monotonic() - start < 30
    calls = [json.loads(line) for line in exec_log.read_text().splitlines()]
    assert calls[0]['timeout'] == 6


def test_helper_reports_sandbox_deadline_as_124(exec_log):
    del exec_log
    proc = _helper('--timeout',
                   '1',
                   'sb-live',
                   'sleep 60',
                   stdin=subprocess.DEVNULL)
    assert proc.wait(timeout=30) == 124


def test_every_command_under_the_limit_fits_in_modal_args(runner):
    largest_fit = ': ' + 'x' * (runner.max_inline_command_length() - 30)
    assert not runner.is_command_length_over_limit(largest_fit)
    returncode, _, _ = runner.run(largest_fit,
                                  require_outputs=True,
                                  stream_logs=False)
    assert returncode == 0
    assert runner.is_command_length_over_limit('x' * (2**16))


def test_over_limit_command_is_not_retried_as_connection_loss(exec_log):
    del exec_log
    returncode, _, stderr = _run('sb-live', f': {"x" * 2**16}')
    assert returncode == 1
    assert 'ARG_MAX' in stderr


@needs_rsync
class TestRsync:
    """Real rsync on both ends, with its protocol going through the helper."""

    @staticmethod
    def _make_tree(root: pathlib.Path) -> bytes:
        payload = os.urandom(5 * 1024 * 1024)
        (root / 'sub dir').mkdir(parents=True)
        (root / 'sub dir' / 'big.bin').write_bytes(payload)
        return payload

    def test_large_file_up(self, runner, tmp_path):
        payload = self._make_tree(tmp_path / 'src')
        target = tmp_path / 'remote dir'
        runner.rsync(str(tmp_path / 'src'),
                     str(target),
                     up=True,
                     stream_logs=False)
        assert (target / 'sub dir' / 'big.bin').read_bytes() == payload

    def test_large_file_down(self, runner, tmp_path):
        payload = self._make_tree(tmp_path / 'remote')
        target = tmp_path / 'local'
        runner.rsync(str(tmp_path / 'remote') + '/',
                     str(target),
                     up=False,
                     stream_logs=False)
        assert (target / 'sub dir' / 'big.bin').read_bytes() == payload

    def test_missing_sandbox_raises_command_error(self, exec_log, tmp_path,
                                                  monkeypatch):
        del exec_log
        monkeypatch.setattr(time, 'sleep', lambda _: None)
        (tmp_path / 'src').mkdir()
        runner = command_runner.ModalCommandRunner(('modal', 'sb-gone'))
        with pytest.raises(exceptions.CommandError):
            runner.rsync(str(tmp_path / 'src'),
                         str(tmp_path / 'remote'),
                         up=True,
                         stream_logs=False)
