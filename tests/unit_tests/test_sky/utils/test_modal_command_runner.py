"""Tests for ModalCommandRunner, on a fake Modal SDK.

The exec helper and rsync run for real as local processes. Only the SDK is
fake (see fake_modal/): its Sandbox.exec runs commands on this machine.
"""
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import threading
import time
from unittest import mock

import pytest

from sky import exceptions
from sky.backends import cloud_vm_ray_backend
from sky.provision import common as provision_common
from sky.provision import instance_setup
from sky.provision import logging as provision_logging
from sky.provision import metadata_utils
from sky.provision import provisioner
from sky.utils import command_runner
from sky.utils import resources_utils

_FAKE_MODAL_DIR = pathlib.Path(__file__).parent / 'fake_modal'
_HELPER_DIR = pathlib.Path(command_runner.__file__).parent / 'modal'
_BOOTSTRAP_SCRIPT = _HELPER_DIR / 'sandbox_bootstrap.sh'

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
    return command_runner.ModalCommandRunner(('sb-live',))


def _execs(exec_log: pathlib.Path):
    return [json.loads(line) for line in exec_log.read_text().splitlines()]


def test_returns_exit_code_and_separate_outputs(runner):
    returncode, stdout, stderr = runner.run('echo out; echo err >&2; exit 7',
                                            require_outputs=True,
                                            separate_stderr=True,
                                            stream_logs=False)
    assert returncode == 7
    assert stdout == 'out\n'
    assert stderr == 'err\n'


def test_merges_stderr_into_stdout_by_default(runner):
    _, stdout, stderr = runner.run('echo out; echo err >&2',
                                   require_outputs=True,
                                   stream_logs=False)
    assert stdout == 'out\nerr\n'
    assert stderr == ''


def test_uses_binary_mode_and_handles_shell_syntax(runner, exec_log):
    command = 'echo "a  b" && echo \'c\' | tr c d; echo $((1 + 1)) > /dev/null'
    _, stdout, _ = runner.run(command, require_outputs=True, stream_logs=False)
    assert stdout == 'a  b\nd\n'
    call = _execs(exec_log)[0]
    assert call['sandbox_id'] == 'sb-live'
    assert call['text'] is False
    assert call['args'][:2] == ['/bin/bash', '-c']


def test_writes_log_file_and_streams(runner, tmp_path, capfd):
    log_path = tmp_path / 'logs' / 'run.log'
    returncode = runner.run('echo hello', log_path=str(log_path))
    assert returncode == 0
    assert 'hello' in log_path.read_text()
    assert 'hello' in capfd.readouterr().out


def test_stream_logs_false_stays_quiet(runner, tmp_path, capfd):
    log_path = tmp_path / 'run.log'
    runner.run('echo hello', log_path=str(log_path), stream_logs=False)
    assert 'hello' in log_path.read_text()
    assert 'hello' not in capfd.readouterr().out


@pytest.mark.parametrize('stream_logs', [True, False])
def test_unprocessed_stream_keeps_exit_code_and_log(runner, tmp_path, capfd,
                                                    stream_logs):
    log_path = tmp_path / 'with space' / 'run.log'
    returncode = runner.run('echo hello; exit 3',
                            log_path=str(log_path),
                            process_stream=False,
                            stream_logs=stream_logs)
    assert returncode == 3
    assert 'hello' in log_path.read_text()
    assert ('hello' in capfd.readouterr().out) == stream_logs


def test_skip_num_lines(runner, tmp_path):
    log_path = tmp_path / 'run.log'
    returncode = runner.run('echo skipped; echo kept; exit 4',
                            log_path=str(log_path),
                            process_stream=False,
                            stream_logs=False,
                            skip_num_lines=2)
    assert returncode == 4
    assert log_path.read_text() == 'kept\n'


def test_env_reaches_command_but_not_command_line(runner, exec_log):
    _, stdout, _ = runner.run('echo "$MY_SECRET"',
                              env={'MY_SECRET': 'value with spaces'},
                              require_outputs=True,
                              stream_logs=False)
    assert stdout == 'value with spaces\n'
    call = _execs(exec_log)[0]
    assert call['env'] == {'MY_SECRET': 'value with spaces'}
    assert 'value with spaces' not in ' '.join(call['args'])


def test_timeout_raises_and_sandbox_ends_the_command(runner, exec_log):
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run('sleep 60', timeout=1, stream_logs=False)
    assert time.monotonic() - start < 30
    assert _execs(exec_log)[0]['timeout'] == 6


def test_helper_handles_non_blocking_pipes(exec_log):
    # rsync can give its transport non-blocking pipes. A full stdout pipe must
    # not lose the connection, and an empty stdin must not look like EOF.
    del exec_log
    payload = os.urandom(2 * 1024 * 1024)
    stdin_read, stdin_write = os.pipe()
    stdout_read, stdout_write = os.pipe()
    os.set_blocking(stdin_read, False)
    os.set_blocking(stdout_write, False)
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        [
            sys.executable,
            str(_HELPER_DIR / 'exec_helper.py'), '--sandbox-id', 'sb-live',
            '--', 'cat'
        ],
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


def test_helper_reports_sandbox_deadline_as_124(exec_log):
    del exec_log
    proc = subprocess.run([
        sys.executable,
        str(_HELPER_DIR / 'exec_helper.py'), '--sandbox-id', 'sb-live',
        '--timeout', '1', '--', 'sleep', '60'
    ],
                          capture_output=True,
                          check=False,
                          timeout=30)
    assert proc.returncode == 124
    assert b'timed out' in proc.stderr


def test_missing_sandbox_returns_255(exec_log):
    del exec_log
    runner = command_runner.ModalCommandRunner(('sb-gone',))
    returncode, _, stderr = runner.run('true',
                                       require_outputs=True,
                                       separate_stderr=True,
                                       stream_logs=False)
    assert returncode == 255
    assert 'sb-gone' in stderr


def test_lost_connection_returns_255(exec_log):
    del exec_log
    runner = command_runner.ModalCommandRunner(('sb-drop',))
    returncode, _, stderr = runner.run('echo first; sleep 60',
                                       require_outputs=True,
                                       separate_stderr=True,
                                       stream_logs=False)
    assert returncode == 255
    assert 'lost the connection' in stderr


def test_connect_timeout_returns_255_quickly(exec_log):
    del exec_log
    runner = command_runner.ModalCommandRunner(('sb-hang',))
    start = time.monotonic()
    returncode, _, stderr = runner.run('true',
                                       connect_timeout=1,
                                       require_outputs=True,
                                       separate_stderr=True,
                                       stream_logs=False)
    assert returncode == 255
    assert 'timed out' in stderr
    assert time.monotonic() - start < 30


def test_bad_credentials_are_not_reported_as_connection_loss(exec_log):
    del exec_log
    runner = command_runner.ModalCommandRunner(('sb-unauthorized',))
    returncode, _, stderr = runner.run('true',
                                       require_outputs=True,
                                       separate_stderr=True,
                                       stream_logs=False)
    assert returncode == 1
    assert 'AuthError' in stderr


def test_missing_sdk_says_how_to_install_it(tmp_path, monkeypatch):
    # `import modal` fails with neither the fake on the path nor a real SDK.
    (tmp_path / 'modal.py').write_text('raise ImportError("no SDK here")')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    runner = command_runner.ModalCommandRunner(('sb-live',))
    returncode, _, stderr = runner.run('true',
                                       require_outputs=True,
                                       separate_stderr=True,
                                       stream_logs=False)
    assert returncode == 1
    assert 'skypilot[modal]' in stderr


def test_daemon_holding_stdout_does_not_hang_the_command(runner):
    # What `ray start` looks like from here: the command exits, but a process it
    # left behind still has stdout open.
    start = time.monotonic()
    returncode, stdout, _ = runner.run('echo started; sleep 60 &',
                                       require_outputs=True,
                                       stream_logs=False)
    assert returncode == 0
    assert 'started' in stdout
    assert time.monotonic() - start < 30


def test_keeps_output_written_just_before_exit(runner):
    _, stdout, _ = runner.run('head -c 5000000 /dev/zero | tr "\\0" x; exit 0',
                              require_outputs=True,
                              stream_logs=False)
    assert len(stdout) == 5000000


def test_interactive_mode_gives_a_live_stdin(runner):
    # With a closed stdin this `read` would return at once and print "eof".
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run('read line; echo eof',
                   ssh_mode=command_runner.SshMode.INTERACTIVE,
                   timeout=2,
                   stream_logs=False)
    assert time.monotonic() - start >= 2


def test_default_stdin_is_at_eof(runner):
    returncode, stdout, _ = runner.run('read line; echo "rc=$?"',
                                       require_outputs=True,
                                       stream_logs=False)
    assert returncode == 0
    assert stdout == 'rc=1\n'


def test_login_mode_attaches_a_pty(runner, exec_log):
    returncode, stdout, stderr = runner.run(
        ['/bin/sh', '-c', 'exit 5'], ssh_mode=command_runner.SshMode.LOGIN)
    assert (returncode, stdout, stderr) == (5, '', '')
    call = _execs(exec_log)[0]
    assert call['pty'] is True
    assert call['args'] == ['/bin/sh', '-c', 'exit 5']


def test_runner_refuses_commands_modal_would_reject(runner):
    # Everything the runner says fits must really go through; the rest is
    # uploaded by the caller instead of being inlined.
    largest_fit = ': ' + 'x' * (runner.max_inline_command_length() - 30)
    assert not runner.is_command_length_over_limit(largest_fit)
    returncode, _, _ = runner.run(largest_fit,
                                  require_outputs=True,
                                  stream_logs=False)
    assert returncode == 0
    assert runner.is_command_length_over_limit('x' * (2**16))


def test_over_limit_command_fails_without_retry_status(runner):
    returncode, _, stderr = runner.run(f'# {"x" * 2**16}',
                                       require_outputs=True,
                                       separate_stderr=True,
                                       stream_logs=False)
    assert returncode == 1
    assert 'ARG_MAX' in stderr


def test_get_remote_home_dir(runner, monkeypatch, tmp_path):
    monkeypatch.setenv('HOME', str(tmp_path))
    assert runner.get_remote_home_dir() == str(tmp_path)


def test_check_connection(runner, exec_log):
    del exec_log
    assert runner.check_connection()
    assert not command_runner.ModalCommandRunner(
        ('sb-gone',)).check_connection()


def test_port_forwarding_is_unsupported(runner):
    with pytest.raises(NotImplementedError):
        runner.port_forward_command([(1, 2)])


def test_make_runner_list_ignores_ssh_credentials():
    runners = command_runner.ModalCommandRunner.make_runner_list(
        [('sb-a',), ('sb-b',)], ssh_user='root', ssh_private_key='key')
    assert [r.node_id for r in runners] == ['sb-a', 'sb-b']


@needs_rsync
class TestRsync:
    """Real rsync on both ends, with its protocol going through the helper."""

    @staticmethod
    def _make_tree(root: pathlib.Path) -> bytes:
        payload = os.urandom(5 * 1024 * 1024)
        (root / 'sub dir').mkdir(parents=True)
        (root / 'small.txt').write_text('hello')
        (root / 'sub dir' / 'big.bin').write_bytes(payload)
        return payload

    def test_up(self, runner, tmp_path, exec_log):
        payload = self._make_tree(tmp_path / 'src')
        target = tmp_path / 'remote'
        runner.rsync(str(tmp_path / 'src'),
                     str(target),
                     up=True,
                     stream_logs=False)
        assert (target / 'small.txt').read_text() == 'hello'
        assert (target / 'sub dir' / 'big.bin').read_bytes() == payload
        # No ssh, and rsync's remote end ran through Sandbox.exec.
        assert any('rsync --server' in ' '.join(call['args'])
                   for call in _execs(exec_log))

    def test_down(self, runner, tmp_path):
        payload = self._make_tree(tmp_path / 'remote')
        target = tmp_path / 'local'
        runner.rsync(str(tmp_path / 'remote') + '/',
                     str(target),
                     up=False,
                     stream_logs=False)
        assert (target / 'sub dir' / 'big.bin').read_bytes() == payload

    def test_up_to_path_with_spaces(self, runner, tmp_path):
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / 'file').write_text('x')
        target = tmp_path / 'remote dir with spaces'
        runner.rsync(str(tmp_path / 'src'),
                     str(target),
                     up=True,
                     stream_logs=False)
        assert (target / 'file').read_text() == 'x'

    def test_single_file(self, runner, tmp_path):
        (tmp_path / 'src.txt').write_text('x')
        target = tmp_path / 'copied.txt'
        runner.rsync(str(tmp_path / 'src.txt'),
                     str(target),
                     up=True,
                     stream_logs=False)
        assert target.read_text() == 'x'

    def test_resolves_home_in_the_sandbox(self, runner, tmp_path, monkeypatch):
        home = tmp_path / 'sandbox-home'
        home.mkdir()
        monkeypatch.setenv('HOME', str(home))
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / 'file').write_text('x')
        runner.rsync(str(tmp_path / 'src'),
                     '~/target',
                     up=True,
                     stream_logs=False)
        assert (home / 'target' / 'file').read_text() == 'x'

    def test_respects_skyignore(self, runner, tmp_path):
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / '.skyignore').write_text('ignored\n')
        (tmp_path / 'src' / 'ignored').write_text('x')
        (tmp_path / 'src' / 'kept').write_text('x')
        target = tmp_path / 'remote'
        runner.rsync(str(tmp_path / 'src'),
                     str(target),
                     up=True,
                     stream_logs=False)
        assert (target / 'kept').exists()
        assert not (target / 'ignored').exists()

    def test_missing_sandbox_raises_command_error(self, exec_log, tmp_path):
        del exec_log
        (tmp_path / 'src').mkdir()
        runner = command_runner.ModalCommandRunner(('sb-gone',))
        with mock.patch('time.sleep'), pytest.raises(exceptions.CommandError):
            runner.rsync(str(tmp_path / 'src'),
                         str(tmp_path / 'remote'),
                         up=True,
                         stream_logs=False)

    def test_explains_missing_rsync_in_sandbox(self, runner, tmp_path,
                                               monkeypatch):
        (tmp_path / 'src').mkdir()
        log_path = tmp_path / 'rsync.log'
        # A sandbox without rsync, without removing it from this machine.
        real_helper = runner._HELPER_DIR  # pylint: disable=protected-access
        wrapped = tmp_path / 'helpers'
        wrapped.mkdir()
        shutil.copy(os.path.join(real_helper, 'exec_helper.py'), wrapped)
        helper_text = pathlib.Path(real_helper, 'rsync_helper.sh').read_text()
        (wrapped / 'rsync_helper.sh').write_text(
            helper_text.replace('command -v rsync', 'command -v no-such-rsync'))
        (wrapped / 'rsync_helper.sh').chmod(stat.S_IRWXU)
        monkeypatch.setattr(runner, '_HELPER_DIR', str(wrapped))
        with mock.patch('time.sleep'), pytest.raises(exceptions.CommandError):
            runner.rsync(str(tmp_path / 'src'),
                         str(tmp_path / 'remote'),
                         up=True,
                         stream_logs=False,
                         log_path=str(log_path))
        assert 'not installed' in log_path.read_text()


class TestBootstrapScript:
    """Runs the script with a stub package manager and only the tools listed."""

    _BASE_TOOLS = ('bash', 'sh', 'cat', 'chmod', 'cut', 'env', 'id', 'touch',
                   'echo')

    @pytest.fixture(name='sandbox_bin')
    def fixture_sandbox_bin(self, tmp_path) -> pathlib.Path:
        sandbox_bin = tmp_path / 'bin'
        sandbox_bin.mkdir()
        for tool in self._BASE_TOOLS:
            (sandbox_bin / tool).symlink_to(shutil.which(tool))
        return sandbox_bin

    def _add_apt(self, sandbox_bin: pathlib.Path, installs: bool) -> None:
        provides = {'pciutils': 'lspci', 'netcat-openbsd': 'nc'}
        install = ''.join(f'    {package}) touch "{sandbox_bin}/{command}"; '
                          f'chmod +x "{sandbox_bin}/{command}";;\n'
                          for package, command in provides.items())
        script = ('#!/bin/bash\n'
                  f'echo "$@" >> "{sandbox_bin.parent}/apt.log"\n'
                  '[ "$1" = install ] || exit 0\n'
                  f'{"" if installs else "exit 100"}\n'
                  'for package in "$@"; do case "$package" in\n'
                  '    -*|install) ;;\n'
                  f'{install}'
                  f'    *) touch "{sandbox_bin}/$package"; '
                  f'chmod +x "{sandbox_bin}/$package";;\n'
                  'esac; done\n')
        apt = sandbox_bin / 'apt-get'
        apt.write_text(script)
        apt.chmod(stat.S_IRWXU)

    def _run(self, sandbox_bin: pathlib.Path) -> subprocess.CompletedProcess:
        return subprocess.run(['bash', str(_BOOTSTRAP_SCRIPT)],
                              env={
                                  'PATH': str(sandbox_bin),
                                  'SKYPILOT_MODAL_SUDO_DIR': str(sandbox_bin),
                              },
                              executable=shutil.which('bash'),
                              capture_output=True,
                              text=True,
                              check=False,
                              timeout=60)

    @pytest.mark.skipif(os.getuid() != 0, reason='the sudo shim is for root')
    def test_installs_missing_tools_once(self, sandbox_bin):
        self._add_apt(sandbox_bin, installs=True)
        result = self._run(sandbox_bin)
        assert result.returncode == 0, result.stderr
        apt_log = sandbox_bin.parent / 'apt.log'
        assert 'rsync curl wget gcc patch pciutils netcat-openbsd' in (
            apt_log.read_text())
        assert (sandbox_bin / 'rsync').exists()

        apt_log.unlink()
        result = self._run(sandbox_bin)
        assert result.returncode == 0, result.stderr
        assert not apt_log.exists()

    @pytest.mark.skipif(os.getuid() != 0, reason='the sudo shim is for root')
    def test_fails_when_required_tool_cannot_be_installed(self, sandbox_bin):
        self._add_apt(sandbox_bin, installs=False)
        result = self._run(sandbox_bin)
        assert result.returncode == 1
        assert "'rsync'" in result.stderr

    def test_no_package_manager_with_tools_present_succeeds(self, sandbox_bin):
        for tool in ('rsync', 'curl', 'patch'):
            (sandbox_bin / tool).touch(mode=0o755)
        result = self._run(sandbox_bin)
        assert result.returncode == 0, result.stderr

    @pytest.mark.skipif(os.getuid() != 0, reason='the sudo shim is for root')
    @pytest.mark.parametrize('sudo_args,expected', [
        (['sh', '-c', 'echo ran'], 'ran\n'),
        (['-n', '-E', 'sh', '-c', 'echo ran'], 'ran\n'),
        (['-u', 'root', 'sh', '-c', 'echo ran'], 'ran\n'),
        (['FOO=bar', 'sh', '-c', 'echo $FOO'], 'bar\n'),
        (['--', 'sh', '-c', 'echo ran'], 'ran\n'),
        (['-l'], ''),
    ])
    def test_sudo_shim(self, sandbox_bin, sudo_args, expected):
        self._add_apt(sandbox_bin, installs=True)
        assert self._run(sandbox_bin).returncode == 0
        result = subprocess.run([str(sandbox_bin / 'sudo')] + sudo_args,
                                env={'PATH': str(sandbox_bin)},
                                capture_output=True,
                                text=True,
                                check=False,
                                timeout=10)
        assert result.returncode == 0, result.stderr
        assert result.stdout == expected


@needs_rsync
def test_setup_stages_run_through_the_runner(exec_log, tmp_path, monkeypatch):
    # The stages SkyPilot runs on a new node, end to end through the helper.
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setattr(metadata_utils, 'SKY_CLUSTER_METADATA_PATH',
                        tmp_path / 'meta')
    (tmp_path / 'local_file').write_text('mounted')
    instance = provision_common.InstanceInfo(instance_id='sb-live',
                                             internal_ip='127.0.0.1',
                                             external_ip=None,
                                             tags={})
    cluster_info = provision_common.ClusterInfo(
        instances={'sb-live': [instance]},
        head_instance_id='sb-live',
        provider_name='modal')
    runners = command_runner.ModalCommandRunner.make_runner_list([('sb-live',)])

    with provision_logging.setup_provision_logging(str(tmp_path / 'logs')), \
            mock.patch.object(instance_setup.provision,
                              'get_command_runners',
                              return_value=runners):
        instance_setup.internal_file_mounts(
            'cluster',
            {str(tmp_path / 'mounted'): str(tmp_path / 'local_file')},
            cluster_info, {})
        instance_setup.setup_runtime_on_cluster(
            'cluster', ['echo "$OMP_NUM_THREADS" > ~/setup_ran'], cluster_info,
            {})
        num_execs = len(_execs(exec_log))
        # A repeated launch with the same setup is cached on this side.
        instance_setup.setup_runtime_on_cluster(
            'cluster', ['echo "$OMP_NUM_THREADS" > ~/setup_ran'], cluster_info,
            {})

    assert (tmp_path / 'mounted').read_text() == 'mounted'
    assert (home / 'setup_ran').read_text() == '1\n'
    assert len(_execs(exec_log)) == num_execs


@pytest.mark.parametrize('cloud_name,uses_ssh', [('Modal', False),
                                                 ('Daytona', True)])
def test_provisioner_bootstraps_instead_of_waiting_for_ssh(
        tmp_path, cloud_name, uses_ssh):

    class _StopAfterTransportStep(Exception):
        pass

    head = provision_common.InstanceInfo(instance_id='sb-live',
                                         internal_ip='127.0.0.1',
                                         external_ip=None,
                                         tags={})
    cluster_info = provision_common.ClusterInfo(
        instances={'sb-live': [head]},
        head_instance_id='sb-live',
        provider_name=cloud_name.lower())
    record = provision_common.ProvisionRecord(provider_name=cloud_name.lower(),
                                              region='region',
                                              zone=None,
                                              cluster_name='cluster-on-cloud',
                                              head_instance_id='sb-live',
                                              resumed_instance_ids=[],
                                              created_instance_ids=['sb-live'])
    resources = mock.MagicMock()
    resources.cloud.__repr__ = lambda _: cloud_name
    with provision_logging.setup_provision_logging(str(tmp_path)), \
            mock.patch.object(provisioner.global_user_state,
                              'get_cluster_yaml_dict',
                              return_value={}), \
            mock.patch.object(provisioner.global_user_state,
                              'get_handle_from_cluster_name'), \
            mock.patch.object(provisioner.global_user_state,
                              'update_cluster_handle'), \
            mock.patch.object(provisioner.provision,
                              'get_cluster_info',
                              return_value=cluster_info), \
            mock.patch.object(provisioner.backend_utils,
                              'ssh_credential_from_yaml',
                              return_value={}), \
            mock.patch.object(provisioner,
                              'wait_for_ssh',
                              side_effect=_StopAfterTransportStep) as wait, \
            mock.patch.object(provisioner.instance_setup,
                              'bootstrap_exec_transport_nodes',
                              side_effect=_StopAfterTransportStep) as bootstrap:
        with pytest.raises(_StopAfterTransportStep):
            # pylint: disable-next=protected-access
            provisioner._post_provision_setup(
                resources,
                resources_utils.ClusterName('cluster', 'cluster-on-cloud'),
                'cluster.yaml', record, None)
    assert wait.called == uses_ssh
    assert bootstrap.called != uses_ssh


def test_bootstrap_failure_raises(runner):
    with mock.patch.object(runner,
                           'run',
                           return_value=(1, '', 'could not install')):
        with pytest.raises(exceptions.CommandError) as error:
            runner.bootstrap_sandbox()
    assert error.value.returncode == 1
    assert 'could not install' in error.value.detailed_reason


def test_bootstrap_of_cluster_retries_only_connection_failures(runner):
    cluster_info = mock.MagicMock(provider_name='modal')
    with mock.patch.object(
            instance_setup.provision, 'get_command_runners',
            return_value=[runner]), mock.patch('time.sleep'), mock.patch.object(
                runner, 'run', side_effect=[(255, '', ''), (0, '', '')]) as run:
        instance_setup.bootstrap_exec_transport_nodes(cluster_info, {})
    assert run.call_count == 2

    with mock.patch.object(
            instance_setup.provision, 'get_command_runners',
            return_value=[runner]), mock.patch('time.sleep'), mock.patch.object(
                runner, 'run', return_value=(1, '', '')) as run:
        with pytest.raises(exceptions.CommandError):
            instance_setup.bootstrap_exec_transport_nodes(cluster_info, {})
    assert run.call_count == 1


@pytest.mark.parametrize('cloud_repr,expected', [('Modal', False),
                                                 ('Kubernetes', True)])
def test_grpc_is_disabled_for_exec_transport_clouds(cloud_repr, expected):
    handle = cloud_vm_ray_backend.CloudVmRayResourceHandle.__new__(
        cloud_vm_ray_backend.CloudVmRayResourceHandle)
    handle.is_grpc_enabled = True
    cloud = mock.MagicMock()
    cloud.__repr__ = lambda _: cloud_repr
    handle.launched_resources = mock.MagicMock(cloud=cloud)
    with mock.patch.object(cloud_vm_ray_backend.env_options.Options,
                           'get',
                           return_value=True):
        assert handle.is_grpc_enabled_with_flag is expected
