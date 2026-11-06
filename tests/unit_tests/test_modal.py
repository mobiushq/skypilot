"""Tests for the Modal cloud provider."""

import inspect
import os
import shutil
import subprocess
import sys
import textwrap
from unittest import mock

import pytest

from sky import clouds
from sky import resources as resources_lib
from sky.adaptors import modal as modal_adaptor
from sky.backends import cloud_vm_ray_backend
from sky.catalog import common as catalog_common
from sky.clouds import modal as modal_cloud
from sky.data import mounting_utils
from sky.data import storage as storage_lib
from sky.provision import common as provision_common
from sky.provision.modal import exec_helper
from sky.provision.modal import instance as modal_instance
from sky.provision.modal import utils as modal_utils
from sky.utils import command_runner
from sky.utils import registry
from sky.utils import status_lib

try:
    import modal as real_modal
except ImportError:  # pragma: no cover
    real_modal = None


class _NotFoundError(Exception):
    pass


class _AlreadyExistsError(Exception):
    pass


class _Process:
    """A finished `Sandbox.exec` process."""

    def __init__(self, returncode):
        self.returncode = None
        self._returncode = returncode

    def wait(self):
        self.returncode = self._returncode


class FakeSandbox:

    def __init__(self, fake, sandbox_id, tags, kwargs):
        self._fake = fake
        self.object_id = sandbox_id
        self.tags = dict(tags)
        self.kwargs = kwargs
        self.exit_code = None
        self.execs = []

    def poll(self):
        return self.exit_code

    def exec(self, *args, timeout=None):
        self.execs.append((args, timeout))
        if self.exit_code is not None or self._fake.probe is None:
            raise RuntimeError('sandbox exited')
        return _Process(self._fake.probe)

    def terminate(self):
        if self._fake.sandboxes.get(self.object_id) is None:
            raise _NotFoundError(self.object_id)
        self.exit_code = 137


class FakeImage:

    def __init__(self, tag):
        self.tag = tag
        self.commands = []

    def run_commands(self, *commands):
        self.commands.extend(commands)
        return self


class FakeModal:
    """The parts of the `modal` SDK the provisioner uses."""

    def __init__(self):
        self.sandboxes = {}
        self.apps = {}
        self.create_calls = []
        self.probe = 0  # the start probe's exit code; None: exec fails
        self.token = {}
        fake = self

        class App:

            @staticmethod
            def lookup(name, environment_name=None, create_if_missing=False):
                key = (name, environment_name)
                if key not in fake.apps:
                    if not create_if_missing:
                        raise _NotFoundError(name)
                    fake.apps[key] = mock.Mock(app_id=f'ap-{len(fake.apps)}')
                return fake.apps[key]

        class Image:

            @staticmethod
            def from_registry(tag):
                return FakeImage(tag)

        class Sandbox:

            @staticmethod
            def create(*args, **kwargs):
                if real_modal is not None:
                    inspect.signature(real_modal.Sandbox.create).bind(
                        *args, **kwargs)
                fake.create_calls.append((args, kwargs))
                if any(sb.app_id == kwargs['app'].app_id and sb.kwargs.get(
                        'name') == kwargs.get('name') and sb.exit_code is None
                       for sb in fake.sandboxes.values()):
                    raise _AlreadyExistsError(kwargs.get('name'))
                sandbox_id = f'sb-{len(fake.sandboxes) + 1}'
                sandbox = FakeSandbox(fake, sandbox_id, kwargs['tags'], kwargs)
                sandbox.app_id = kwargs['app'].app_id
                fake.sandboxes[sandbox_id] = sandbox
                return sandbox

            @staticmethod
            def from_name(app_name, name, environment_name=None):
                if real_modal is not None:
                    inspect.signature(real_modal.Sandbox.from_name).bind(
                        app_name, name, environment_name=environment_name)
                app = fake.apps[(app_name, environment_name)]
                for sandbox in fake.sandboxes.values():
                    if (sandbox.app_id == app.app_id and
                            sandbox.kwargs.get('name') == name and
                            sandbox.exit_code is None):
                        return sandbox
                raise _NotFoundError(name)

            @staticmethod
            def list(app_id=None, tags=None):
                for sandbox in fake.sandboxes.values():
                    if sandbox.exit_code is not None:
                        continue  # Modal lists running sandboxes only.
                    if app_id is not None and sandbox.app_id != app_id:
                        continue
                    if all(
                            sandbox.tags.get(k) == v
                            for k, v in (tags or {}).items()):
                        yield sandbox

        self.App = App
        self.Image = Image
        self.Sandbox = Sandbox
        self.exception = mock.Mock(NotFoundError=_NotFoundError,
                                   AlreadyExistsError=_AlreadyExistsError)
        self.config = mock.Mock(config=mock.Mock(get=fake.token.get))


@pytest.fixture
def fake_modal(monkeypatch):
    fake = FakeModal()
    monkeypatch.setattr(modal_adaptor, 'modal', fake)
    return fake


def _provision_config(count=1, environment='', **node_config):
    return provision_common.ProvisionConfig(
        provider_config={'environment': environment},
        authentication_config={},
        docker_config={},
        node_config={
            'InstanceType': '1x-T4',
            'Cpus': 4,
            'MemoryGiB': 16,
            'ImageId': '',
            'TtlMinutes': 0,
            **node_config,
        },
        count=count,
        tags={},
        resume_stopped_nodes=False,
        ports_to_open_on_launch=None,
    )


class TestModalInstanceTypes:

    @pytest.mark.parametrize('instance_type,gpu', [
        ('1x-T4', 'T4'),
        ('2x-H100', 'H100:2'),
        ('8x-A100', 'A100-40GB:8'),
        ('1x-A100-80GB', 'A100-80GB'),
        ('1x-RTXPRO6000', 'RTX-PRO-6000'),
        ('cpu-2x-8gb', None),
    ])
    def test_modal_gpu(self, instance_type, gpu):
        assert modal_utils.modal_gpu(instance_type) == gpu

    def test_unknown_gpu(self):
        with pytest.raises(modal_utils.ModalError):
            modal_utils.modal_gpu('1x-V100')

    def test_timeout(self):
        assert modal_utils.timeout_seconds(None) == 24 * 3600
        assert modal_utils.timeout_seconds(30) == 1800
        assert modal_utils.timeout_seconds(10_000) == 24 * 3600


class TestModalLifecycle:
    """Provision, status, runners and teardown against a fake SDK."""

    def test_full_lifecycle(self, fake_modal):
        record = modal_instance.run_instances('global', 'c', 'c-abcd',
                                              _provision_config(TtlMinutes=20))
        sandbox_id = record.head_instance_id
        assert record.created_instance_ids == [sandbox_id]
        assert not record.runtime_metadata.ssh_available

        (args, kwargs), = fake_modal.create_calls
        assert args == modal_utils.SANDBOX_COMMAND
        assert kwargs['gpu'] == 'T4'
        assert kwargs['cpu'] == 2  # 4 vCPUs are 2 Modal cores.
        assert kwargs['memory'] == 16 * 1024
        assert kwargs['timeout'] == 20 * 60
        assert kwargs['name'] == 'c-abcd'
        assert kwargs['tags'] == {
            modal_utils.CLUSTER_TAG: 'c-abcd',
            modal_utils.MANAGED_TAG: 'true'
        }
        sandbox = fake_modal.sandboxes[sandbox_id]
        assert [args for args, _ in sandbox.execs] == [('true',)]
        assert kwargs['image'].tag == modal_utils.DEFAULT_IMAGE
        assert kwargs['image'].commands == modal_utils.IMAGE_SETUP_COMMANDS

        again = modal_instance.run_instances('global', 'c', 'c-abcd',
                                             _provision_config())
        assert again.head_instance_id == sandbox_id
        assert not again.created_instance_ids
        assert len(fake_modal.create_calls) == 1

        assert modal_instance.query_instances('c', 'c-abcd') == {
            sandbox_id: (status_lib.ClusterStatus.UP, None)
        }
        info = modal_instance.get_cluster_info('global', 'c-abcd')
        assert info.head_instance_id == sandbox_id
        runners = modal_instance.get_command_runners(info)
        assert [r.sandbox_id for r in runners] == [sandbox_id]
        assert isinstance(runners[0], command_runner.ModalCommandRunner)

        modal_instance.terminate_instances('c-abcd')
        assert fake_modal.sandboxes[sandbox_id].exit_code is not None
        assert not modal_instance.query_instances('c', 'c-abcd')
        modal_instance.terminate_instances('c-abcd')  # Harmless twice.

    def test_other_clusters_untouched(self, fake_modal):
        modal_instance.run_instances('global', 'a', 'a-1', _provision_config())
        modal_instance.run_instances('global', 'b', 'b-1', _provision_config())
        modal_instance.terminate_instances('a-1')
        assert not modal_instance.query_instances('a', 'a-1')
        assert len(modal_instance.query_instances('b', 'b-1')) == 1

    def test_sandbox_ended_by_timeout_reads_as_terminated(self, fake_modal):
        record = modal_instance.run_instances('global', 'c', 'c-abcd',
                                              _provision_config())
        fake_modal.sandboxes[record.head_instance_id].exit_code = 0
        assert not modal_instance.query_instances('c', 'c-abcd')
        info = modal_instance.get_cluster_info('global', 'c-abcd')
        assert info.head_instance_id is None

    def test_no_app_yet_means_no_cluster(self, fake_modal):
        del fake_modal
        assert not modal_instance.query_instances('c', 'c-abcd')
        modal_instance.terminate_instances('c-abcd')

    def test_environment_is_used(self, fake_modal):
        modal_instance.run_instances('global', 'c', 'c-abcd',
                                     _provision_config(environment='staging'))
        assert (modal_utils.APP_NAME, 'staging') in fake_modal.apps
        assert not modal_instance.query_instances('c', 'c-abcd')
        assert modal_instance.query_instances(
            'c', 'c-abcd', provider_config={'environment': 'staging'})

    def test_custom_image_and_cpu_shape(self, fake_modal):
        modal_instance.run_instances(
            'global', 'c', 'c-abcd',
            _provision_config(InstanceType='cpu-2x-8gb',
                              Cpus=2,
                              MemoryGiB=8,
                              ImageId='python:3.11-slim'))
        (_, kwargs), = fake_modal.create_calls
        assert 'gpu' not in kwargs
        assert kwargs['image'].tag == 'python:3.11-slim'

    def test_create_is_idempotent_by_name(self, fake_modal):
        # A launch interrupted after create, before SkyPilot saw the ID.
        first = modal_utils.create_sandbox('c-abcd', '1x-T4', 4, 16, None, None)
        again = modal_utils.create_sandbox('c-abcd', '1x-T4', 4, 16, None, None)
        assert again is first
        assert len(fake_modal.sandboxes) == 1

    def test_name_is_free_again_after_teardown(self, fake_modal):
        modal_instance.run_instances('global', 'c', 'c-abcd',
                                     _provision_config())
        modal_instance.terminate_instances('c-abcd')
        record = modal_instance.run_instances('global', 'c', 'c-abcd',
                                              _provision_config())
        assert record.created_instance_ids == ['sb-2']

    @pytest.mark.parametrize('probe, match', [(None, 'did not start'),
                                              (1, 'exited with code 1')])
    def test_start_probe_failure_fails_the_launch(self, fake_modal, probe,
                                                  match):
        fake_modal.probe = probe
        with pytest.raises(modal_utils.ModalError, match=match):
            modal_instance.run_instances('global', 'c', 'c-abcd',
                                         _provision_config())

    def test_terminate_tolerates_a_gone_sandbox(self, fake_modal):
        record = modal_instance.run_instances('global', 'c', 'c-abcd',
                                              _provision_config())
        sandbox = fake_modal.sandboxes.pop(record.head_instance_id)
        modal_utils.terminate_sandbox(sandbox)

    def test_multi_node_rejected(self, fake_modal):
        with pytest.raises(modal_utils.ModalError):
            modal_instance.run_instances('global', 'c', 'c-abcd',
                                         _provision_config(count=2))
        assert not fake_modal.create_calls

    def test_stop_unsupported(self):
        with pytest.raises(NotImplementedError):
            modal_instance.stop_instances('c-abcd')


class TestModalCloud:

    def test_registered(self):
        assert isinstance(registry.CLOUD_REGISTRY.from_str('modal'),
                          modal_cloud.Modal)
        assert repr(modal_cloud.Modal()) == 'Modal'

    def test_unsupported_features(self):
        features = modal_cloud.Modal._CLOUD_UNSUPPORTED_FEATURES  # pylint: disable=protected-access
        for feature in ('STOP', 'AUTOSTOP', 'MULTI_NODE', 'SPOT_INSTANCE',
                        'OPEN_PORTS', 'HOST_CONTROLLERS'):
            assert getattr(clouds.CloudImplementationFeatures,
                           feature) in features
        # Autodown runs on the sandbox with the mounted token. Storage mounts
        # run over exec, after a FUSE check on the sandbox.
        for feature in ('IMAGE_ID', 'AUTODOWN', 'STORAGE_MOUNTING'):
            assert getattr(clouds.CloudImplementationFeatures,
                           feature) not in features

    def test_catalog_is_bundled(self):
        assert catalog_common.get_bundled_catalog_path('modal/vms.csv')

    def test_gpu_request_picks_catalog_shape(self):
        r = resources_lib.Resources(infra='modal', accelerators='T4:1')
        feasible = modal_cloud.Modal()._get_feasible_launchable_resources(r)  # pylint: disable=protected-access
        (chosen,) = feasible.resources_list
        assert chosen.instance_type == '1x-T4'
        cost = chosen.get_cost(3600)
        # GPU plus 2 cores and 16 GiB at the sandbox rate.
        assert cost == pytest.approx(0.5904 + 3 * 3600 *
                                     (2 * 0.0000131 + 16 * 0.00000222),
                                     rel=1e-4)

    @pytest.mark.parametrize('acc, instance_type, gpu', [
        ('H100:3', '3x-H100', 'H100:3'),
        ('A10G:1', '1x-A10G', 'A10'),
        ('A10:4', '4x-A10', 'A10:4'),
    ])
    def test_any_gpu_count_up_to_the_max(self, acc, instance_type, gpu):
        r = resources_lib.Resources(infra='modal', accelerators=acc)
        feasible = modal_cloud.Modal()._get_feasible_launchable_resources(r)  # pylint: disable=protected-access
        (chosen,) = feasible.resources_list
        assert chosen.instance_type == instance_type
        assert modal_utils.modal_gpu(instance_type) == gpu

    def test_a10_stops_at_four(self):
        r = resources_lib.Resources(infra='modal', accelerators='A10:5')
        feasible = modal_cloud.Modal()._get_feasible_launchable_resources(r)  # pylint: disable=protected-access
        assert not feasible.resources_list

    def test_spot_is_infeasible(self):
        r = resources_lib.Resources(infra='modal',
                                    accelerators='T4:1',
                                    use_spot=True)
        feasible = modal_cloud.Modal()._get_feasible_launchable_resources(r)  # pylint: disable=protected-access
        assert not feasible.resources_list

    def test_credentials(self, monkeypatch, tmp_path):
        monkeypatch.setattr(modal_utils, 'CREDENTIALS_PATH',
                            str(tmp_path / 'missing.toml'))
        monkeypatch.delenv(modal_utils.ENV_TOKEN_ID, raising=False)
        monkeypatch.delenv(modal_utils.ENV_TOKEN_SECRET, raising=False)
        ok, msg = modal_cloud.Modal._check_credentials()  # pylint: disable=protected-access
        assert not ok and 'modal token new' in msg
        monkeypatch.setenv(modal_utils.ENV_TOKEN_ID, 'ak-x')
        monkeypatch.setenv(modal_utils.ENV_TOKEN_SECRET, 'as-x')
        assert modal_cloud.Modal._check_credentials() == (True, None)  # pylint: disable=protected-access

    def test_token_file_for_autodown(self, fake_modal, monkeypatch, tmp_path):
        generated = tmp_path / 'generated' / 'modal.toml'
        monkeypatch.setattr(modal_utils, '_GENERATED_CREDENTIALS_PATH',
                            str(generated))
        assert not modal_cloud.Modal.get_credential_file_mounts()
        fake_modal.token.update(token_id='ak-1', token_secret='as-1')
        mounts = modal_cloud.Modal.get_credential_file_mounts()
        assert mounts == {'~/.modal.toml': str(generated)}
        assert generated.stat().st_mode & 0o777 == 0o600
        assert _read_toml(generated) == {
            'skypilot': {
                'token_id': 'ak-1',
                'token_secret': 'as-1',
                'active': True
            }
        }

    @pytest.mark.parametrize('remote', ['1', None])
    def test_autodown_on_the_sandbox_reads_the_token(self, fake_modal,
                                                     monkeypatch, tmp_path,
                                                     remote):
        # The SDK ignores ~/.modal.toml inside Modal containers.
        generated = tmp_path / 'modal.toml'
        monkeypatch.setattr(modal_utils, '_GENERATED_CREDENTIALS_PATH',
                            str(generated))
        fake_modal.token.update(token_id='ak-1', token_secret='as-1')
        modal_utils.write_credentials_file()
        monkeypatch.setattr(modal_utils, 'CREDENTIALS_PATH', str(generated))
        for var in (modal_utils.ENV_TOKEN_ID, modal_utils.ENV_TOKEN_SECRET,
                    'MODAL_IS_REMOTE'):
            monkeypatch.delenv(var, raising=False)
        if remote:
            monkeypatch.setenv('MODAL_IS_REMOTE', remote)
        modal_instance.terminate_instances('c-abcd')
        exported = (os.environ.get(modal_utils.ENV_TOKEN_ID),
                    os.environ.get(modal_utils.ENV_TOKEN_SECRET))
        assert exported == (('ak-1', 'as-1') if remote else (None, None))

    def test_no_sdk_no_mounts(self, monkeypatch):
        monkeypatch.setattr(modal_utils, 'write_credentials_file',
                            mock.Mock(side_effect=ImportError('modal')))
        assert not modal_cloud.Modal.get_credential_file_mounts()

    def test_deploy_variables(self):
        r = resources_lib.Resources(infra='modal/global',
                                    instance_type='1x-H100',
                                    image_id='docker:pytorch/pytorch:latest')
        r = r.copy(_cluster_config_overrides={
            'modal': {
                'ttl_minutes': 45,
                'environment': 'staging'
            }
        })
        variables = modal_cloud.Modal().make_deploy_resources_variables(
            r, mock.MagicMock(), clouds.Region('global'), None, 1)
        assert variables['cpus'] == 8
        assert variables['memory_gib'] == 32
        assert variables['ttl_minutes'] == 45
        assert variables['modal_environment'] == 'staging'
        assert variables['image_id'] == 'pytorch/pytorch:latest'


def _read_toml(path):
    try:
        import tomllib  # pylint: disable=import-outside-toplevel
        with open(path, 'rb') as f:
            return tomllib.load(f)
    except ImportError:  # Python < 3.11
        import toml  # pylint: disable=import-outside-toplevel
        return toml.load(str(path))


class TestModalExecHelper:

    def test_parse_args(self):
        assert exec_helper.parse_args(['sb-1', 'echo',
                                       'hi']) == ('sb-1', 'echo hi')
        # rsync passes `-l user host command...`.
        assert exec_helper.parse_args(
            ['-l', 'root', 'sb-1', 'rsync', '--server',
             '.']) == ('sb-1', 'rsync --server .')
        assert exec_helper.parse_args(['root@sb-1', 'true']) == ('sb-1', 'true')
        with pytest.raises(ValueError):
            exec_helper.parse_args(['sb-1'])

    def test_unreachable_sandbox(self, capsys):
        fake = mock.Mock()
        fake.Sandbox.from_id.side_effect = RuntimeError('not found')
        assert exec_helper.main(['sb-gone', 'true'], modal_module=fake) == 255
        assert 'Cannot reach Modal sandbox sb-gone' in capsys.readouterr().err


# A stand-in `modal` package that runs exec'd commands locally, so the real
# helper script, ModalCommandRunner and rsync can be exercised end to end.
_LOCAL_MODAL = textwrap.dedent('''
    import subprocess

    class _Reader:
        def __init__(self, f):
            self.f = f
        def __iter__(self):
            while True:
                data = self.f.read1(65536)
                if not data:
                    return
                yield data

    class _Writer:
        def __init__(self, f):
            self.f = f
        def write(self, data):
            self.f.write(data)
        def drain(self):
            self.f.flush()
        def write_eof(self):
            self.f.close()

    class _Process:
        def __init__(self, args):
            self.p = subprocess.Popen(args, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
            self.stdin = _Writer(self.p.stdin)
            self.stdout = _Reader(self.p.stdout)
            self.stderr = _Reader(self.p.stderr)
        def wait(self):
            return self.p.wait()

    class Sandbox:
        @staticmethod
        def from_id(sandbox_id):
            if sandbox_id == 'sb-gone':
                raise RuntimeError('no such sandbox')
            return Sandbox()
        def exec(self, *args, text=True):
            assert text is False
            return _Process(list(args))
''')


@pytest.fixture
def local_modal(tmp_path, monkeypatch):
    pkg = tmp_path / 'fake_modal_pkg'
    pkg.mkdir()
    (pkg / 'modal.py').write_text(_LOCAL_MODAL)
    monkeypatch.setenv('PYTHONPATH', str(pkg))
    return command_runner.ModalCommandRunner(('modal', 'sb-local'))


class TestModalCommandRunner:

    def test_helper_path_exists(self):
        assert os.path.exists(command_runner.MODAL_EXEC_HELPER)

    def test_run_through_exec(self, local_modal):
        rc, out, err = local_modal.run('echo "a b" && echo oops >&2 && exit 3',
                                       require_outputs=True,
                                       stream_logs=False,
                                       separate_stderr=True)
        assert rc == 3
        assert 'a b' in out
        assert 'oops' in err

    def test_stdin_is_forwarded(self, tmp_path):
        pkg = tmp_path / 'p'
        pkg.mkdir()
        (pkg / 'modal.py').write_text(_LOCAL_MODAL)
        env = dict(os.environ, PYTHONPATH=str(pkg))
        result = subprocess.run(
            [sys.executable, command_runner.MODAL_EXEC_HELPER, 'sb-1', 'cat'],
            input=b'\x00binary\xffdata',
            capture_output=True,
            env=env,
            check=False)
        assert result.returncode == 0
        assert result.stdout == b'\x00binary\xffdata'

    def test_unreachable_is_255(self, tmp_path, local_modal):
        del tmp_path
        runner = command_runner.ModalCommandRunner(('modal', 'sb-gone'))
        rc = runner.run('true', stream_logs=False)
        assert rc == 255
        assert local_modal.node_id == 'modal-sb-local'

    def test_rsync_up_and_down(self, tmp_path, local_modal):
        src = tmp_path / 'src'
        src.mkdir()
        (src / 'a.txt').write_text('hello')
        (src / 'sub').mkdir()
        (src / 'sub' / 'b.bin').write_bytes(os.urandom(200_000))
        remote = tmp_path / 'remote'
        local_modal.rsync(str(src), str(remote), up=True, stream_logs=False)
        assert (remote / 'a.txt').read_text() == 'hello'
        assert ((remote / 'sub' /
                 'b.bin').read_bytes() == (src / 'sub' / 'b.bin').read_bytes())

        back = tmp_path / 'back'
        local_modal.rsync(str(remote / 'a.txt'),
                          str(back),
                          up=False,
                          stream_logs=False)
        assert back.read_text() == 'hello'

    def test_login_is_refused(self, local_modal):
        with pytest.raises(NotImplementedError, match='modal shell'):
            local_modal.run(['bash'], ssh_mode=command_runner.SshMode.LOGIN)


def _fake_bin(directory, name, body='exit 0'):
    path = directory / name
    path.write_text(f'#!/bin/bash\n{body}\n')
    path.chmod(0o755)


class TestModalStorageMounts:
    """A gs:// MOUNT runs over exec: the FUSE check, then SkyPilot's script."""

    @pytest.fixture
    def sandbox(self, tmp_path, monkeypatch, local_modal):
        bin_dir = tmp_path / 'bin'
        bin_dir.mkdir()
        # gcsfuse records its arguments; the tools the check looks for exist.
        _fake_bin(bin_dir, 'gcsfuse', f'echo "$@" > {tmp_path}/gcsfuse.args')
        _fake_bin(bin_dir, 'fusermount3')
        _fake_bin(bin_dir, 'wget')
        _fake_bin(bin_dir, 'apt-get', f'touch {tmp_path}/apt.ran')
        del monkeypatch
        return tmp_path, local_modal

    @staticmethod
    def _on_path(tmp_path, cmd):
        # The command's own shell sets PATH, so the fakes come first.
        return f'export PATH={tmp_path}/bin:$PATH && {cmd}'

    def _mount_command(self, tmp_path, fuse_device):
        mount_path = str(tmp_path / 'results')
        mount = mounting_utils.get_mounting_command(
            mount_path, mounting_utils.get_gcs_mount_install_cmd(),
            mounting_utils.get_gcs_mount_cmd('bk-results', mount_path,
                                             'bk-run-1'))
        check = mounting_utils.get_sandbox_mount_prerequisites_cmd(
            'Modal', fuse_device)
        return self._on_path(tmp_path, f'({check}) && {mount}')

    def test_mount_runs_over_exec(self, sandbox):
        tmp_path, runner = sandbox
        fuse = tmp_path / 'fuse'
        fuse.touch()
        rc, out, err = runner.run(self._mount_command(tmp_path, str(fuse)),
                                  require_outputs=True,
                                  stream_logs=False)
        assert rc == 0, (out, err)
        args = (tmp_path / 'gcsfuse.args').read_text().split()
        assert args[-2:] == ['bk-results', str(tmp_path / 'results')]
        assert args[args.index('--only-dir') + 1] == 'bk-run-1'
        assert not (tmp_path / 'apt.ran').exists()  # the tools were there

    def test_no_fuse_fails_before_mounting(self, sandbox):
        tmp_path, runner = sandbox
        rc, out, err = runner.run(self._mount_command(tmp_path,
                                                      str(tmp_path /
                                                          'no-fuse')),
                                  require_outputs=True,
                                  stream_logs=False,
                                  separate_stderr=True)
        assert rc == 1, (out, err)
        assert 'this Modal sandbox has no' in err and 'mode: COPY' in err
        assert not (tmp_path / 'gcsfuse.args').exists()

    def test_missing_tools_are_installed(self, sandbox):
        tmp_path, runner = sandbox
        (tmp_path / 'bin' / 'wget').unlink()
        fuse = tmp_path / 'fuse'
        fuse.touch()
        check = mounting_utils.get_sandbox_mount_prerequisites_cmd(
            'Modal', str(fuse))
        # Only the fakes on PATH, so tools on this machine don't count.
        for tool in ('whoami', 'env', 'touch'):
            os.symlink(shutil.which(tool), tmp_path / 'bin' / tool)
        assert runner.run(f'export PATH={tmp_path}/bin && {check}',
                          stream_logs=False) == 0
        assert (tmp_path / 'apt.ran').exists()

    @pytest.mark.parametrize('cloud', [modal_cloud.Modal, clouds.Daytona])
    def test_backend_runs_the_check_first(self, cloud):
        store = mock.Mock()
        store.mount_command.return_value = 'MOUNT-GCS'
        storage = mock.Mock(mode=storage_lib.StorageMode.MOUNT,
                            mount_config=None,
                            source='gs://bk-results/bk-run-1',
                            stores={'gcs': store})
        handle = mock.Mock()
        handle.launched_resources.cloud = cloud()
        backend = cloud_vm_ray_backend.CloudVmRayBackend()
        backend.log_dir = '/tmp'
        with mock.patch.object(cloud_vm_ray_backend.backend_utils,
                               'parallel_data_transfer_to_nodes') as transfer:
            backend._execute_storage_mounts(handle, {'/results': storage})  # pylint: disable=protected-access
        cmd = transfer.call_args.kwargs['cmd']
        assert cmd.startswith('({ [ -e /dev/fuse ]')
        assert cmd.endswith(') && MOUNT-GCS')
        assert f'this {cloud._REPR} sandbox' in cmd  # pylint: disable=protected-access

    def test_clouds_with_fuse_run_no_check(self):
        assert clouds.GCP.storage_mount_prerequisites_command() is None
