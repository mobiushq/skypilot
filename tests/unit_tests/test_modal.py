"""Tests for the Modal cloud provider."""

import inspect
import os
import subprocess
import sys
import textwrap
from unittest import mock

import pytest

from sky import clouds
from sky import resources as resources_lib
from sky.adaptors import modal as modal_adaptor
from sky.catalog import common as catalog_common
from sky.clouds import modal as modal_cloud
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


class FakeSandbox:

    def __init__(self, fake, sandbox_id, tags, kwargs):
        self._fake = fake
        self.object_id = sandbox_id
        self.tags = dict(tags)
        self.kwargs = kwargs
        self.exit_code = None

    def poll(self):
        return self.exit_code

    def terminate(self):
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
                sandbox_id = f'sb-{len(fake.sandboxes) + 1}'
                sandbox = FakeSandbox(fake, sandbox_id, kwargs['tags'], kwargs)
                sandbox.app_id = kwargs['app'].app_id
                fake.sandboxes[sandbox_id] = sandbox
                return sandbox

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
        self.exception = mock.Mock(NotFoundError=_NotFoundError)


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
        assert kwargs['tags'] == {modal_utils.CLUSTER_TAG: 'c-abcd'}
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
        for feature in ('STOP', 'MULTI_NODE', 'SPOT_INSTANCE', 'OPEN_PORTS',
                        'AUTODOWN', 'HOST_CONTROLLERS'):
            assert getattr(clouds.CloudImplementationFeatures,
                           feature) in features
        assert clouds.CloudImplementationFeatures.IMAGE_ID not in features

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
