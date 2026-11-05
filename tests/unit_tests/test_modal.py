"""Tests for the Modal cloud provider.

All Modal SDK calls go to an in-memory fake; nothing reaches Modal.
"""

import os
import types
from typing import Any, Dict, List, Optional

import jinja2
import pytest
import yaml

import sky
from sky import clouds
from sky.adaptors import modal as modal_adaptor
from sky.catalog import common as catalog_common
from sky.catalog.data_fetchers import fetch_modal
from sky.clouds import modal as modal_cloud
from sky.provision import common as provision_common
from sky.provision.modal import instance as modal_instance
from sky.provision.modal import utils as modal_utils
from sky.utils import command_runner
from sky.utils import resources_utils
from sky.utils import status_lib


class _AlreadyExistsError(Exception):
    pass


class _NotFoundError(Exception):
    pass


class _FakeProcess:

    def __init__(self, returncode: int = 0):
        self.returncode = returncode

    def wait(self):
        return self.returncode


class _FakeSandbox:

    def __init__(self, sandbox_id: str, app_id: str, name: Optional[str],
                 tags: Dict[str, str], kwargs: Dict[str, Any], args):
        self.object_id = sandbox_id
        self.app_id = app_id
        self.name = name
        self.tags = dict(tags)
        self.kwargs = kwargs
        self.args = args
        self.terminated = False
        self.exec_calls: List[Any] = []
        self.probe_returncode = 0

    def exec(self, *args, timeout=None):
        self.exec_calls.append((args, timeout))
        return _FakeProcess(self.probe_returncode)

    def poll(self):
        return 137 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def set_tags(self, tags):
        self.tags = dict(tags)


class _FakeImage:

    def __init__(self, source: str):
        self.source = source
        self.apt_packages: List[str] = []

    @classmethod
    def from_registry(cls, tag):
        return cls(f'registry:{tag}')

    @classmethod
    def debian_slim(cls, python_version=None):
        return cls(f'debian_slim:{python_version}')

    def apt_install(self, *packages):
        self.apt_packages.extend(packages)
        return self


class FakeModal:
    """Minimal stand-in for the `modal` package."""

    def __init__(self):
        self.sandboxes: List[_FakeSandbox] = []
        self.apps: Dict[str, str] = {}
        self.exception = types.SimpleNamespace(
            AlreadyExistsError=_AlreadyExistsError,
            NotFoundError=_NotFoundError)
        self.Image = _FakeImage
        fake = self

        class _App:

            def __init__(self, app_id):
                self.app_id = app_id

            @staticmethod
            def lookup(name, create_if_missing=False, environment_name=None):
                del environment_name
                if name not in fake.apps:
                    assert create_if_missing
                    fake.apps[name] = f'ap-{len(fake.apps) + 1}'
                return _App(fake.apps[name])

        class _Sandbox:

            @staticmethod
            def create(*args, app, name=None, tags=None, **kwargs):
                for sb in fake.sandboxes:
                    if (not sb.terminated and sb.app_id == app.app_id and
                            name is not None and sb.name == name):
                        raise _AlreadyExistsError(name)
                sb = _FakeSandbox(f'sb-{len(fake.sandboxes) + 1}', app.app_id,
                                  name, tags or {}, kwargs, args)
                fake.sandboxes.append(sb)
                return sb

            @staticmethod
            def from_name(app_name, name):
                app_id = fake.apps[app_name]
                for sb in fake.sandboxes:
                    if (not sb.terminated and sb.app_id == app_id and
                            sb.name == name):
                        return sb
                raise _NotFoundError(name)

            @staticmethod
            def list(app_id=None, tags=None):
                tags = tags or {}
                for sb in list(fake.sandboxes):
                    if sb.terminated or sb.app_id != app_id:
                        continue
                    if all(sb.tags.get(k) == v for k, v in tags.items()):
                        yield sb

        self.App = _App
        self.Sandbox = _Sandbox

    def live(self) -> List[_FakeSandbox]:
        return [sb for sb in self.sandboxes if not sb.terminated]


@pytest.fixture
def fake_modal(monkeypatch):
    fake = FakeModal()
    monkeypatch.setattr(modal_adaptor, 'modal', fake)
    return fake


def _provision_config(node_config: Dict[str, Any],
                      count: int = 1) -> provision_common.ProvisionConfig:
    return provision_common.ProvisionConfig(
        provider_config={'module': 'sky.provision.modal'},
        authentication_config={},
        docker_config={},
        node_config=node_config,
        count=count,
        tags={},
        resume_stopped_nodes=False,
        ports_to_open_on_launch=None,
    )


_H100_NODE = {
    'InstanceType': '1x-H100',
    'Cpus': 4,
    'MemoryGiB': 64,
    'Gpu': 'H100',
    'ImageId': '',
    'TimeoutSeconds': 86400,
}


class TestModalCredentials:

    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(modal_utils, 'CREDENTIALS_PATH',
                            str(tmp_path / '.modal.toml'))
        monkeypatch.delenv('MODAL_TOKEN_ID', raising=False)
        monkeypatch.delenv('MODAL_TOKEN_SECRET', raising=False)

    def test_missing(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        valid, msg = modal_cloud.Modal.check_credentials(
            clouds.CloudCapability.COMPUTE)
        assert not valid
        assert 'modal token new' in msg

    def test_env_needs_both_vars(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        monkeypatch.setenv('MODAL_TOKEN_ID', 'ak-x')
        assert not modal_utils.has_credentials()
        monkeypatch.setenv('MODAL_TOKEN_SECRET', 'as-x')
        assert modal_utils.has_credentials()

    def test_file(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        (tmp_path / '.modal.toml').write_text('[default]\n')
        valid, msg = modal_cloud.Modal.check_credentials(
            clouds.CloudCapability.COMPUTE)
        assert valid and msg is None


class TestModalProvisioning:

    def test_run_instances_creates_tagged_sandbox(self, fake_modal):
        record = modal_instance.run_instances('global', 'c', 'c-abcd',
                                              _provision_config(_H100_NODE))

        [sb] = fake_modal.sandboxes
        assert record.head_instance_id == sb.object_id
        assert record.created_instance_ids == [sb.object_id]
        assert sb.args == ('sleep', 'infinity')
        assert sb.name == 'c-abcd'
        assert sb.tags == {
            'skypilot-cluster': 'c-abcd',
            'skypilot-managed': 'true'
        }
        assert sb.kwargs['gpu'] == 'H100'
        assert sb.kwargs['cpu'] == 4.0
        assert sb.kwargs['memory'] == 64 * 1024
        assert sb.kwargs['timeout'] == 86400
        assert sb.kwargs['image'].source == 'debian_slim:3.11'
        assert 'rsync' in sb.kwargs['image'].apt_packages
        # The start probe ran.
        assert sb.exec_calls and sb.exec_calls[0][0] == ('true',)

    def test_run_instances_caps_timeout_and_uses_registry_image(
            self, fake_modal):
        node = dict(_H100_NODE,
                    Gpu='',
                    ImageId='ubuntu:22.04',
                    TimeoutSeconds=10**9)
        modal_instance.run_instances('global', 'c', 'c-1',
                                     _provision_config(node))
        [sb] = fake_modal.sandboxes
        assert sb.kwargs['timeout'] == modal_utils.MAX_TIMEOUT_SECONDS
        assert sb.kwargs['image'].source == 'registry:ubuntu:22.04'
        assert 'gpu' not in sb.kwargs

    def test_run_instances_reuses_existing(self, fake_modal):
        first = modal_instance.run_instances('global', 'c', 'c-1',
                                             _provision_config(_H100_NODE))
        second = modal_instance.run_instances('global', 'c', 'c-1',
                                              _provision_config(_H100_NODE))
        assert len(fake_modal.sandboxes) == 1
        assert second.head_instance_id == first.head_instance_id
        assert second.created_instance_ids == []

    def test_create_race_reuses_named_sandbox(self, fake_modal, monkeypatch):
        modal_instance.run_instances('global', 'c', 'c-1',
                                     _provision_config(_H100_NODE))
        # Simulate a launch that listed before the other one created.
        monkeypatch.setattr(modal_instance, '_cluster_sandboxes',
                            lambda name: {})
        record = modal_instance.run_instances('global', 'c', 'c-1',
                                              _provision_config(_H100_NODE))
        assert len(fake_modal.sandboxes) == 1
        assert record.head_instance_id == fake_modal.sandboxes[0].object_id

    def test_old_sdk_without_create_tags(self, fake_modal, monkeypatch):
        original = fake_modal.Sandbox.create

        def create(*args,
                   app,
                   name=None,
                   image=None,
                   timeout=None,
                   workdir=None,
                   gpu=None,
                   cpu=None,
                   memory=None):
            return original(*args,
                            app=app,
                            name=name,
                            image=image,
                            timeout=timeout,
                            workdir=workdir,
                            gpu=gpu,
                            cpu=cpu,
                            memory=memory)

        monkeypatch.setattr(fake_modal.Sandbox, 'create', create)
        modal_instance.run_instances('global', 'c', 'c-1',
                                     _provision_config(_H100_NODE))
        [sb] = fake_modal.sandboxes
        assert sb.tags == modal_utils.cluster_tags('c-1')
        assert list(modal_instance.query_instances('c',
                                                   'c-1')) == [sb.object_id]

    def test_run_instances_rejects_multi_node(self, fake_modal):
        with pytest.raises(modal_utils.ModalError):
            modal_instance.run_instances('global', 'c', 'c-1',
                                         _provision_config(_H100_NODE, 2))
        assert not fake_modal.sandboxes

    def test_failed_start_probe_raises(self, fake_modal, monkeypatch):
        original = fake_modal.Sandbox.create

        def create(*args, **kwargs):
            sb = original(*args, **kwargs)
            sb.probe_returncode = 1
            return sb

        monkeypatch.setattr(fake_modal.Sandbox, 'create', create)
        with pytest.raises(modal_utils.ModalError):
            modal_instance.run_instances('global', 'c', 'c-1',
                                         _provision_config(_H100_NODE))

    def test_query_and_cluster_info(self, fake_modal):
        record = modal_instance.run_instances('global', 'c', 'c-1',
                                              _provision_config(_H100_NODE))
        # Another cluster's sandbox must not show up.
        modal_instance.run_instances('global', 'd', 'd-1',
                                     _provision_config(_H100_NODE))
        sandbox_id = record.head_instance_id

        assert modal_instance.query_instances('c', 'c-1') == {
            sandbox_id: (status_lib.ClusterStatus.UP, None)
        }
        info = modal_instance.get_cluster_info('global', 'c-1', {'module': 'x'})
        assert info.head_instance_id == sandbox_id
        assert list(info.instances) == [sandbox_id]
        assert info.instances[sandbox_id][0].instance_id == sandbox_id
        assert info.ssh_user == 'root'
        assert info.provider_name == 'modal'

    def test_terminate(self, fake_modal):
        modal_instance.run_instances('global', 'c', 'c-1',
                                     _provision_config(_H100_NODE))
        modal_instance.run_instances('global', 'd', 'd-1',
                                     _provision_config(_H100_NODE))
        modal_instance.terminate_instances('c-1', worker_only=True)
        assert len(fake_modal.live()) == 2

        modal_instance.terminate_instances('c-1')
        assert [sb.name for sb in fake_modal.live()] == ['d-1']
        assert modal_instance.query_instances('c', 'c-1') == {}
        info = modal_instance.get_cluster_info('global', 'c-1')
        assert info.head_instance_id is None and not info.instances
        # Terminating an already-terminated cluster is a no-op.
        modal_instance.terminate_instances('c-1')

    def test_stop_unsupported(self):
        with pytest.raises(NotImplementedError):
            modal_instance.stop_instances('c-1')

    def test_get_command_runners_passes_sandbox_ids(self, monkeypatch):
        calls = []

        class FakeRunner:

            @classmethod
            def make_runner_list(cls, node_list, **credentials):
                calls.append((list(node_list), credentials))
                return ['runner']

        monkeypatch.setattr(command_runner,
                            'ModalCommandRunner',
                            FakeRunner,
                            raising=False)
        info = provision_common.ClusterInfo(
            instances={
                'sb-2': [],
                'sb-1': []
            },
            head_instance_id='sb-1',
            provider_name='modal',
        )
        assert modal_instance.get_command_runners(
            info, ssh_user='root') == ['runner']
        assert calls == [(['sb-1', 'sb-2'], {'ssh_user': 'root'})]


class TestModalCatalog:

    def test_bundled_catalog_matches_generator(self):
        path = catalog_common.get_bundled_catalog_path('modal/vms.csv')
        assert path is not None
        with open(path, encoding='utf-8') as f:
            assert f.read() == fetch_modal.catalog_csv(), (
                'Regenerate with python sky/catalog/data_fetchers/'
                'fetch_modal.py')

    def test_all_requested_gpus_present(self):
        rows = fetch_modal.catalog_rows()
        accs = {row[1] for row in rows if row[1]}
        assert accs == {
            'T4', 'L4', 'A10G', 'L40S', 'A100', 'A100-80GB', 'H100', 'H200',
            'B200'
        }
        assert all(row[6] == '' for row in rows)  # No spot.

    def test_price_is_gpu_plus_sandbox_cpu_and_memory(self):
        cloud = modal_cloud.Modal()
        expected = (0.001097 + 4 * 0.00003942 + 64 * 0.00000667) * 3600
        assert cloud.instance_type_to_hourly_cost(
            '1x-H100', use_spot=False) == pytest.approx(expected, rel=1e-6)
        assert cloud.instance_type_to_hourly_cost(
            '2x-H100', use_spot=False) == pytest.approx(2 * expected, rel=1e-6)

    @pytest.mark.parametrize('acc,count,gpu', [
        ('A100', 1, 'A100-40GB'),
        ('A100-80GB', 4, 'A100-80GB:4'),
        ('H100', 8, 'H100:8'),
        ('T4', 1, 'T4'),
    ])
    def test_modal_gpu_string(self, acc, count, gpu):
        assert fetch_modal.modal_gpu_string(acc, count) == gpu


class TestModalCloud:

    def test_feasible_resources_for_gpu(self):
        resources = sky.Resources(infra='modal', accelerators='H100:2')
        feasible = modal_cloud.Modal()._get_feasible_launchable_resources(
            resources)
        assert [r.instance_type for r in feasible.resources_list] == ['2x-H100']

    def test_a10g_capped_at_four(self):
        resources = sky.Resources(infra='modal', accelerators='A10G:8')
        feasible = modal_cloud.Modal()._get_feasible_launchable_resources(
            resources)
        assert not feasible.resources_list

    def test_default_cpu_instance_type(self):
        assert modal_cloud.Modal.get_default_instance_type(
            cpus='4', memory='16') == 'cpu-4x-16gb'
        assert modal_cloud.Modal.get_default_instance_type(
            use_spot=True) is None

    def test_features(self):
        unsupported = modal_cloud.Modal._CLOUD_UNSUPPORTED_FEATURES
        features = clouds.CloudImplementationFeatures
        for feature in (features.STOP, features.AUTOSTOP, features.MULTI_NODE,
                        features.SPOT_INSTANCE, features.OPEN_PORTS):
            assert feature in unsupported
        assert features.AUTODOWN not in unsupported
        assert repr(modal_cloud.Modal()) == 'Modal'

    @pytest.mark.parametrize('config,expected', [
        (None, 86400),
        ({
            'modal': {
                'timeout_minutes': 90
            }
        }, 5400),
    ])
    def test_timeout_from_config(self, monkeypatch, config, expected):
        cloud = modal_cloud.Modal()
        resources = sky.Resources(infra='modal', cpus='4', memory='16')
        [launchable
        ] = cloud._get_feasible_launchable_resources(resources).resources_list
        launchable = launchable.copy(_cluster_config_overrides=config)
        variables = cloud.make_deploy_resources_variables(
            launchable, resources_utils.ClusterName('c', 'c-1'),
            clouds.Region('global'), None, 1)
        assert variables['timeout_seconds'] == expected
        assert variables['gpu'] == ''

    def test_deploy_variables_render_template(self):
        cloud = modal_cloud.Modal()
        resources = sky.Resources(infra='modal',
                                  accelerators='A100-80GB:2',
                                  image_id='docker:nvcr.io/nvidia/pytorch:24')
        [launchable
        ] = cloud._get_feasible_launchable_resources(resources).resources_list
        variables = cloud.make_deploy_resources_variables(
            launchable,
            resources_utils.ClusterName('c', 'c-1'),
            clouds.Region('global'),
            None,
            1,
        )
        assert variables['gpu'] == 'A100-80GB:2'
        assert variables['cpus'] == 8
        assert variables['memory_gib'] == 128
        assert variables['image_id'] == 'nvcr.io/nvidia/pytorch:24'
        assert variables['timeout_seconds'] == 86400

        template_path = os.path.join(os.path.dirname(sky.__file__), 'templates',
                                     'modal-ray.yml.j2')
        with open(template_path, encoding='utf-8') as f:
            template = jinja2.Template(f.read())
        rendered = yaml.safe_load(
            template.render(cluster_name_on_cloud='c-1',
                            num_nodes=1,
                            ssh_private_key='~/.ssh/key',
                            credentials={},
                            **variables))
        node_config = rendered['available_node_types']['ray_head_default'][
            'node_config']
        assert rendered['provider']['module'] == 'sky.provision.modal'
        assert node_config == {
            'InstanceType': '2x-A100-80GB',
            'Cpus': 8,
            'MemoryGiB': 128,
            'Gpu': 'A100-80GB:2',
            'ImageId': 'nvcr.io/nvidia/pytorch:24',
            'TimeoutSeconds': 86400,
        }
