"""Modal and Daytona credentials reach controller clusters only, by default."""
import pathlib
from unittest import mock

import pytest

from sky import clouds
from sky import skypilot_config
from sky.backends import backend_utils
from sky.resources import Resources
from sky.utils import common

_CONTROLLER = f'{common.JOB_CONTROLLER_PREFIX}abcd1234'
_SERVE_CONTROLLER = f'{common.SKY_SERVE_CONTROLLER_PREFIX}abcd1234'
_SANDBOX_CLOUDS = {clouds.Modal(), clouds.Daytona()}


def _excluded_clouds(tmp_path, monkeypatch, cluster_name: str,
                     config: str) -> set:
    """The clouds write_cluster_config keeps out of the cluster's uploads."""
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(config)
    monkeypatch.setenv(skypilot_config.ENV_VAR_SKYPILOT_CONFIG,
                       str(config_path))
    skypilot_config.reload_config()
    with mock.patch('sky.check.get_cloud_credential_file_mounts',
                    return_value={}) as credential_mounts:
        backend_utils.write_cluster_config(
            to_provision=Resources(cloud=clouds.AWS(),
                                   instance_type='fake-type: 3'),
            num_nodes=1,
            cluster_config_template='aws-ray.yml.j2',
            cluster_name=cluster_name,
            local_wheel_path=pathlib.Path('/tmp/fake'),
            wheel_hash='b1bd84059bc0342f7843fcbe04ab563e',
            region=clouds.Region(name='fake-region'),
            zones=[clouds.Zone(name='fake-zone')],
            dryrun=True,
            keep_launch_fields_in_existing_config=True)
    credential_mounts.assert_called_once()
    return set(credential_mounts.call_args[0][0])


@pytest.fixture(autouse=True)
def _fake_cluster_config(monkeypatch):
    monkeypatch.setattr(skypilot_config, '_global_config_context',
                        skypilot_config.ConfigContext())
    patches = [
        mock.patch.object(clouds.aws, 'DEFAULT_SECURITY_GROUP_NAME',
                          'fake-default-sg'),
        mock.patch.object(clouds.AWS,
                          'get_image_root_device_name',
                          return_value='/dev/sda1'),
        mock.patch('sky.catalog.instance_type_exists', return_value=True),
        mock.patch('sky.catalog.get_accelerators_from_instance_type',
                   return_value={'fake-acc': 2}),
        mock.patch('sky.catalog.get_image_id_from_tag',
                   return_value='fake-image'),
        mock.patch('sky.catalog.get_arch_from_instance_type',
                   return_value='fake-arch'),
        mock.patch(
            'sky.backends.backend_utils._get_yaml_path_from_cluster_name',
            return_value='/tmp/fake/path'),
        mock.patch(
            'sky.backends.backend_utils._deterministic_cluster_yaml_hash',
            return_value='fake-hash'),
        mock.patch('sky.utils.common_utils.fill_template'),
    ]
    for p in patches:
        p.start()
    yield
    for p in reversed(patches):
        p.stop()
    skypilot_config.reload_config()


@pytest.mark.parametrize('config', ['{}\n', 'modal:\n  ttl_minutes: 30\n'])
def test_task_cluster_gets_no_sandbox_credentials(tmp_path, monkeypatch,
                                                  config):
    excluded = _excluded_clouds(tmp_path, monkeypatch, 'my-task', config)
    assert _SANDBOX_CLOUDS <= excluded
    assert clouds.AWS() not in excluded  # other clouds keep their defaults


@pytest.mark.parametrize('cluster_name', [_CONTROLLER, _SERVE_CONTROLLER])
def test_controllers_get_sandbox_credentials(tmp_path, monkeypatch,
                                             cluster_name):
    excluded = _excluded_clouds(tmp_path, monkeypatch, cluster_name, '{}\n')
    assert not _SANDBOX_CLOUDS & excluded


def test_local_credentials_opts_one_cloud_back_in(tmp_path, monkeypatch):
    excluded = _excluded_clouds(
        tmp_path, monkeypatch, 'my-task',
        'daytona:\n  remote_identity: local_credentials\n')
    assert clouds.Daytona() not in excluded
    assert clouds.Modal() in excluded


def test_no_upload_keeps_credentials_from_controllers_too(
        tmp_path, monkeypatch):
    excluded = _excluded_clouds(tmp_path, monkeypatch, _CONTROLLER,
                                'modal:\n  remote_identity: NO_UPLOAD\n')
    assert clouds.Modal() in excluded
    assert clouds.Daytona() not in excluded


def test_a_task_config_cannot_opt_in(tmp_path, monkeypatch):
    """remote_identity is not a task-overridable key for these clouds."""
    resources = Resources(cloud=clouds.AWS(),
                          instance_type='fake-type: 3',
                          _cluster_config_overrides={
                              'modal': {
                                  'remote_identity': 'LOCAL_CREDENTIALS'
                              }
                          })
    config_path = tmp_path / 'config.yaml'
    config_path.write_text('{}\n')
    monkeypatch.setenv(skypilot_config.ENV_VAR_SKYPILOT_CONFIG,
                       str(config_path))
    skypilot_config.reload_config()
    with mock.patch('sky.check.get_cloud_credential_file_mounts',
                    return_value={}) as credential_mounts:
        backend_utils.write_cluster_config(
            to_provision=resources,
            num_nodes=1,
            cluster_config_template='aws-ray.yml.j2',
            cluster_name='my-task',
            local_wheel_path=pathlib.Path('/tmp/fake'),
            wheel_hash='b1bd84059bc0342f7843fcbe04ab563e',
            region=clouds.Region(name='fake-region'),
            zones=[clouds.Zone(name='fake-zone')],
            dryrun=True,
            keep_launch_fields_in_existing_config=True)
    assert clouds.Modal() in set(credential_mounts.call_args[0][0])


@pytest.mark.parametrize('cloud', ['modal', 'daytona'])
@pytest.mark.parametrize('value,valid', [('LOCAL_CREDENTIALS', True),
                                         ('NO_UPLOAD', True),
                                         ('SERVICE_ACCOUNT', False)])
def test_remote_identity_schema(tmp_path, monkeypatch, cloud, value, valid):
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(f'{cloud}:\n  remote_identity: {value}\n')
    monkeypatch.setenv(skypilot_config.ENV_VAR_SKYPILOT_CONFIG,
                       str(config_path))
    if valid:
        skypilot_config.reload_config()
        assert skypilot_config.get_nested((cloud, 'remote_identity'),
                                          None) == value
    else:
        with pytest.raises(ValueError):
            skypilot_config.reload_config()
        monkeypatch.delenv(skypilot_config.ENV_VAR_SKYPILOT_CONFIG)
