"""Tests for the Daytona cloud provider."""

import csv
import json
import logging
import os
import re
import stat
from unittest import mock

import jinja2
import pytest
import yaml

import sky
from sky import clouds
from sky import exceptions
from sky import provision
from sky import resources as resources_lib
from sky import skypilot_config
from sky.backends import backend_utils
from sky.backends import cloud_vm_ray_backend
from sky.catalog import common as catalog_common
from sky.catalog import daytona_catalog
from sky.clouds import daytona
from sky.provision.daytona import instance as daytona_instance
from sky.provision.daytona import utils as daytona_utils
from sky.skylet import events
from sky.utils import common_utils
from sky.utils import schemas
from sky.utils import status_lib


class TestDaytonaCredentials:
    """Credential resolution and checking."""

    def test_check_credentials_missing(self, monkeypatch, tmp_path):
        fake_path = tmp_path / 'api_key'
        monkeypatch.setattr(daytona_utils, 'CREDENTIALS_PATH', str(fake_path))
        monkeypatch.setattr(daytona.Daytona, 'get_credentials_path',
                            classmethod(lambda cls: str(fake_path)))
        monkeypatch.delenv('DAYTONA_API_KEY', raising=False)

        valid, msg = daytona.Daytona.check_credentials(
            clouds.CloudCapability.COMPUTE)

        assert not valid
        assert 'Daytona API key not found' in msg

    def test_check_credentials_from_env(self, monkeypatch, tmp_path):
        fake_path = tmp_path / 'api_key'
        monkeypatch.setattr(daytona.Daytona, 'get_credentials_path',
                            classmethod(lambda cls: str(fake_path)))
        monkeypatch.setenv('DAYTONA_API_KEY', 'dtn_test')

        valid, msg = daytona.Daytona.check_credentials(
            clouds.CloudCapability.COMPUTE)

        assert valid
        assert msg is None

    def test_check_credentials_from_file(self, monkeypatch, tmp_path):
        cred_path = tmp_path / 'api_key'
        cred_path.write_text('dtn_test')
        monkeypatch.setattr(daytona.Daytona, 'get_credentials_path',
                            classmethod(lambda cls: str(cred_path)))
        monkeypatch.delenv('DAYTONA_API_KEY', raising=False)

        valid, msg = daytona.Daytona.check_credentials(
            clouds.CloudCapability.COMPUTE)

        assert valid
        assert msg is None

    def test_get_api_key_prefers_env(self, monkeypatch, tmp_path):
        cred_path = tmp_path / 'api_key'
        cred_path.write_text('dtn_file')
        monkeypatch.setattr(daytona_utils, 'CREDENTIALS_PATH', str(cred_path))
        monkeypatch.setenv('DAYTONA_API_KEY', 'dtn_env')

        assert daytona_utils.get_api_key() == 'dtn_env'

    def test_get_api_key_missing_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daytona_utils, 'CREDENTIALS_PATH',
                            str(tmp_path / 'missing'))
        monkeypatch.delenv('DAYTONA_API_KEY', raising=False)

        with pytest.raises(daytona_utils.DaytonaError):
            daytona_utils.get_api_key()


class TestDaytonaStatusMapping:
    """Sandbox state to SkyPilot cluster status mapping."""

    @pytest.mark.parametrize('state,expected', [
        ('started', status_lib.ClusterStatus.UP),
        ('creating', status_lib.ClusterStatus.INIT),
        ('pulling_snapshot', status_lib.ClusterStatus.INIT),
        ('error', status_lib.ClusterStatus.INIT),
        ('stopped', status_lib.ClusterStatus.STOPPED),
        ('stopping', status_lib.ClusterStatus.STOPPED),
        ('archived', status_lib.ClusterStatus.STOPPED),
        ('destroyed', None),
        ('deleted', None),
        (None, None),
    ])
    def test_to_cluster_status(self, state, expected):
        assert daytona_utils.to_cluster_status(state) == expected


class TestDaytonaInstanceTypes:
    """Instance type parsing into sandbox create request bodies."""

    def test_gpu_instance_type(self):
        body = daytona_utils._instance_type_to_create_body(  # pylint: disable=protected-access
            '2x-H100', use_spot=False)
        assert body == {
            'gpu': 2,
            'gpuType': ['H100'],
            'cpu': 16,
            'memory': 200,
        }

    def test_gpu_instance_type_spot(self):
        body = daytona_utils._instance_type_to_create_body(  # pylint: disable=protected-access
            '1x-MI355X', use_spot=True)
        assert body['spot'] is True
        assert body['gpuType'] == ['MI355X']

    def test_gpu_instance_type_name_mapping(self):
        body = daytona_utils._instance_type_to_create_body(  # pylint: disable=protected-access
            '1x-RTXPRO6000',
            use_spot=False)
        assert body['gpuType'] == ['RTX-PRO-6000']

    def test_consumer_gpu_memory(self):
        body = daytona_utils._instance_type_to_create_body(  # pylint: disable=protected-access
            '4x-RTX4090', use_spot=False)
        assert body['cpu'] == 32
        assert body['memory'] == 200  # 50 GiB per consumer GPU.

    def test_cpu_instance_type(self):
        body = daytona_utils._instance_type_to_create_body(  # pylint: disable=protected-access
            'cpu-4x-16gb', use_spot=False)
        assert body == {'cpu': 4, 'memory': 16}

    def test_cpu_instance_type_spot_rejected(self):
        with pytest.raises(daytona_utils.DaytonaError):
            daytona_utils._instance_type_to_create_body(  # pylint: disable=protected-access
                'cpu-4x-16gb',
                use_spot=True)

    def test_unknown_gpu_rejected(self):
        with pytest.raises(daytona_utils.DaytonaError):
            daytona_utils._instance_type_to_create_body(  # pylint: disable=protected-access
                '1x-A100', use_spot=False)


class TestDaytonaCapacityCheck:
    """Fail-fast GPU capacity checking."""

    _CAPACITY = {
        'capacity': [{
            'gpuType': 'H100',
            'availableOnDemand': 3,
            'availableSpot': 0
        },]
    }

    @mock.patch.object(daytona_utils, '_request')
    @mock.patch.object(daytona_utils,
                       '_get_organization_id',
                       return_value='org-1')
    def test_sufficient_capacity_passes(self, mock_org, mock_request):
        mock_request.return_value = self._CAPACITY
        daytona_utils.check_gpu_capacity('H100', 2, use_spot=False)

    @mock.patch.object(daytona_utils, '_request')
    @mock.patch.object(daytona_utils,
                       '_get_organization_id',
                       return_value='org-1')
    def test_insufficient_capacity_raises(self, mock_org, mock_request):
        mock_request.return_value = self._CAPACITY
        with pytest.raises(daytona_utils.DaytonaError, match='Insufficient'):
            daytona_utils.check_gpu_capacity('H100', 4, use_spot=False)

    @mock.patch.object(daytona_utils, '_request')
    @mock.patch.object(daytona_utils,
                       '_get_organization_id',
                       return_value='org-1')
    def test_zero_spot_capacity_raises(self, mock_org, mock_request):
        mock_request.return_value = self._CAPACITY
        with pytest.raises(daytona_utils.DaytonaError, match='Insufficient'):
            daytona_utils.check_gpu_capacity('H100', 1, use_spot=True)

    @mock.patch.object(daytona_utils, '_request')
    @mock.patch.object(daytona_utils,
                       '_get_organization_id',
                       return_value='org-1')
    def test_absent_type_fails_open(self, mock_org, mock_request):
        # A missing capacity row is advisory-only: creation proceeds and
        # the create API decides.
        mock_request.return_value = self._CAPACITY
        daytona_utils.check_gpu_capacity('B300', 1, use_spot=False)

    @mock.patch.object(daytona_utils,
                       '_get_organization_id',
                       side_effect=daytona_utils.DaytonaError('down'))
    def test_capacity_endpoint_failure_is_ignored(self, mock_org):
        # The create call remains the source of truth.
        daytona_utils.check_gpu_capacity('H100', 1, use_spot=False)


class TestDaytonaProvisioning:
    """Provisioner behavior with a mocked Daytona API."""

    def test_run_instances_rejects_multi_node(self):
        config = mock.MagicMock()
        config.count = 2
        with pytest.raises(daytona_utils.DaytonaError):
            daytona_instance.run_instances('earth', 'cluster', 'cluster-abcd',
                                           config)

    @mock.patch.object(daytona_utils, 'wait_for_sandbox_started')
    @mock.patch.object(daytona_utils, 'launch_sandbox', return_value='sbx-1')
    @mock.patch.object(daytona_utils, 'list_cluster_sandboxes', return_value=[])
    def test_run_instances_creates_sandbox(self, mock_list, mock_launch,
                                           mock_wait):
        config = mock.MagicMock()
        config.count = 1
        config.node_config = {
            'InstanceType': '1x-H100',
            'DiskSize': 256,
            'ImageId': '',
            'Preemptible': False,
        }
        record = daytona_instance.run_instances('earth', 'cluster',
                                                'cluster-abcd', config)
        assert record.head_instance_id == 'sbx-1'
        assert record.created_instance_ids == ['sbx-1']
        mock_launch.assert_called_once_with(
            cluster_name_on_cloud='cluster-abcd',
            instance_type='1x-H100',
            region='earth',
            use_spot=False,
            disk_size=256,
            image_id=None,
            ttl_minutes=None,
            labels=None,
            cluster_display_name='cluster',
        )
        mock_wait.assert_called_once_with('sbx-1')

    @mock.patch.object(daytona_utils, 'wait_for_sandbox_started')
    @mock.patch.object(daytona_utils, 'list_cluster_sandboxes')
    def test_run_instances_reuses_existing(self, mock_list, mock_wait):
        mock_list.return_value = [{'id': 'sbx-1', 'state': 'started'}]
        config = mock.MagicMock()
        config.count = 1
        record = daytona_instance.run_instances('earth', 'cluster',
                                                'cluster-abcd', config)
        assert record.head_instance_id == 'sbx-1'
        assert not record.created_instance_ids

    @mock.patch.object(daytona_utils,
                       'create_ssh_token',
                       return_value='token-123')
    @mock.patch.object(daytona_utils, 'list_cluster_sandboxes')
    def test_get_cluster_info_uses_token_as_ssh_user(self, mock_list,
                                                     mock_token):
        mock_list.return_value = [{'id': 'sbx-1', 'state': 'started'}]
        info = daytona_instance.get_cluster_info('earth', 'cluster-abcd')
        assert info.head_instance_id == 'sbx-1'
        assert info.ssh_user == 'token-123'
        instance = info.instances['sbx-1'][0]
        assert instance.external_ip == daytona_utils.SSH_GATEWAY_HOST
        assert instance.ssh_port == daytona_utils.SSH_GATEWAY_PORT

    @mock.patch.object(daytona_utils, 'delete_sandbox')
    @mock.patch.object(daytona_utils, 'list_cluster_sandboxes')
    def test_terminate_instances(self, mock_list, mock_delete):
        mock_list.return_value = [
            {
                'id': 'sbx-1',
                'state': 'started'
            },
            {
                'id': 'sbx-2',
                'state': 'error'
            },
        ]
        daytona_instance.terminate_instances('cluster-abcd')
        assert mock_delete.call_count == 2

    def test_stop_instances_unsupported(self):
        with pytest.raises(NotImplementedError):
            daytona_instance.stop_instances('cluster-abcd')

    @mock.patch.object(daytona_utils, 'list_cluster_sandboxes')
    def test_query_instances(self, mock_list):
        mock_list.return_value = [
            {
                'id': 'sbx-1',
                'state': 'started',
                'errorReason': None
            },
            {
                'id': 'sbx-2',
                'state': 'destroyed',
                'errorReason': None
            },
        ]
        statuses = daytona_instance.query_instances('cluster', 'cluster-abcd')
        assert statuses == {'sbx-1': (status_lib.ClusterStatus.UP, None)}


class TestDaytonaCloudFeatures:
    """Cloud feature declarations."""

    def test_unsupported_features(self):
        features = daytona.Daytona._CLOUD_UNSUPPORTED_FEATURES  # pylint: disable=protected-access
        assert clouds.CloudImplementationFeatures.STOP in features
        assert clouds.CloudImplementationFeatures.MULTI_NODE in features
        assert clouds.CloudImplementationFeatures.AUTOSTOP in features
        # Ports are exposed as preview URLs.
        assert clouds.CloudImplementationFeatures.OPEN_PORTS not in features
        # Custom docker images are supported.
        assert (clouds.CloudImplementationFeatures.IMAGE_ID not in features)
        # Spot GPU sandboxes are supported.
        assert (clouds.CloudImplementationFeatures.SPOT_INSTANCE
                not in features)

    def test_repr(self):
        assert repr(daytona.Daytona()) == 'Daytona'


class TestDaytonaLaunchBody:
    """Disk handling in the sandbox create request."""

    def _launch_body(self, instance_type, disk_size):
        captured = {}

        def fake_request(method, path, params=None, body=None, timeout=None):
            del method, path, params, timeout
            captured.update(body)
            return {'id': 'sbx-1'}

        with mock.patch.object(daytona_utils, '_request', fake_request), \
                mock.patch.object(daytona_utils, 'check_gpu_capacity'):
            daytona_utils.launch_sandbox(
                cluster_name_on_cloud='c-1',
                instance_type=instance_type,
                region='us',
                use_spot=False,
                disk_size=disk_size,
            )
        return captured

    def test_gpu_disk_passthrough(self):
        body = self._launch_body('1x-H100', 256)
        assert body['disk'] == 256

    def test_cpu_explicit_disk_applied(self):
        body = self._launch_body('cpu-2x-8gb', 8)
        assert body['disk'] == 8

    def test_cpu_default_disk_omitted(self):
        from sky.resources import DEFAULT_DISK_SIZE_GB
        body = self._launch_body('cpu-2x-8gb', DEFAULT_DISK_SIZE_GB)
        assert 'disk' not in body


class TestDaytonaSpotDefaults:
    """Spot requests must not select CPU shapes."""

    def test_no_default_instance_type_for_spot(self):
        from sky.catalog import daytona_catalog
        assert daytona_catalog.get_default_instance_type(use_spot=True) is None


_json_loads = json.loads
_TOKEN = 'preview-token-do-not-log'
_SIGNED_TOKEN = 'signed-token-do-not-log'
_PROXY = 'proxy.daytona.test'


class _FakeResponse:

    def __init__(self, status_code, data=None, text=''):
        self.status_code = status_code
        self._data = data
        self.content = b'' if data is None else b'{}'
        self.text = text

    def json(self):
        return self._data


class FakeDaytonaApi:
    """The Daytona REST API in memory, standing in for `requests.request`."""

    def __init__(self):
        self.sandboxes = {}
        self.calls = []
        self.created = []
        self.delete_errors = []
        self.preview_urls_without_port_prefix = False
        self.sign_fails = False
        self.sign_expiries = []

    def add(self, sandbox_id, cluster, state='started', public=False):
        self.sandboxes[sandbox_id] = {
            'id': sandbox_id,
            'state': state,
            'public': public,
            'labels': {
                daytona_utils.CLUSTER_LABEL: cluster
            },
        }

    def count(self, method, path_regex):
        return sum(1 for m, path in self.calls
                   if m == method and re.search(path_regex, path))

    def request(self, method, url, headers=None, params=None, json=None, **_):  # pylint: disable=redefined-outer-name
        del headers
        path = url[len(daytona_utils.DEFAULT_API_URL):]
        self.calls.append((method, path))
        if method == 'POST' and path == '/sandbox':
            sandbox_id = f'sbx-{len(self.created) + 1}'
            self.created.append(json)
            self.sandboxes[sandbox_id] = {
                'id': sandbox_id,
                'state': 'started',
                'public': json.get('public', False),
                'labels': json['labels'],
            }
            return _FakeResponse(200, {'id': sandbox_id})
        if method == 'GET' and path == '/sandbox':
            wanted = _json_loads(params['labels'])
            return _FakeResponse(
                200, {
                    'items': [
                        s for s in self.sandboxes.values()
                        if wanted.items() <= s['labels'].items()
                    ]
                })
        if path == '/api-keys/current':
            return _FakeResponse(200, {'organizationId': 'org-1'})
        if path == '/organizations/org-1/gpu-capacity':
            return _FakeResponse(
                200, {
                    'capacity': [{
                        'gpuType': 'H100',
                        'availableOnDemand': 8,
                        'availableSpot': 8,
                    }]
                })
        match = re.fullmatch(r'/sandbox/([^/]+)(/.*)?', path)
        sandbox = self.sandboxes.get(match.group(1))
        action = match.group(2)
        if method == 'DELETE' and self.delete_errors:
            status = self.delete_errors.pop(0)
            return _FakeResponse(status, text=f'error on {path}')
        if sandbox is None:
            return _FakeResponse(404, text='not found')
        if method == 'DELETE':
            del self.sandboxes[sandbox['id']]
            return _FakeResponse(200)
        if method == 'POST' and action == '/start':
            sandbox['state'] = 'started'
            return _FakeResponse(200, sandbox)
        if method == 'POST' and action == '/ssh-access':
            return _FakeResponse(200, {'token': 'ssh-token'})
        if method == 'GET' and action is None:
            return _FakeResponse(200, sandbox)
        signed = re.fullmatch(r'/ports/(\d+)/signed-preview-url', action or '')
        if method == 'GET' and signed:
            if self.sign_fails:
                return _FakeResponse(400, text='cannot sign')
            self.sign_expiries.append(params['expiresInSeconds'])
            return _FakeResponse(
                200,
                {'url': f'https://{signed.group(1)}-{_SIGNED_TOKEN}.{_PROXY}'})
        port = re.fullmatch(r'/ports/(\d+)/preview-url', action or '')
        if method == 'GET' and port:
            host = (f'p{port.group(1)}'
                    if self.preview_urls_without_port_prefix else port.group(1))
            return _FakeResponse(
                200, {
                    'sandboxId': sandbox['id'],
                    'url': f'https://{host}-{sandbox["id"]}.{_PROXY}',
                    'token': _TOKEN,
                })
        return _FakeResponse(500, text=f'unhandled {method} {path}')


@pytest.fixture(name='api')
def fixture_api(monkeypatch):
    fake = FakeDaytonaApi()
    monkeypatch.setenv('DAYTONA_API_KEY', 'dtn_test')
    monkeypatch.delenv('DAYTONA_API_URL', raising=False)
    monkeypatch.setattr(daytona_utils.requests, 'request', fake.request)
    monkeypatch.setattr(daytona_utils.time, 'sleep', lambda _: None)
    daytona_utils._get_organization_id.cache_clear()  # pylint: disable=protected-access
    return fake


def _provision_config(**node_config):
    config = mock.MagicMock()
    config.count = 1
    config.node_config = {'InstanceType': '1x-H100', **node_config}
    return config


class TestDaytonaPorts:
    """Preview URLs and tokens, through the routed provision API."""

    def test_private_sandbox_gets_signed_urls_and_token(self, api):
        api.add('sbx-1', 'c-1')
        endpoints = provision.query_ports('Daytona', 'c-1', ['8080', '9000'])
        assert sorted(endpoints) == [8080, 9000]
        assert endpoints[8080][0].url() == (
            f'https://8080-{_SIGNED_TOKEN}.{_PROXY}')
        assert endpoints[9000][0].url() == (
            f'https://9000-{_SIGNED_TOKEN}.{_PROXY}')
        assert endpoints[8080][0].token == _TOKEN
        assert endpoints[9000][0].token == _TOKEN

    def test_signed_url_lifetime_comes_from_config(self, api):
        api.add('sbx-1', 'c-1')
        provision.query_ports(
            'Daytona',
            'c-1', ['8080'],
            provider_config={'preview_url_expires_seconds': 600})
        provision.query_ports('Daytona', 'c-1', ['8080'])
        assert api.sign_expiries == [
            600, daytona_utils.DEFAULT_PREVIEW_URL_EXPIRES_SECONDS
        ]

    def test_signing_failure_falls_back_to_plain_url_and_token(self, api):
        api.add('sbx-1', 'c-1')
        api.sign_fails = True
        endpoints = provision.query_ports('Daytona', 'c-1', ['8080-8081'])
        assert endpoints[8081][0].url() == f'https://8081-sbx-1.{_PROXY}'
        assert endpoints[8081][0].token == _TOKEN

    def test_public_sandbox_needs_neither_signing_nor_token(self, api):
        api.add('sbx-1', 'c-1', public=True)
        endpoints = provision.query_ports('Daytona', 'c-1', ['8080'])
        assert endpoints[8080][0].url() == f'https://8080-sbx-1.{_PROXY}'
        assert endpoints[8080][0].token is None
        assert not api.sign_expiries

    def test_sandbox_with_unknown_visibility_keeps_token(self, api):
        api.add('sbx-1', 'c-1')
        del api.sandboxes['sbx-1']['public']
        endpoints = provision.query_ports('Daytona', 'c-1', ['8080'])
        assert endpoints[8080][0].token == _TOKEN

    @pytest.mark.parametrize('state', ['creating', 'stopped'])
    def test_not_started_has_no_endpoints(self, api, state):
        api.add('sbx-1', 'c-1', state=state)
        assert provision.query_ports('Daytona', 'c-1', ['8080']) == {}

    def test_no_sandbox_has_no_endpoints(self, api):
        assert provision.query_ports('Daytona', 'c-1', ['8080']) == {}

    def test_other_clusters_sandbox_ignored(self, api):
        api.add('sbx-1', 'other')
        assert provision.query_ports('Daytona', 'c-1', ['8080']) == {}

    def test_wide_port_range_takes_one_api_call(self, api):
        api.add('sbx-1', 'c-1')
        endpoints = provision.query_ports('Daytona', 'c-1', ['8000-8999'])
        assert len(endpoints) == 1000
        assert endpoints[8500][0].url() == f'https://8500-sbx-1.{_PROXY}'
        assert endpoints[8500][0].token == _TOKEN
        assert api.count('GET', 'preview-url') == 1

    def test_unexpected_url_form_asks_the_api(self, api):
        api.add('sbx-1', 'c-1')
        api.sign_fails = True
        api.preview_urls_without_port_prefix = True
        endpoints = provision.query_ports('Daytona', 'c-1', ['8080', '9000'])
        assert endpoints[9000][0].url() == f'https://p9000-sbx-1.{_PROXY}'

    def test_tokens_stay_out_of_logs_and_text(self, api):
        api.add('sbx-1', 'c-1')
        records = []

        class _Capture(logging.Handler):

            def emit(self, record):
                records.append(record.getMessage())

        root = logging.getLogger('sky')
        handler = _Capture(level=logging.DEBUG)
        previous_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        try:
            endpoints = provision.query_ports('Daytona', 'c-1', ['8080'])
            api.sign_fails = True
            fallback = provision.query_ports('Daytona', 'c-1', ['8080'])
        finally:
            root.removeHandler(handler)
            root.setLevel(previous_level)
        assert endpoints[8080][0].token == _TOKEN
        for text in records + [repr(endpoints), str(fallback)]:
            assert _TOKEN not in text
            assert _SIGNED_TOKEN not in text
        assert _TOKEN not in endpoints[8080][0].url()

    def test_get_endpoints_leaves_the_sandbox_token_out(self, api):
        api.add('sbx-1', 'c-1')
        handle = mock.MagicMock(
            spec=cloud_vm_ray_backend.CloudVmRayResourceHandle)
        handle.cluster_name_on_cloud = 'c-1'
        handle.head_ip = daytona_utils.SSH_GATEWAY_HOST
        handle.launched_resources = resources_lib.Resources(
            cloud=daytona.Daytona(), instance_type='1x-H100', ports=['8080'])
        record = {
            'name': 'c',
            'status': status_lib.ClusterStatus.UP,
            'handle': handle,
        }
        with mock.patch.object(backend_utils,
                               'get_clusters',
                               return_value=[record]), mock.patch.object(
                                   backend_utils.global_user_state,
                                   'get_cluster_yaml_dict',
                                   return_value={'provider': {}}):
            result = backend_utils.get_endpoints('c')
            single = backend_utils.get_endpoints('c', port=8080)
        assert result == {8080: f'https://8080-{_SIGNED_TOKEN}.{_PROXY}'}
        assert single == result

    def test_open_and_cleanup_ports_take_the_backends_arguments(self, api):
        provision.open_ports('Daytona', 'c-1', ['8080'], {})
        provision.cleanup_ports('Daytona', 'c-1', ['8080'], {})
        assert not api.calls

    @pytest.mark.parametrize('port', ['22', '80', '3000', '10000', '65535'])
    def test_ports_outside_the_old_preview_range_are_allowed(self, port):
        provision.open_ports('Daytona', 'c-1', [port], {})
        resources = resources_lib.Resources(infra='daytona',
                                            accelerators='H100:1',
                                            ports=[port])
        feasible = daytona.Daytona()._get_feasible_launchable_resources(  # pylint: disable=protected-access
            resources)
        assert feasible.resources_list

    @pytest.mark.parametrize('ports', [['2280'], ['22222'], ['33000-34000']])
    def test_reserved_ports_infeasible_before_launch(self, ports):
        resources = resources_lib.Resources(infra='daytona',
                                            accelerators='H100:1',
                                            ports=ports)
        feasible = daytona.Daytona()._get_feasible_launchable_resources(  # pylint: disable=protected-access
            resources)
        assert not feasible.resources_list
        assert 'reserved' in feasible.hint
        with pytest.raises(daytona_utils.DaytonaError, match='reserved'):
            provision.open_ports('Daytona', 'c-1', ports, {})

    def test_reserved_port_in_a_query_is_skipped(self, api):
        api.add('sbx-1', 'c-1')
        assert sorted(provision.query_ports('Daytona', 'c-1',
                                            ['2280', '8080'])) == [8080]


class TestDaytonaLifecycle:
    """Provision, status and teardown against the fake API."""

    def test_provision_status_down(self, api):
        record = daytona_instance.run_instances('earth', 'c', 'c-1',
                                                _provision_config())
        assert record.created_instance_ids == ['sbx-1']
        assert daytona_instance.query_instances('c', 'c-1') == {
            'sbx-1': (status_lib.ClusterStatus.UP, None)
        }
        info = daytona_instance.get_cluster_info('earth', 'c-1')
        assert info.head_instance_id == 'sbx-1'
        assert info.ssh_user == 'ssh-token'
        daytona_instance.terminate_instances('c-1')
        assert daytona_instance.query_instances('c', 'c-1') == {}
        assert not provision.query_ports('Daytona', 'c-1', ['8080'])
        # Tearing down twice is harmless.
        daytona_instance.terminate_instances('c-1')

    def test_sandbox_destroyed_by_ttl_reads_as_terminated(self, api):
        api.add('sbx-1', 'c-1', state='destroyed')
        assert not daytona_instance.query_instances('c', 'c-1')

    def test_second_launch_reuses_sandbox(self, api):
        daytona_instance.run_instances('earth', 'c', 'c-1', _provision_config())
        record = daytona_instance.run_instances('earth', 'c', 'c-1',
                                                _provision_config())
        assert record.head_instance_id == 'sbx-1'
        assert not record.created_instance_ids
        assert len(api.created) == 1

    @pytest.mark.parametrize('state', ['stopped', 'archived'])
    def test_stopped_sandbox_is_started_again(self, api, state):
        api.add('sbx-1', 'c-1', state=state)
        record = daytona_instance.run_instances('earth', 'c', 'c-1',
                                                _provision_config())
        assert record.resumed_instance_ids == ['sbx-1']
        assert api.sandboxes['sbx-1']['state'] == 'started'
        assert api.count('POST', '/start$') == 1

    def test_sandbox_stopping_is_started_once_stopped(self, api):
        api.add('sbx-1', 'c-1', state='stopping')
        get_sandbox = daytona_utils.get_sandbox
        polls = []

        def settle(sandbox_id):
            polls.append(sandbox_id)
            if len(polls) == 2:
                api.sandboxes[sandbox_id]['state'] = 'stopped'
            return get_sandbox(sandbox_id)

        with mock.patch.object(daytona_utils, 'get_sandbox', settle):
            record = daytona_instance.run_instances('earth', 'c', 'c-1',
                                                    _provision_config())
        assert record.resumed_instance_ids == ['sbx-1']
        assert api.count('POST', '/start$') == 1

    def test_errored_sandbox_fails_the_launch(self, api):
        api.add('sbx-1', 'c-1', state='error')
        with pytest.raises(daytona_utils.DaytonaError, match='error'):
            daytona_instance.run_instances('earth', 'c', 'c-1',
                                           _provision_config())

    def test_stop_on_platform_reads_as_stopped_then_gone(self, api):
        # A GPU sandbox is ephemeral: it is deleted once it stops.
        api.add('sbx-1', 'c-1', state='stopped')
        assert daytona_instance.query_instances('c', 'c-1') == {
            'sbx-1': (status_lib.ClusterStatus.STOPPED, None)
        }
        del api.sandboxes['sbx-1']
        assert daytona_instance.query_instances('c', 'c-1') == {}

    def test_worker_only_termination_keeps_the_head(self, api):
        api.add('sbx-1', 'c-1')
        provision.terminate_instances('Daytona', 'c-1', {}, worker_only=True)
        assert 'sbx-1' in api.sandboxes

    def test_delete_tolerates_already_deleted(self, api):
        daytona_utils.delete_sandbox('sbx-gone')

    def test_delete_retries_while_state_changes(self, api):
        api.add('sbx-1', 'c-1')
        api.delete_errors = [409, 409]
        daytona_utils.delete_sandbox('sbx-1')
        assert 'sbx-1' not in api.sandboxes

    def test_delete_gives_up_on_a_lasting_conflict(self, api):
        api.add('sbx-1', 'c-1')
        api.delete_errors = [409] * 50
        with pytest.raises(daytona_utils.DaytonaError):
            daytona_utils.delete_sandbox('sbx-1')

    def test_delete_failure_on_id_containing_404_is_raised(self, api):
        # The status decides, not digits that happen to be in the message.
        api.add('sbx-404', 'c-1')
        api.delete_errors = [500]
        with pytest.raises(daytona_utils.DaytonaError):
            daytona_utils.delete_sandbox('sbx-404')

    def test_autodown_on_the_cluster_deletes_the_sandbox(self, api):
        api.add('sbx-1', 'c-1')
        event = events.StopEvent()
        autostop_config = mock.MagicMock(down=True)
        cluster_config = {
            'cluster_name': 'c-1',
            'max_workers': 0,
            'provider': {
                'type': 'external',
                'module': 'sky.provision.daytona',
            },
        }
        with mock.patch.dict(os.environ), mock.patch.object(
                events.subprocess,
                'run'), mock.patch.object(event,
                                          '_execute_hook_if_present'), \
                mock.patch.object(events.autostop_lib,
                                  'set_autostopping_started'):
            event._stop_cluster_with_new_provisioner(  # pylint: disable=protected-access
                autostop_config, cluster_config, 'daytona', daytona.Daytona())
        assert not api.sandboxes

    def test_post_teardown_cleanup_succeeds_with_ports(self, api):
        # A cleanup_ports failure would make every `sky down` fail.
        handle = mock.MagicMock(
            spec=cloud_vm_ray_backend.CloudVmRayResourceHandle)
        handle.cluster_name = 'c'
        handle.cluster_name_on_cloud = 'c-1'
        handle.launched_resources = resources_lib.Resources(
            cloud=daytona.Daytona(), instance_type='1x-H100', ports=['8080'])
        backend = cloud_vm_ray_backend.CloudVmRayBackend()
        state = cloud_vm_ray_backend.global_user_state
        with mock.patch.object(
                state, 'get_cluster_yaml_dict',
                return_value={'provider': {}}), mock.patch.object(
                    state, 'remove_cluster') as remove_cluster, \
                mock.patch.object(backend, 'remove_cluster_config'), \
                mock.patch.object(
                    cloud_vm_ray_backend.cluster_utils.SSHConfigHelper,
                    'remove_cluster'):
            backend.post_teardown_cleanup(handle, terminate=True)
        remove_cluster.assert_called_once()

    def test_multinode_rejected(self):
        with pytest.raises(exceptions.NotSupportedError):
            resources = resources_lib.Resources(cloud=daytona.Daytona(),
                                                instance_type='1x-H100')
            daytona.Daytona.check_features_are_supported(
                resources, {clouds.CloudImplementationFeatures.MULTI_NODE})


class TestDaytonaAutostopFeatures:
    """What `sky autostop`, `sky stop` and managed jobs are allowed to do."""

    def _resources(self):
        return resources_lib.Resources(cloud=daytona.Daytona(),
                                       instance_type='1x-H100')

    def test_autodown_allowed(self):
        # Managed jobs launch their clusters with down=True.
        daytona.Daytona.check_features_are_supported(
            self._resources(), {clouds.CloudImplementationFeatures.AUTODOWN})

    @pytest.mark.parametrize('feature', [
        clouds.CloudImplementationFeatures.STOP,
        clouds.CloudImplementationFeatures.AUTOSTOP,
    ])
    def test_stop_refused_rather_than_silently_deleting(self, feature):
        with pytest.raises(exceptions.NotSupportedError):
            daytona.Daytona.check_features_are_supported(
                self._resources(), {feature})


class TestDaytonaCredentialMounts:
    """The cluster and the jobs controller need the key as a file."""

    def _use_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv('HOME', str(tmp_path))

    def test_env_key_written_privately_and_mounted(self, monkeypatch, tmp_path):
        self._use_home(monkeypatch, tmp_path)
        monkeypatch.setenv('DAYTONA_API_KEY', ' dtn_env\n')
        mounts = daytona.Daytona.get_credential_file_mounts()
        local_path = mounts['~/.daytona/api_key']
        with open(local_path, encoding='utf-8') as f:
            assert f.read() == 'dtn_env'
        assert stat.S_IMODE(os.stat(local_path).st_mode) == 0o600
        # The user's own file is not touched.
        assert not (tmp_path / '.daytona').exists()

    def test_file_key_mounted(self, monkeypatch, tmp_path):
        self._use_home(monkeypatch, tmp_path)
        monkeypatch.delenv('DAYTONA_API_KEY', raising=False)
        key = tmp_path / '.daytona' / 'api_key'
        key.parent.mkdir()
        key.write_text('dtn_file')
        assert daytona.Daytona.get_credential_file_mounts() == {
            '~/.daytona/api_key': str(key)
        }

    def test_no_key_no_mount(self, monkeypatch, tmp_path):
        self._use_home(monkeypatch, tmp_path)
        monkeypatch.delenv('DAYTONA_API_KEY', raising=False)
        assert daytona.Daytona.get_credential_file_mounts() == {}


class TestDaytonaGpuAndCatalog:
    """GPU mapping and the catalog that ships with the fork."""

    def _rows(self):
        path = catalog_common.get_bundled_catalog_path('daytona/vms.csv')
        with open(path, encoding='utf-8') as f:
            return list(csv.DictReader(f))

    @pytest.mark.parametrize('gpu', ['H100', 'H200', 'B200', 'B300'])
    def test_datacenter_gpu_maps_to_api_type(self, gpu):
        body = daytona_utils._instance_type_to_create_body(  # pylint: disable=protected-access
            f'8x-{gpu}', use_spot=True)
        assert body == {
            'gpu': 8,
            'gpuType': [gpu],
            'cpu': 64,
            'memory': 800,
            'spot': True,
        }

    @pytest.mark.parametrize('gpu', ['H100', 'H200', 'B200', 'B300'])
    def test_datacenter_gpu_offered_in_catalog(self, gpu):
        rows = [r for r in self._rows() if r['AcceleratorName'] == gpu]
        assert sorted(int(r['AcceleratorCount']) for r in rows) == list(
            range(1, 9))
        assert all(float(r['Price']) > 0 for r in rows)
        assert all(float(r['SpotPrice']) > 0 for r in rows)
        assert {r['Region'] for r in rows} == {'earth'}

    def test_every_catalog_row_can_be_created_as_priced(self):
        for row in self._rows():
            body = daytona_utils._instance_type_to_create_body(  # pylint: disable=protected-access
                row['InstanceType'],
                use_spot=False)
            assert body['cpu'] == float(row['vCPUs'])
            assert body['memory'] == float(row['MemoryGiB'])
            if row['AcceleratorName']:
                assert body['gpu'] == int(row['AcceleratorCount'])

    def test_bundled_catalog_needs_no_download(self, monkeypatch, tmp_path):
        monkeypatch.setattr(catalog_common, '_ABSOLUTE_VERSIONED_CATALOG_DIR',
                            str(tmp_path))
        with mock.patch.object(catalog_common.requests, 'get') as mock_get:
            df = catalog_common.read_catalog('daytona/vms.csv')
            assert '1x-B200' in list(df['InstanceType'])
        mock_get.assert_not_called()

    def test_prices_come_from_the_bundled_catalog(self):
        on_demand = daytona_catalog.get_hourly_cost('1x-RTX4090',
                                                    use_spot=False)
        spot = daytona_catalog.get_hourly_cost('1x-RTX4090', use_spot=True)
        assert 0 < spot < on_demand

    def test_catalog_answers_gpu_requests(self):
        instances, _ = sky.catalog.get_instance_type_for_accelerator(
            'B200', 2, use_spot=True, clouds='daytona')
        assert instances == ['2x-B200']


class TestDaytonaLabelsAndTtl:
    """Labels and TTL from config to the sandbox create request."""

    def _node_config(self, **variables):
        path = os.path.join(os.path.dirname(sky.__file__), 'templates',
                            'daytona-ray.yml.j2')
        with open(path, encoding='utf-8') as f:
            template = jinja2.Template(f.read())
        defaults = {
            'num_nodes': 1,
            'instance_type': '1x-H100',
            'disk_size': 256,
            'image_id': '',
            'use_spot': False,
            'ttl_minutes': 0,
            'preview_url_expires_seconds': 3600,
            'region': 'earth',
            'labels': {},
            'user': 'alice',
            'credentials': {},
            'initial_setup_commands': [],
        }
        rendered = template.render(**{**defaults, **variables})
        return yaml.safe_load(
            rendered)['available_node_types']['ray_head_default']['node_config']

    def test_template_passes_labels_and_ttl(self):
        node_config = self._node_config(labels={
            'team': 'mobius',
            'odd: key': 'yes'
        },
                                        ttl_minutes=30)
        assert node_config['TtlMinutes'] == 30
        assert node_config['Labels'] == {
            'skypilot-user': 'alice',
            'team': 'mobius',
            'odd: key': 'yes',
        }

    def test_create_request_carries_labels_and_ttl(self, api):
        config = _provision_config(Labels={
            'team': 'mobius',
            'count': 3
        },
                                   TtlMinutes=30)
        daytona_instance.run_instances('earth', 'my cluster', 'c-1', config)
        body = api.created[0]
        assert body['ttlMinutes'] == 30
        assert body['public'] is False
        assert body['labels'] == {
            'team': 'mobius',
            'count': '3',
            'skypilot-cluster-name': 'my cluster',
            'skypilot-cluster': 'c-1',
        }

    def test_no_ttl_by_default(self, api):
        daytona_instance.run_instances('earth', 'c', 'c-1',
                                       _provision_config(TtlMinutes=0))
        assert 'ttlMinutes' not in api.created[0]

    def test_user_label_cannot_hide_the_sandbox_from_skypilot(self, api):
        config = _provision_config(Labels={'skypilot-cluster': 'other'})
        daytona_instance.run_instances('earth', 'c', 'c-1', config)
        assert daytona_instance.query_instances('c', 'c-1')

    def test_ttl_from_config_and_task_override(self):
        cloud = daytona.Daytona()
        region = clouds.Region('earth')
        resources = resources_lib.Resources(cloud=cloud,
                                            instance_type='1x-H100',
                                            region='earth')
        with skypilot_config.override_skypilot_config(
            {'daytona': {
                'ttl_minutes': 60
            }}):
            variables = cloud.make_deploy_resources_variables(
                resources, mock.MagicMock(), region, None, 1)
            assert variables['ttl_minutes'] == 60
            overridden = resources.copy(_cluster_config_overrides={
                'daytona': {
                    'ttl_minutes': 5,
                    'preview_url_expires_seconds': 900
                }
            })
            variables = cloud.make_deploy_resources_variables(
                overridden, mock.MagicMock(), region, None, 1)
            assert variables['ttl_minutes'] == 5
            assert variables['preview_url_expires_seconds'] == 900

    def test_config_schema(self):
        common_utils.validate_schema(
            {
                'daytona': {
                    'ttl_minutes': 30,
                    'preview_url_expires_seconds': 600,
                    'labels': {
                        'team': 'mobius'
                    }
                }
            }, schemas.get_config_schema(), 'Invalid config: ')
        with pytest.raises(ValueError):
            common_utils.validate_schema({'daytona': {
                'ttl_minutes': -1
            }}, schemas.get_config_schema(), 'Invalid config: ')
