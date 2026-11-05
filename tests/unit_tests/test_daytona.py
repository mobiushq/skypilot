"""Tests for the Daytona cloud provider."""

import json
from unittest import mock

import pytest

from sky import clouds
from sky import resources as resources_lib
from sky.catalog import common as catalog_common
from sky.clouds import daytona
from sky.provision.daytona import instance as daytona_instance
from sky.provision.daytona import utils as daytona_utils
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
                '1x-B200', use_spot=False)


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

    def test_open_ports_checks_preview_range(self):
        daytona_instance.open_ports('cluster-abcd', ['8080', '3000-3002'])
        with pytest.raises(daytona_utils.DaytonaError, match='3000-9999'):
            daytona_instance.open_ports('cluster-abcd', ['22'])

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


class FakeDaytonaAPI:
    """In-memory stand-in for the Daytona REST API behind `_request`."""

    def __init__(self, sign_fails=False):
        self.sandboxes = {}
        self.created = []
        self.sign_fails = sign_fails
        self._next = 0

    def __call__(self, method, path, params=None, body=None, timeout=None):
        del timeout
        parts = path.strip('/').split('/')
        if path == '/api-keys/current':
            return {'organizationId': 'org-1'}
        if path.endswith('/gpu-capacity'):
            return {'capacity': []}
        if method == 'POST' and path == '/sandbox':
            self._next += 1
            sandbox_id = f'sbx-{self._next}'
            self.sandboxes[sandbox_id] = {
                'id': sandbox_id,
                'state': 'started',
                'labels': body['labels'],
            }
            self.created.append(body)
            return {'id': sandbox_id}
        if method == 'GET' and path == '/sandbox':
            want = json.loads(params['labels'])
            return [
                s for s in self.sandboxes.values()
                if all(s['labels'].get(k) == v for k, v in want.items())
            ]
        sandbox_id = parts[1]
        if sandbox_id not in self.sandboxes:
            raise daytona_utils.DaytonaError(
                f'Daytona API error 404 on {method} {path}')
        if method == 'DELETE':
            del self.sandboxes[sandbox_id]
            return None
        if parts[2:3] == ['ssh-access']:
            return {'token': f'ssh-{sandbox_id}'}
        if parts[2:3] == ['ports']:
            port = parts[3]
            if parts[4] == 'signed-preview-url':
                if self.sign_fails:
                    raise daytona_utils.DaytonaError('Daytona API error 400')
                assert params['expiresInSeconds'] > 0
                return {
                    'url': f'https://{port}-signed.proxy.daytona.works',
                    'token': 'signed',
                    'port': int(port),
                    'sandboxId': sandbox_id,
                }
            return {
                'url': f'https://{port}-{sandbox_id}.proxy.daytona.works',
                'token': 'preview-token',
                'sandboxId': sandbox_id,
            }
        return self.sandboxes[sandbox_id]


def _provision_config(**node_config):
    config = mock.MagicMock()
    config.count = 1
    config.node_config = {
        'InstanceType': '1x-RTX4090',
        'DiskSize': 50,
        'ImageId': '',
        'Preemptible': False,
        **node_config,
    }
    return config


class TestDaytonaLifecycle:
    """Provision, status, endpoints and teardown against a fake API."""

    def test_full_lifecycle(self):
        api = FakeDaytonaAPI()
        with mock.patch.object(daytona_utils, '_request', api):
            daytona_utils._get_organization_id.cache_clear()  # pylint: disable=protected-access
            record = daytona_instance.run_instances(
                'earth', 'c', 'c-abcd', _provision_config(TtlMinutes=30))
            sandbox_id = record.head_instance_id
            assert record.created_instance_ids == [sandbox_id]
            body = api.created[0]
            assert body['ttlMinutes'] == 30
            assert body['labels'] == {daytona_utils.CLUSTER_LABEL: 'c-abcd'}
            assert body['autoStopInterval'] == 0

            # Launching again reuses the sandbox.
            again = daytona_instance.run_instances('earth', 'c', 'c-abcd',
                                                   _provision_config())
            assert not again.created_instance_ids
            assert len(api.created) == 1

            assert daytona_instance.query_instances('c', 'c-abcd') == {
                sandbox_id: (status_lib.ClusterStatus.UP, None)
            }
            info = daytona_instance.get_cluster_info('earth', 'c-abcd')
            assert info.ssh_user == f'ssh-{sandbox_id}'

            endpoints = daytona_instance.query_ports(
                'c-abcd', ['8080'],
                provider_config={'preview_url_expires_seconds': 600})
            assert endpoints[8080][0].url() == (
                'https://8080-signed.proxy.daytona.works/')

            daytona_instance.terminate_instances('c-abcd')
            assert not api.sandboxes
            assert not daytona_instance.query_instances('c', 'c-abcd')
            assert not daytona_instance.query_ports('c-abcd', ['8080'])
            # Tearing down twice is harmless.
            daytona_instance.terminate_instances('c-abcd')

    def test_no_ttl_by_default(self):
        api = FakeDaytonaAPI()
        with mock.patch.object(daytona_utils, '_request', api):
            daytona_instance.run_instances('earth', 'c', 'c-abcd',
                                           _provision_config(TtlMinutes=0))
        assert 'ttlMinutes' not in api.created[0]

    def test_sandbox_destroyed_by_ttl_reads_as_terminated(self):
        api = FakeDaytonaAPI()
        with mock.patch.object(daytona_utils, '_request', api):
            daytona_instance.run_instances('earth', 'c', 'c-abcd',
                                           _provision_config(TtlMinutes=5))
            for sandbox in api.sandboxes.values():
                sandbox['state'] = 'destroyed'
            assert not daytona_instance.query_instances('c', 'c-abcd')


class TestDaytonaPorts:
    """Ports are exposed as Daytona preview URLs."""

    def test_signed_url_failure_falls_back_to_token_url(self):
        api = FakeDaytonaAPI(sign_fails=True)
        with mock.patch.object(daytona_utils, '_request', api):
            daytona_instance.run_instances('earth', 'c', 'c-abcd',
                                           _provision_config())
            endpoints = daytona_instance.query_ports('c-abcd', ['8080-8081'])
            assert sorted(endpoints) == [8080, 8081]
            assert endpoints[8080][0].url() == (
                'https://8080-sbx-1.proxy.daytona.works/')
            assert daytona_utils.get_preview_url(
                'sbx-1', 8080) == ('https://8080-sbx-1.proxy.daytona.works',
                                   'preview-token')

    def test_ports_outside_preview_range_are_skipped(self):
        api = FakeDaytonaAPI()
        with mock.patch.object(daytona_utils, '_request', api):
            daytona_instance.run_instances('earth', 'c', 'c-abcd',
                                           _provision_config())
            assert sorted(daytona_instance.query_ports(
                'c-abcd', ['22', '8080'])) == [8080]

    def test_only_started_sandboxes_are_queried(self):
        api = FakeDaytonaAPI()
        with mock.patch.object(daytona_utils, '_request', api):
            daytona_instance.run_instances('earth', 'c', 'c-abcd',
                                           _provision_config())
            api.sandboxes['sbx-1']['state'] = 'starting'
            assert not daytona_instance.query_ports('c-abcd', ['8080'])

    def test_endpoint_keeps_scheme_and_path(self):
        endpoint = daytona_instance._endpoint(  # pylint: disable=protected-access
            'http://localhost:4000/proxy/8080?token=x')
        assert endpoint.url() == 'http://localhost:4000/proxy/8080?token=x'

    def test_feasibility_rejects_ports_outside_range(self):
        r = resources_lib.Resources(infra='daytona',
                                    accelerators='RTX4090:1',
                                    ports=['22'])
        feasible = daytona.Daytona()._get_feasible_launchable_resources(r)  # pylint: disable=protected-access
        assert not feasible.resources_list
        assert '3000-9999' in feasible.hint

    def test_deploy_variables_carry_ttl_and_expiry(self):
        r = resources_lib.Resources(infra='daytona/earth',
                                    instance_type='1x-RTX4090')
        overrides = {
            'daytona': {
                'ttl_minutes': 20,
                'preview_url_expires_seconds': 900
            }
        }
        r = r.copy(_cluster_config_overrides=overrides)
        variables = daytona.Daytona().make_deploy_resources_variables(
            r, mock.MagicMock(), clouds.Region('earth'), None, 1)
        assert variables['ttl_minutes'] == 20
        assert variables['preview_url_expires_seconds'] == 900


class TestDaytonaBundledCatalog:
    """The fork ships the Daytona catalog; nothing is fetched."""

    def test_catalog_is_bundled(self):
        assert catalog_common.get_bundled_catalog_path('daytona/vms.csv')

    def test_prices_come_from_the_bundled_catalog(self):
        from sky.catalog import daytona_catalog
        on_demand = daytona_catalog.get_hourly_cost('1x-RTX4090',
                                                    use_spot=False)
        spot = daytona_catalog.get_hourly_cost('1x-RTX4090', use_spot=True)
        assert 0 < spot < on_demand
