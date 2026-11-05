"""Modal provider implementation for SkyPilot.

Modal (https://modal.com) runs sandboxes: containers with per-second billing
for the GPU, CPU and memory they request. A SkyPilot cluster on Modal is one
sandbox that SkyPilot owns, and commands reach it through `Sandbox.exec`
rather than SSH.
"""

import typing
from typing import Any, Dict, List, Optional, Tuple, Union

from sky import catalog
from sky import clouds
from sky import skypilot_config
from sky.provision.modal import utils as modal_utils
from sky.utils import registry
from sky.utils import resources_utils
from sky.utils.resources_utils import DiskTier

if typing.TYPE_CHECKING:
    from sky import resources as resources_lib
    from sky.utils import volume as volume_lib


@registry.CLOUD_REGISTRY.register
class Modal(clouds.Cloud):
    """Modal sandbox cloud provider."""

    _REPR = 'Modal'
    _MAX_CLUSTER_NAME_LEN_LIMIT = 60

    _CLOUD_UNSUPPORTED_FEATURES = {
        clouds.CloudImplementationFeatures.STOP:
            (f'Stopping clusters is not supported on {_REPR}; sandboxes '
             'can only be terminated.'),
        clouds.CloudImplementationFeatures.AUTOSTOP:
            (f'Auto-stop is not supported on {_REPR}.'),
        clouds.CloudImplementationFeatures.AUTO_TERMINATE:
            (f'Auto-terminate is not supported on {_REPR}. Use '
             '`modal.ttl_minutes` to bound a cluster\'s lifetime.'),
        clouds.CloudImplementationFeatures.AUTODOWN:
            (f'Auto-down is not supported on {_REPR}. Use '
             '`modal.ttl_minutes` to bound a cluster\'s lifetime.'),
        clouds.CloudImplementationFeatures.MULTI_NODE:
            (f'Multi-node clusters are not supported on {_REPR}. '
             'Each cluster runs on one sandbox.'),
        clouds.CloudImplementationFeatures.SPOT_INSTANCE:
            (f'Spot instances are not supported on {_REPR}.'),
        clouds.CloudImplementationFeatures.OPEN_PORTS:
            (f'Opening ports is not supported on {_REPR}.'),
        clouds.CloudImplementationFeatures.STORAGE_MOUNTING:
            (f'Storage mounting is not supported on {_REPR}.'),
        clouds.CloudImplementationFeatures.CUSTOM_DISK_TIER:
            (f'Custom disk tier is not supported on {_REPR}.'),
        clouds.CloudImplementationFeatures.CUSTOM_NETWORK_TIER:
            (f'Custom network tier is not supported on {_REPR}.'),
        clouds.CloudImplementationFeatures.HOST_CONTROLLERS:
            (f'Running controllers is not supported on {_REPR}.'),
        clouds.CloudImplementationFeatures.HIGH_AVAILABILITY_CONTROLLERS:
            (f'High availability controllers are not supported on {_REPR}.'),
        clouds.CloudImplementationFeatures.CLONE_DISK_FROM_CLUSTER:
            (f'Disk cloning is not supported on {_REPR}.'),
        clouds.CloudImplementationFeatures.CUSTOM_MULTI_NETWORK:
            (f'Customized multiple network interfaces are not supported '
             f'on {_REPR}.'),
        clouds.CloudImplementationFeatures.LOCAL_DISK:
            (f'Local disk is not supported on {_REPR}.'),
    }

    PROVISIONER_VERSION = clouds.ProvisionerVersion.SKYPILOT
    STATUS_VERSION = clouds.StatusVersion.SKYPILOT

    @classmethod
    def _unsupported_features_for_resources(
        cls,
        resources: 'resources_lib.Resources',
        region: Optional[str] = None,
    ) -> Dict[clouds.CloudImplementationFeatures, str]:
        del resources, region  # unused
        return cls._CLOUD_UNSUPPORTED_FEATURES

    @classmethod
    def max_cluster_name_length(cls) -> Optional[int]:
        return cls._MAX_CLUSTER_NAME_LEN_LIMIT

    def instance_type_exists(self, instance_type: str) -> bool:
        return catalog.instance_type_exists(instance_type, 'modal')

    @classmethod
    def regions_with_offering(
        cls,
        instance_type: str,
        accelerators: Optional[Dict[str, int]],
        use_spot: bool,
        region: Optional[str],
        zone: Optional[str],
        resources: Optional['resources_lib.Resources'] = None,
    ) -> List[clouds.Region]:
        assert zone is None, 'Modal does not support zones.'
        del accelerators, zone, resources  # unused
        regions = catalog.get_region_zones_for_instance_type(
            instance_type, use_spot, 'modal')
        if region is not None:
            regions = [r for r in regions if r.name == region]
        return regions

    @classmethod
    def get_vcpus_mem_from_instance_type(
            cls, instance_type: str) -> Tuple[Optional[float], Optional[float]]:
        return catalog.get_vcpus_mem_from_instance_type(instance_type,
                                                        clouds='modal')

    def instance_type_to_hourly_cost(
        self,
        instance_type: str,
        use_spot: bool,
        region: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> float:
        return catalog.get_hourly_cost(instance_type,
                                       use_spot=use_spot,
                                       region=region,
                                       zone=zone,
                                       clouds='modal')

    @classmethod
    def get_default_instance_type(
        cls,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        disk_tier: Optional[DiskTier] = None,
        local_disk: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        use_spot: bool = False,
        max_hourly_cost: Optional[float] = None,
    ) -> Optional[str]:
        return catalog.get_default_instance_type(
            cpus=cpus,
            memory=memory,
            disk_tier=disk_tier,
            local_disk=local_disk,
            region=region,
            zone=zone,
            use_spot=use_spot,
            max_hourly_cost=max_hourly_cost,
            clouds='modal',
        )

    @classmethod
    def get_accelerators_from_instance_type(
            cls, instance_type: str) -> Optional[Dict[str, Union[int, float]]]:
        return catalog.get_accelerators_from_instance_type(instance_type,
                                                           clouds='modal')

    @classmethod
    def _check_credentials(cls) -> Tuple[bool, Optional[str]]:
        if modal_utils.has_credentials():
            return True, None
        return False, (
            'Modal credentials not found. Run `modal token new`, or set '
            f'{modal_utils.ENV_TOKEN_ID} and {modal_utils.ENV_TOKEN_SECRET}.')

    @classmethod
    def _check_compute_credentials(cls) -> Tuple[bool, Optional[str]]:
        return cls._check_credentials()

    @classmethod
    def get_credential_file_mounts(cls) -> Dict[str, str]:
        # Nothing on the cluster calls Modal, so no credentials are copied.
        return {}

    def __repr__(self):
        return self._REPR

    def _get_feasible_launchable_resources(
        self, resources: 'resources_lib.Resources'
    ) -> 'resources_utils.FeasibleResources':
        if resources.use_spot:
            return resources_utils.FeasibleResources(
                [], [], 'Modal has no spot sandboxes.')
        if resources.instance_type is not None:
            assert resources.is_launchable(), resources
            resources = resources.copy(accelerators=None)
            return resources_utils.FeasibleResources([resources], [], None)

        def _make(instance_list):
            return [
                resources.copy(
                    cloud=Modal(),
                    instance_type=instance_type,
                    # GPUs are included in the instance type.
                    accelerators=None,
                    cpus=None,
                    memory=None,
                ) for instance_type in instance_list
            ]

        accelerators = resources.accelerators
        if accelerators is None:
            default_instance_type = Modal.get_default_instance_type(
                cpus=resources.cpus,
                memory=resources.memory,
                disk_tier=resources.disk_tier,
                local_disk=resources.local_disk,
                region=resources.region,
                zone=resources.zone,
                max_hourly_cost=resources.max_hourly_cost,
            )
            if default_instance_type is None:
                return resources_utils.FeasibleResources([], [], None)
            return resources_utils.FeasibleResources(
                _make([default_instance_type]), [], None)

        assert len(accelerators) == 1, resources
        acc, acc_count = list(accelerators.items())[0]
        (instance_list,
         fuzzy_candidate_list) = catalog.get_instance_type_for_accelerator(
             acc,
             acc_count,
             use_spot=False,
             cpus=resources.cpus,
             memory=resources.memory,
             local_disk=resources.local_disk,
             region=resources.region,
             zone=resources.zone,
             max_hourly_cost=resources.max_hourly_cost,
             clouds='modal',
         )
        if instance_list is None:
            return resources_utils.FeasibleResources([], fuzzy_candidate_list,
                                                     None)
        return resources_utils.FeasibleResources(_make(instance_list),
                                                 fuzzy_candidate_list, None)

    def validate_region_zone(
            self, region: Optional[str],
            zone: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if zone is not None:
            raise ValueError('Modal does not support zones.')
        return catalog.validate_region_zone(region, zone, 'modal')

    @classmethod
    def regions(cls) -> List[clouds.Region]:
        return catalog.regions('modal')

    @classmethod
    def zones_provision_loop(
        cls,
        *,
        region: str,
        num_nodes: int,
        instance_type: str,
        accelerators: Optional[Dict[str, int]] = None,
        use_spot: bool = False,
    ):
        yield None

    @classmethod
    def get_zone_shell_cmd(cls) -> Optional[str]:
        return None

    def get_egress_cost(self, num_gigabytes: float) -> float:
        return 0.0

    def accelerators_to_hourly_cost(
        self,
        accelerators: Dict[str, int],
        use_spot: bool,
        region: Optional[str],
        zone: Optional[str],
    ) -> float:
        # GPUs are billed as part of the instance type.
        return 0.0

    def make_deploy_resources_variables(
        self,
        resources: 'resources_lib.Resources',
        cluster_name: resources_utils.ClusterName,
        region: 'clouds.Region',
        zones: Optional[List['clouds.Zone']],
        num_nodes: int,
        dryrun: bool = False,
        volume_mounts: Optional[List['volume_lib.VolumeMount']] = None,
    ) -> Dict[str, Any]:
        del cluster_name, dryrun, num_nodes, volume_mounts  # unused
        assert zones is None, ('Modal does not support zones', zones)
        resources = resources.assert_launchable()
        acc_dict = self.get_accelerators_from_instance_type(
            resources.instance_type)
        custom_resources = resources_utils.make_ray_custom_resources_str(
            acc_dict)
        cpus, memory = self.get_vcpus_mem_from_instance_type(
            resources.instance_type)

        image_id = ''
        if resources.image_id is not None:
            docker_image = resources.extract_docker_image()
            if docker_image is not None:
                image_id = docker_image
            elif resources.region in resources.image_id:
                image_id = resources.image_id[resources.region]
            else:
                image_id = resources.image_id.get(None, '')

        def _config(key: str, default: Any) -> Any:
            return skypilot_config.get_effective_region_config(
                cloud='modal',
                region=region.name,
                keys=(key,),
                default_value=default,
                override_configs=resources.cluster_config_overrides,
            )

        return {
            'instance_type': resources.instance_type,
            'custom_resources': custom_resources,
            'region': region.name,
            'cpus': cpus,
            'memory_gib': memory,
            'image_id': image_id or '',
            # Hard lifetime: Modal ends the sandbox this many minutes after
            # it starts (at most 24 hours), whatever SkyPilot's state.
            'ttl_minutes': int(_config('ttl_minutes', 0) or 0),
            'modal_environment': _config('environment', '') or '',
        }
