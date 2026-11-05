"""Modal sandbox provisioning.

Each cluster is one Modal sandbox that SkyPilot creates and owns: it runs
`sleep infinity` and SkyPilot runs everything else in it through
`Sandbox.exec` (see ModalCommandRunner), so no SSH server or open port is
needed. Tearing down terminates the sandbox, which stops billing.
"""

from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.provision import common
from sky.provision.modal import utils
from sky.utils import command_runner
from sky.utils import status_lib

PROVIDER_NAME = 'modal'
# Commands run through exec, not over the network; the address is only a
# placeholder for code that expects one.
_PLACEHOLDER_IP = '127.0.0.1'
# No SSH server: keeps the cluster out of the API server's SSH config.
_RUNTIME_METADATA = common.ProvisionRuntimeMetadata(ssh_available=False)

logger = sky_logging.init_logger(__name__)


def _environment(provider_config: Optional[Dict[str, Any]]) -> Optional[str]:
    return (provider_config or {}).get('environment') or None


def run_instances(
    region: str,
    cluster_name: str,
    cluster_name_on_cloud: str,
    config: common.ProvisionConfig,
) -> common.ProvisionRecord:
    """Creates the cluster's sandbox, or reuses the running one."""
    del cluster_name  # unused
    if config.count > 1:
        raise utils.ModalError('Modal does not support multi-node clusters '
                               f'(requested {config.count} nodes).')
    environment = _environment(config.provider_config)
    existing = utils.list_cluster_sandboxes(cluster_name_on_cloud, environment)
    if existing:
        sandbox_id = existing[0].object_id
        logger.debug(f'Reusing sandbox {sandbox_id} for cluster '
                     f'{cluster_name_on_cloud}')
        return common.ProvisionRecord(
            provider_name=PROVIDER_NAME,
            cluster_name=cluster_name_on_cloud,
            region=region,
            zone=None,
            head_instance_id=sandbox_id,
            resumed_instance_ids=[],
            created_instance_ids=[],
            runtime_metadata=_RUNTIME_METADATA,
        )

    node_config = config.node_config
    instance_type = node_config.get('InstanceType')
    if not instance_type:
        raise utils.ModalError('InstanceType is not set in node_config.')
    sandbox_id = utils.create_sandbox(
        cluster_name_on_cloud=cluster_name_on_cloud,
        instance_type=instance_type,
        cpus=float(node_config['Cpus']),
        memory_gib=float(node_config['MemoryGiB']),
        image_id=node_config.get('ImageId') or None,
        ttl_minutes=node_config.get('TtlMinutes') or None,
        environment=environment,
    )
    logger.debug(f'Created sandbox {sandbox_id} for cluster '
                 f'{cluster_name_on_cloud}')
    return common.ProvisionRecord(
        provider_name=PROVIDER_NAME,
        cluster_name=cluster_name_on_cloud,
        region=region,
        zone=None,
        head_instance_id=sandbox_id,
        resumed_instance_ids=[],
        created_instance_ids=[sandbox_id],
        runtime_metadata=_RUNTIME_METADATA,
    )


def wait_instances(region: str, cluster_name_on_cloud: str,
                   state: Optional[status_lib.ClusterStatus]) -> None:
    """Sandbox.create returns once the sandbox is scheduled; exec waits for
    it to start, so there is nothing to wait for here."""
    del region, cluster_name_on_cloud, state  # unused


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[dict] = None,
    worker_only: bool = False,
) -> None:
    """Terminates the cluster's sandboxes."""
    del worker_only  # single node
    for sandbox in utils.list_cluster_sandboxes(cluster_name_on_cloud,
                                                _environment(provider_config)):
        logger.debug(f'Terminating sandbox {sandbox.object_id} for cluster '
                     f'{cluster_name_on_cloud}')
        utils.terminate_sandbox(sandbox)


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """Modal sandboxes cannot be stopped and resumed."""
    raise NotImplementedError('stop_instances is not supported for Modal.')


def get_cluster_info(
    region: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
) -> common.ClusterInfo:
    del region  # unused
    instances: Dict[str, List[common.InstanceInfo]] = {}
    head_instance_id = None
    for sandbox in utils.list_cluster_sandboxes(cluster_name_on_cloud,
                                                _environment(provider_config)):
        sandbox_id = sandbox.object_id
        if head_instance_id is None:
            head_instance_id = sandbox_id
        instances[sandbox_id] = [
            common.InstanceInfo(
                instance_id=sandbox_id,
                internal_ip=_PLACEHOLDER_IP,
                external_ip=_PLACEHOLDER_IP,
                ssh_port=22,
                tags={utils.CLUSTER_TAG: cluster_name_on_cloud},
            )
        ]
    return common.ClusterInfo(
        instances=instances,
        head_instance_id=head_instance_id,
        provider_name=PROVIDER_NAME,
        provider_config=provider_config,
        ssh_user='root',
    )


def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[dict] = None,
    non_terminated_only: bool = True,
    retry_if_missing: bool = False,
) -> Dict[str, Tuple[Optional['status_lib.ClusterStatus'], Optional[str]]]:
    """Returns the statuses of the cluster's sandboxes.

    Finished sandboxes (terminated, or ended by their timeout) are not
    listed, so the cluster reads as terminated.
    """
    del cluster_name, non_terminated_only, retry_if_missing  # unused
    return {
        sandbox.object_id: (utils.to_cluster_status(sandbox), None)
        for sandbox in utils.list_cluster_sandboxes(
            cluster_name_on_cloud, _environment(provider_config))
    }


def get_command_runners(
    cluster_info: common.ClusterInfo,
    **credentials: Dict[str, Any],
) -> List[command_runner.CommandRunner]:
    """Runners that exec into the sandboxes instead of using SSH."""
    del credentials  # unused
    ids = list(cluster_info.instances)
    if cluster_info.head_instance_id in ids:
        ids.remove(cluster_info.head_instance_id)
        ids.insert(0, cluster_info.head_instance_id)
    return [command_runner.ModalCommandRunner(('modal', i)) for i in ids]


def open_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    provider_config: Optional[dict] = None,
) -> None:
    raise NotImplementedError('open_ports is not supported for Modal.')


def cleanup_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    provider_config: Optional[dict] = None,
) -> None:
    del cluster_name_on_cloud, ports, provider_config  # nothing opened


def cleanup_custom_multi_network(
    cluster_name_on_cloud: str,
    provider_config: Dict[str, Any],
    failover: bool = False,
) -> None:
    raise NotImplementedError(
        'cleanup_custom_multi_network is not supported for Modal.')
