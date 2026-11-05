"""Modal sandbox provisioning."""

from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.provision import common
from sky.provision.modal import utils
from sky.utils import command_runner
from sky.utils import status_lib

PROVIDER_NAME = 'modal'

logger = sky_logging.init_logger(__name__)


def _cluster_sandboxes(cluster_name_on_cloud: str) -> Dict[str, Any]:
    return {
        sandbox.object_id: sandbox
        for sandbox in utils.list_cluster_sandboxes(cluster_name_on_cloud)
    }


def run_instances(
    region: str,
    cluster_name: str,
    cluster_name_on_cloud: str,
    config: common.ProvisionConfig,
) -> common.ProvisionRecord:
    """Provisions the cluster's sandbox.

    Modal clusters are single-node: one sandbox per cluster. If a sandbox
    tagged with the cluster name is still running, it is reused.
    """
    del cluster_name  # unused
    if config.count > 1:
        raise utils.ModalError('Modal does not support multi-node clusters '
                               f'(requested {config.count} nodes).')

    created: List[str] = []
    existing = _cluster_sandboxes(cluster_name_on_cloud)
    if existing:
        sandbox_id, sandbox = next(iter(existing.items()))
        logger.debug(f'Reusing existing sandbox {sandbox_id} for cluster '
                     f'{cluster_name_on_cloud}')
    else:
        node_config = config.node_config
        sandbox = utils.create_sandbox(
            cluster_name_on_cloud=cluster_name_on_cloud,
            cpus=node_config.get('Cpus'),
            memory_gib=node_config.get('MemoryGiB'),
            gpu=node_config.get('Gpu') or None,
            image_id=node_config.get('ImageId') or None,
            timeout_seconds=node_config.get('TimeoutSeconds',
                                            utils.MAX_TIMEOUT_SECONDS),
        )
        sandbox_id = sandbox.object_id
        created.append(sandbox_id)
        logger.debug(f'Created sandbox {sandbox_id} for cluster '
                     f'{cluster_name_on_cloud}')
    utils.wait_for_sandbox_running(sandbox)

    return common.ProvisionRecord(
        provider_name=PROVIDER_NAME,
        cluster_name=cluster_name_on_cloud,
        region=region,
        zone=None,
        head_instance_id=sandbox_id,
        resumed_instance_ids=[],
        created_instance_ids=created,
    )


def wait_instances(region: str, cluster_name_on_cloud: str,
                   state: Optional[status_lib.ClusterStatus]) -> None:
    """Waiting is handled in run_instances(); no-op."""
    del region, cluster_name_on_cloud, state  # unused


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[dict] = None,
    worker_only: bool = False,
) -> None:
    """Terminates the cluster's sandbox."""
    del provider_config  # unused
    if worker_only:
        # Single node: there are no workers.
        return
    for sandbox_id, sandbox in _cluster_sandboxes(
            cluster_name_on_cloud).items():
        logger.debug(f'Terminating sandbox {sandbox_id} for cluster '
                     f'{cluster_name_on_cloud}')
        utils.terminate_sandbox(sandbox)


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """Modal sandboxes cannot be stopped and resumed, only terminated."""
    raise NotImplementedError('stop_instances is not supported for Modal.')


def get_cluster_info(
    region: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
) -> common.ClusterInfo:
    """Returns the cluster's sandbox.

    The instance ID is the sandbox's Modal object ID (`sb-...`), which
    `ModalCommandRunner` passes to `modal.Sandbox.from_id`. There is no
    SSH endpoint; the IP fields are placeholders for code that expects
    them.
    """
    del region  # unused
    instances: Dict[str, List[common.InstanceInfo]] = {}
    head_instance_id = None
    for sandbox_id in _cluster_sandboxes(cluster_name_on_cloud):
        if head_instance_id is None:
            head_instance_id = sandbox_id
        instances[sandbox_id] = [
            common.InstanceInfo(
                instance_id=sandbox_id,
                internal_ip='127.0.0.1',
                external_ip=None,
                ssh_port=22,
                tags=utils.cluster_tags(cluster_name_on_cloud),
            )
        ]

    return common.ClusterInfo(
        instances=instances,
        head_instance_id=head_instance_id,
        provider_name=PROVIDER_NAME,
        provider_config=provider_config,
        ssh_user='root',
    )


def get_command_runners(
    cluster_info: common.ClusterInfo,
    **credentials: Dict[str, Any],
) -> List[command_runner.CommandRunner]:
    """Returns a ModalCommandRunner per sandbox, head first.

    Each runner's node is a 1-tuple of the sandbox's Modal object ID.
    """
    sandbox_ids = list(cluster_info.instances.keys())
    head = cluster_info.head_instance_id
    if head is not None and head in sandbox_ids:
        sandbox_ids.remove(head)
        sandbox_ids.insert(0, head)
    # ModalCommandRunner is added to command_runner by the Modal runner
    # change; drop the ignore once both are on the same branch.
    runner_cls = command_runner.ModalCommandRunner  # type: ignore[attr-defined]
    return runner_cls.make_runner_list([(sid,) for sid in sandbox_ids],
                                       **credentials)


def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[dict] = None,
    non_terminated_only: bool = True,
    retry_if_missing: bool = False,
) -> Dict[str, Tuple[Optional['status_lib.ClusterStatus'], Optional[str]]]:
    """Returns the statuses of the cluster's sandboxes.

    Modal only lists live sandboxes, so every listed sandbox is UP and a
    terminated cluster has no entries.
    """
    del cluster_name, provider_config, non_terminated_only  # unused
    del retry_if_missing  # unused
    return {
        sandbox_id: (status_lib.ClusterStatus.UP, None)
        for sandbox_id in _cluster_sandboxes(cluster_name_on_cloud)
    }


def open_ports(
    cluster_name_on_cloud: str,
    ports: list,
    provider_config: Optional[dict] = None,
) -> None:
    """Opening ports is not supported for Modal."""
    raise NotImplementedError('open_ports is not supported for Modal.')


def cleanup_ports(
    cluster_name_on_cloud: str,
    provider_config: Optional[dict] = None,
    ports: Optional[list] = None,
) -> None:
    """Cleanup ports. Not supported for Modal."""
    raise NotImplementedError('cleanup_ports is not supported for Modal.')


def cleanup_custom_multi_network(
    cluster_name_on_cloud: str,
    provider_config: Dict[str, Any],
    failover: bool = False,
) -> None:
    """Cleanup custom multi-network. Not supported for Modal."""
    raise NotImplementedError(
        'cleanup_custom_multi_network is not supported for Modal.')
