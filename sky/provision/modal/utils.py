"""Modal sandbox helpers for the provisioner."""

import os
from typing import Any, Dict, List, Optional

from sky import sky_logging
from sky.adaptors import modal as modal_adaptor
from sky.utils import status_lib

logger = sky_logging.init_logger(__name__)

# All SkyPilot sandboxes live in one Modal app, in the configured Modal
# environment (`modal.environment`), and are tagged with their cluster.
APP_NAME = 'skypilot'
CLUSTER_TAG = 'skypilot-cluster'

ENV_TOKEN_ID = 'MODAL_TOKEN_ID'
ENV_TOKEN_SECRET = 'MODAL_TOKEN_SECRET'
CREDENTIALS_PATH = '~/.modal.toml'

# Modal ends a sandbox at its timeout; 24 hours is the maximum.
MAX_TIMEOUT_SECONDS = 24 * 60 * 60

DEFAULT_IMAGE = 'nvidia/cuda:12.4.1-base-ubuntu22.04'
# Baked into the sandbox image (Modal caches the build) so that SkyPilot's
# runtime setup and rsync work on stock Debian/Ubuntu images. Images without
# apt are left as they are.
IMAGE_SETUP_COMMANDS = [
    '(command -v apt-get >/dev/null && apt-get update -y && '
    'DEBIAN_FRONTEND=noninteractive apt-get install -y '
    '--no-install-recommends rsync sudo curl ca-certificates procps lsof '
    'patch && rm -rf /var/lib/apt/lists/*) || true',
]
# The sandbox only has to stay alive; SkyPilot runs everything through exec.
SANDBOX_COMMAND = ('sleep', 'infinity')

# SkyPilot accelerator name -> Modal GPU type.
ACC_TO_MODAL_GPU = {
    'B300': 'B300',
    'B200': 'B200',
    'H200': 'H200',
    'H100': 'H100',
    'RTXPRO6000': 'RTX-PRO-6000',
    'A100-80GB': 'A100-80GB',
    'A100': 'A100-40GB',
    'L40S': 'L40S',
    'A10': 'A10',
    'L4': 'L4',
    'T4': 'T4',
}


class ModalError(Exception):
    """Raised for Modal provisioning errors."""


def has_credentials() -> bool:
    if os.environ.get(ENV_TOKEN_ID) and os.environ.get(ENV_TOKEN_SECRET):
        return True
    return os.path.exists(os.path.expanduser(CREDENTIALS_PATH))


def modal_gpu(instance_type: str) -> Optional[str]:
    """Returns the Modal `gpu` argument for an instance type, e.g. `H100:2`.

    GPU instance types are `<count>x-<ACC>`; CPU ones `cpu-<vcpus>x-<mem>gb`.
    """
    if instance_type.startswith('cpu-'):
        return None
    count, acc = instance_type.split('x-', maxsplit=1)
    gpu = ACC_TO_MODAL_GPU.get(acc)
    if gpu is None:
        raise ModalError(f'Unknown Modal instance type: {instance_type}')
    return gpu if int(count) == 1 else f'{gpu}:{int(count)}'


def timeout_seconds(ttl_minutes: Optional[int]) -> int:
    """The sandbox timeout: the TTL if set, capped at Modal's maximum."""
    if not ttl_minutes:
        return MAX_TIMEOUT_SECONDS
    return min(int(ttl_minutes) * 60, MAX_TIMEOUT_SECONDS)


def _lookup_app(environment: Optional[str], create: bool) -> Any:
    modal = modal_adaptor.modal
    try:
        return modal.App.lookup(APP_NAME,
                                environment_name=environment or None,
                                create_if_missing=create)
    except modal.exception.NotFoundError:
        if create:
            raise
        return None


def list_cluster_sandboxes(cluster_name_on_cloud: str,
                           environment: Optional[str] = None) -> List[Any]:
    """Returns the cluster's sandboxes that are still running."""
    app = _lookup_app(environment, create=False)
    if app is None:
        return []
    sandboxes = []
    for sandbox in modal_adaptor.modal.Sandbox.list(
            app_id=app.app_id, tags={CLUSTER_TAG: cluster_name_on_cloud}):
        if sandbox.poll() is None:
            sandboxes.append(sandbox)
    return sandboxes


def create_sandbox(
    cluster_name_on_cloud: str,
    instance_type: str,
    cpus: float,
    memory_gib: float,
    image_id: Optional[str],
    ttl_minutes: Optional[int],
    environment: Optional[str] = None,
) -> str:
    """Creates the cluster's sandbox and returns its ID."""
    modal = modal_adaptor.modal
    app = _lookup_app(environment, create=True)
    image = modal.Image.from_registry(
        image_id or DEFAULT_IMAGE).run_commands(*IMAGE_SETUP_COMMANDS)
    kwargs: Dict[str, Any] = {
        'app': app,
        'image': image,
        'tags': {
            CLUSTER_TAG: cluster_name_on_cloud
        },
        'timeout': timeout_seconds(ttl_minutes),
        # One Modal CPU core is two vCPUs.
        'cpu': cpus / 2,
        'memory': int(memory_gib * 1024),
    }
    gpu = modal_gpu(instance_type)
    if gpu is not None:
        kwargs['gpu'] = gpu
    logger.debug(f'Creating Modal sandbox for {cluster_name_on_cloud}: '
                 f'{instance_type}, timeout {kwargs["timeout"]}s')
    sandbox = modal.Sandbox.create(*SANDBOX_COMMAND, **kwargs)
    return sandbox.object_id


def terminate_sandbox(sandbox: Any) -> None:
    sandbox.terminate()


def to_cluster_status(sandbox: Any) -> Optional['status_lib.ClusterStatus']:
    """A running sandbox is UP; a finished one no longer exists."""
    return status_lib.ClusterStatus.UP if sandbox.poll() is None else None
