"""Modal sandbox helpers for the provisioner."""

import json
import os
from typing import Any, Dict, List, Optional

from sky import sky_logging
from sky.adaptors import modal as modal_adaptor
from sky.utils import status_lib

logger = sky_logging.init_logger(__name__)

# All SkyPilot sandboxes live in one Modal app, in the configured Modal
# environment (`modal.environment`), and are tagged with their cluster.
# Sandbox names are unique within an app, so each sandbox is named after its
# cluster: a retried or concurrent launch finds it instead of making another.
APP_NAME = 'skypilot'
CLUSTER_TAG = 'skypilot-cluster'
MANAGED_TAG = 'skypilot-managed'

ENV_TOKEN_ID = 'MODAL_TOKEN_ID'
ENV_TOKEN_SECRET = 'MODAL_TOKEN_SECRET'
CREDENTIALS_PATH = '~/.modal.toml'
_GENERATED_CREDENTIALS_PATH = '~/.sky/generated/modal/modal.toml'

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

# How long a new sandbox may take to run its first command. GPU sandboxes
# can wait for capacity.
START_TIMEOUT_SECONDS = 20 * 60

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
    'A10G': 'A10',
    'L4': 'L4',
    'T4': 'T4',
}


class ModalError(Exception):
    """Raised for Modal provisioning errors."""


def has_credentials() -> bool:
    if os.environ.get(ENV_TOKEN_ID) and os.environ.get(ENV_TOKEN_SECRET):
        return True
    return os.path.exists(os.path.expanduser(CREDENTIALS_PATH))


def get_credentials_path() -> str:
    return CREDENTIALS_PATH


def write_credentials_file() -> Optional[str]:
    """Writes the active Modal token to a private file for mounting.

    Autodown runs on the sandbox and managed jobs launch from their
    controller, so both need a token file, even when the token only exists
    in this process's environment. Only the token in use is written, not
    every profile in ~/.modal.toml. Returns the file's path, or None when
    there is no token.
    """
    config = modal_adaptor.modal.config.config
    token_id, token_secret = config.get('token_id'), config.get('token_secret')
    if not token_id or not token_secret:
        return None
    path = os.path.expanduser(_GENERATED_CREDENTIALS_PATH)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp_path = f'{path}.{os.getpid()}.tmp'
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        # JSON strings are valid TOML basic strings.
        f.write(f'[skypilot]\ntoken_id = {json.dumps(token_id)}\n'
                f'token_secret = {json.dumps(token_secret)}\n'
                'active = true\n')
    os.replace(tmp_path, path)
    return path


def _load_mounted_token() -> None:
    """Exports the mounted token on a Modal sandbox.

    Inside Modal containers (`MODAL_IS_REMOTE=1`) the SDK ignores
    ~/.modal.toml, so autodown on the sandbox reads the file written by
    write_credentials_file() into the environment, which the SDK always
    reads.
    """
    if os.environ.get('MODAL_IS_REMOTE') != '1' or (
            os.environ.get(ENV_TOKEN_ID) and os.environ.get(ENV_TOKEN_SECRET)):
        return
    path = os.path.expanduser(CREDENTIALS_PATH)
    if not os.path.exists(path):
        return
    token = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            key, sep, value = line.partition('=')
            if sep and key.strip() in ('token_id', 'token_secret'):
                token[key.strip()] = json.loads(value.strip())
    if len(token) == 2:
        os.environ[ENV_TOKEN_ID] = token['token_id']
        os.environ[ENV_TOKEN_SECRET] = token['token_secret']


def cluster_tags(cluster_name_on_cloud: str) -> Dict[str, str]:
    return {CLUSTER_TAG: cluster_name_on_cloud, MANAGED_TAG: 'true'}


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
    _load_mounted_token()
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
) -> Any:
    """Creates the cluster's sandbox and returns it.

    If a sandbox with the cluster's name exists, that one is returned.
    """
    modal = modal_adaptor.modal
    app = _lookup_app(environment, create=True)
    image = modal.Image.from_registry(
        image_id or DEFAULT_IMAGE).run_commands(*IMAGE_SETUP_COMMANDS)
    kwargs: Dict[str, Any] = {
        'app': app,
        'name': cluster_name_on_cloud,
        'image': image,
        'tags': cluster_tags(cluster_name_on_cloud),
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
    try:
        return modal.Sandbox.create(*SANDBOX_COMMAND, **kwargs)
    except modal.exception.AlreadyExistsError:
        # An interrupted or concurrent launch already created it.
        logger.debug(f'Sandbox {cluster_name_on_cloud} already exists; '
                     'reusing it.')
        return modal.Sandbox.from_name(APP_NAME,
                                       cluster_name_on_cloud,
                                       environment_name=environment or None)


def wait_for_sandbox_running(sandbox: Any,
                             timeout: int = START_TIMEOUT_SECONDS) -> None:
    """Blocks until the sandbox runs commands.

    Raises:
        ModalError: the sandbox exited, or the probe failed.
    """
    try:
        process = sandbox.exec('true', timeout=timeout)
        process.wait()
        returncode = process.returncode
    except Exception as e:  # pylint: disable=broad-except
        raise ModalError(f'Sandbox {sandbox.object_id} did not start (exit '
                         f'code {sandbox.poll()}): {e}') from e
    if returncode != 0:
        raise ModalError(f'Sandbox {sandbox.object_id} start probe exited '
                         f'with code {returncode}.')


def terminate_sandbox(sandbox: Any) -> None:
    try:
        sandbox.terminate()
    except modal_adaptor.modal.exception.NotFoundError:
        logger.debug(f'Sandbox {sandbox.object_id} is already gone.')


def to_cluster_status(sandbox: Any) -> Optional['status_lib.ClusterStatus']:
    """A running sandbox is UP; a finished one no longer exists."""
    return status_lib.ClusterStatus.UP if sandbox.poll() is None else None
