"""Modal API helpers for the SkyPilot provisioner.

Modal (https://modal.com) runs sandboxes: containers with per-second
billing that are created and driven through the Modal Python SDK. A
SkyPilot cluster on Modal is one sandbox running `sleep infinity` under
the SkyPilot-owned Modal app `skypilot`, named and tagged with the
cluster name. Commands reach the sandbox through `ModalCommandRunner`
(sky/utils/command_runner.py), which attaches to it with
`modal.Sandbox.from_id(<sandbox object_id>)`; there is no SSH.

Authentication uses the Modal SDK's own credentials: `~/.modal.toml`
(written by `modal setup` / `modal token new`) or the `MODAL_TOKEN_ID`
and `MODAL_TOKEN_SECRET` environment variables.
"""

import inspect
import os
from typing import Any, Dict, List, Optional

from sky import sky_logging
from sky.adaptors import modal as modal_adaptor

logger = sky_logging.init_logger(__name__)

CREDENTIALS_PATH = '~/.modal.toml'
ENV_TOKEN_ID = 'MODAL_TOKEN_ID'
ENV_TOKEN_SECRET = 'MODAL_TOKEN_SECRET'

# All SkyPilot sandboxes live under this Modal app. Sandbox names are
# unique within an app, so the cluster name doubles as an idempotency key.
APP_NAME = 'skypilot'

# Tags used to associate sandboxes with a SkyPilot cluster.
CLUSTER_TAG = 'skypilot-cluster'
MANAGED_TAG = 'skypilot-managed'

# Modal caps a sandbox's lifetime at 24 hours. The lifetime is absolute
# (not idle-based), so it is a backstop against leaked sandboxes; idle
# autodown is done by SkyPilot's skylet on the sandbox.
MAX_TIMEOUT_SECONDS = 24 * 60 * 60

# The process that keeps the sandbox alive; commands run via exec.
SANDBOX_COMMAND = ('sleep', 'infinity')
SANDBOX_WORKDIR = '/root'

# Default image when the user does not pass `image_id: docker:<image>`.
# SkyPilot's runtime setup needs bash, curl and rsync on the node.
DEFAULT_PYTHON_VERSION = '3.11'
DEFAULT_APT_PACKAGES = (
    'ca-certificates',
    'curl',
    'git',
    'lsof',
    'procps',
    'rsync',
    'sudo',
    'wget',
)

# How long to wait for a new sandbox to start running commands. GPU
# sandboxes may wait for capacity.
LAUNCH_TIMEOUT_SECONDS = 20 * 60


class ModalError(Exception):
    """Raised for Modal provisioning errors."""


def get_credentials_path() -> str:
    return CREDENTIALS_PATH


def has_credentials() -> bool:
    if os.environ.get(ENV_TOKEN_ID) and os.environ.get(ENV_TOKEN_SECRET):
        return True
    return os.path.exists(os.path.expanduser(CREDENTIALS_PATH))


def get_app() -> Any:
    return modal_adaptor.modal.App.lookup(APP_NAME, create_if_missing=True)


def cluster_tags(cluster_name_on_cloud: str) -> Dict[str, str]:
    return {CLUSTER_TAG: cluster_name_on_cloud, MANAGED_TAG: 'true'}


def list_cluster_sandboxes(cluster_name_on_cloud: str) -> List[Any]:
    """Returns the cluster's running sandboxes.

    Modal only lists sandboxes that have not finished, so every returned
    sandbox is live (scheduled or running).
    """
    app = get_app()
    return list(
        modal_adaptor.modal.Sandbox.list(
            app_id=app.app_id, tags={CLUSTER_TAG: cluster_name_on_cloud}))


def build_image(image_id: Optional[str]) -> Any:
    modal = modal_adaptor.modal
    if image_id:
        return modal.Image.from_registry(image_id)
    return modal.Image.debian_slim(
        python_version=DEFAULT_PYTHON_VERSION).apt_install(
            *DEFAULT_APT_PACKAGES)


def create_sandbox(
    cluster_name_on_cloud: str,
    cpus: Optional[float],
    memory_gib: Optional[float],
    gpu: Optional[str],
    image_id: Optional[str],
    timeout_seconds: int,
) -> Any:
    """Creates the cluster's sandbox and returns it.

    If a sandbox with the cluster's name already exists (e.g. a concurrent
    or interrupted launch), that sandbox is returned instead.
    """
    modal = modal_adaptor.modal
    app = get_app()
    tags = cluster_tags(cluster_name_on_cloud)
    timeout_seconds = max(1, min(int(timeout_seconds), MAX_TIMEOUT_SECONDS))
    kwargs: Dict[str, Any] = {
        'app': app,
        'name': cluster_name_on_cloud,
        'image': build_image(image_id),
        'timeout': timeout_seconds,
        'workdir': SANDBOX_WORKDIR,
    }
    if gpu:
        kwargs['gpu'] = gpu
    if cpus:
        kwargs['cpu'] = float(cpus)
    if memory_gib:
        kwargs['memory'] = int(float(memory_gib) * 1024)
    # Modal SDKs before 1.3 (the last ones supporting Python 3.9) do not
    # take `tags` at creation; tag right after instead. The sandbox name
    # still makes creation idempotent in between.
    tags_at_create = 'tags' in inspect.signature(
        modal.Sandbox.create).parameters
    if tags_at_create:
        kwargs['tags'] = tags
    try:
        sandbox = modal.Sandbox.create(*SANDBOX_COMMAND, **kwargs)
    except modal.exception.AlreadyExistsError:
        # A concurrent or interrupted launch already created the sandbox.
        logger.debug(f'Sandbox for cluster {cluster_name_on_cloud} already '
                     'exists; reusing it.')
        sandbox = modal.Sandbox.from_name(APP_NAME, cluster_name_on_cloud)
        tags_at_create = False
    if not tags_at_create:
        sandbox.set_tags(tags)
    return sandbox


def wait_for_sandbox_running(sandbox: Any,
                             timeout: int = LAUNCH_TIMEOUT_SECONDS) -> None:
    """Blocks until the sandbox runs commands.

    Raises ModalError if the sandbox exited or the probe command failed.
    """
    try:
        process = sandbox.exec('true', timeout=timeout)
        process.wait()
        returncode = process.returncode
    except Exception as e:  # pylint: disable=broad-except
        exit_code = sandbox.poll()
        raise ModalError(f'Sandbox {sandbox.object_id} did not start '
                         f'(exit code {exit_code}): {e}') from e
    if returncode != 0:
        raise ModalError(f'Sandbox {sandbox.object_id} start probe exited '
                         f'with code {returncode}.')


def terminate_sandbox(sandbox: Any) -> None:
    modal = modal_adaptor.modal
    try:
        sandbox.terminate()
    except modal.exception.NotFoundError:
        logger.debug(f'Sandbox {sandbox.object_id} already gone.')
