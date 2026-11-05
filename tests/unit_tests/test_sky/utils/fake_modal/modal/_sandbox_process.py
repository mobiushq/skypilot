"""Fake sandboxes for the end-to-end launch test: a process tree in namespaces.

A sandbox is `sleep infinity` as PID 1 of new PID, mount, UTS and IPC
namespaces, chrooted into an overlay on an Ubuntu root filesystem. Exec enters
the namespaces with nsenter under a clean environment, so processes outlive the
exec that started them, and runtime setup (`pkill -9 dpkg`, `rm -rf
/etc/security/limits.d`) acts on the sandbox and not on the host. The network
is the host's. Needs root.

    FAKE_MODAL_ROOTFS: the root filesystem, standing in for every image.
    FAKE_MODAL_STATE: where sandboxes and built images are kept.
"""
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional
import uuid

# What a process in a container started from the image gets.
_ENV = {
    'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
    'HOME': '/root',
    'LANG': 'C.UTF-8',
}

# Mounts what a container has on the root filesystem at `root`.
_MOUNT_CONTAINER = r'''
mount -t proc proc root/proc
mount -t sysfs -o ro sys root/sys
mount -t tmpfs -o mode=755 dev root/dev
for device in null:1:3 zero:1:5 full:1:7 random:1:8 urandom:1:9 tty:5:0; do
  IFS=: read -r name major minor <<DEVICE
$device
DEVICE
  mknod -m 666 "root/dev/$name" c "$major" "$minor"
done
mkdir root/dev/shm root/dev/pts
mount -t tmpfs tmpfs root/dev/shm
mount -t devpts -o newinstance,ptmxmode=0666 devpts root/dev/pts
ln -s pts/ptmx root/dev/ptmx
ln -s /proc/self/fd root/dev/fd
ln -s /proc/self/fd/0 root/dev/stdin
ln -s /proc/self/fd/1 root/dev/stdout
ln -s /proc/self/fd/2 root/dev/stderr
mount -t tmpfs tmpfs root/tmp
cat /etc/resolv.conf > root/etc/resolv.conf
'''

_MOUNT_SANDBOX = r'''
set -e
cd "$1"
mount -t overlay overlay -o "lowerdir=$2,upperdir=upper,workdir=work" root
''' + _MOUNT_CONTAINER + r'''
shift 2
exec chroot root /usr/bin/env -i "$@"
'''

_BUILD_IMAGE = r'''
set -e
cd "$1"
shift
''' + _MOUNT_CONTAINER + r'''
exec chroot root /usr/bin/env -i "$@"
'''


def _state_dir() -> str:
    return os.environ['FAKE_MODAL_STATE']


def _sandbox_dir(sandbox_id: str) -> str:
    return os.path.join(_state_dir(), 'sandboxes', sandbox_id)


def _env_args() -> List[str]:
    return [f'{key}={value}' for key, value in _ENV.items()]


def is_process_sandbox(sandbox_id: str) -> bool:
    return 'FAKE_MODAL_STATE' in os.environ and os.path.exists(
        os.path.join(_sandbox_dir(sandbox_id), 'sandbox.json'))


def build_image(commands: List[str]) -> str:
    """Runs an image's commands on the root filesystem, cached like Modal."""
    rootfs = os.environ['FAKE_MODAL_ROOTFS']
    digest = hashlib.sha256('\n'.join(commands).encode()).hexdigest()[:16]
    image = os.path.join(_state_dir(), 'images', digest)
    if os.path.exists(image):
        return image
    os.makedirs(os.path.dirname(image), exist_ok=True)
    building = tempfile.mkdtemp(dir=os.path.dirname(image))
    root = os.path.join(building, 'root')
    os.mkdir(root)
    subprocess.run(['cp', '-a', f'{rootfs}/.', root], check=True)
    for command in commands:
        subprocess.run([
            'unshare', '--pid', '--fork', '--mount', '--uts', '--ipc', 'sh',
            '-c', _BUILD_IMAGE, 'build', building, *_env_args(), 'sh', '-c',
            command
        ],
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    os.rename(root, image)
    shutil.rmtree(building)
    return image


def create(command: List[str], image_dir: str, tags: Dict[str, str],
           app_id: str, timeout: Optional[int], kwargs: Dict[str, Any]) -> str:
    sandbox_id = f'sb-{uuid.uuid4().hex[:16]}'
    directory = _sandbox_dir(sandbox_id)
    for name in ('upper', 'work', 'root'):
        os.makedirs(os.path.join(directory, name))
    if timeout is not None:
        command = ['timeout', '-s', 'KILL', str(timeout)] + command
    # Started through a shell that exits at once: a sandbox is not a descendant
    # of the client, which kills its own process tree on the way out.
    log = shlex.quote(os.path.join(directory, 'start.log'))
    unshare = shlex.join([
        'unshare', '--pid', '--fork', '--mount', '--uts', '--ipc', 'sh', '-c',
        _MOUNT_SANDBOX, 'sandbox', directory, image_dir, *_env_args(), *command
    ])
    unshare_pid = int(
        subprocess.run(
            f'{unshare} </dev/null >/dev/null 2>{log} & echo $!',
            shell=True,
            stdout=subprocess.PIPE,
            check=True,
            start_new_session=True,
        ).stdout)
    pid = _wait_for_init(unshare_pid, directory)
    with open(os.path.join(directory, 'sandbox.json'), 'w',
              encoding='utf-8') as f:
        json.dump({
            'pid': pid,
            'tags': tags,
            'app_id': app_id,
            'kwargs': kwargs
        }, f)
    return sandbox_id


def _children(parent: int) -> List[int]:
    children = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            with open(f'/proc/{entry}/stat', encoding='utf-8') as f:
                ppid = int(f.read().rpartition(')')[2].split()[1])
        except OSError:
            continue
        if ppid == parent:
            children.append(int(entry))
    return children


def _wait_for_init(unshare_pid: int, directory: str) -> int:
    """The namespaces' PID 1, once it has entered the sandbox's root."""
    root = os.path.join(directory, 'root')
    deadline = time.time() + 30
    while time.time() < deadline:
        if not _alive(unshare_pid):
            break
        for pid in _children(unshare_pid):
            try:
                if os.readlink(f'/proc/{pid}/root') == root:
                    return pid
            except OSError:
                pass
        time.sleep(0.02)
    with open(os.path.join(directory, 'start.log'), encoding='utf-8') as f:
        raise RuntimeError(f'Fake sandbox did not start: {f.read()}')


def _record(sandbox_id: str) -> Dict[str, Any]:
    with open(os.path.join(_sandbox_dir(sandbox_id), 'sandbox.json'),
              encoding='utf-8') as f:
        return json.load(f)


def _alive(pid: int) -> bool:
    try:
        with open(f'/proc/{pid}/stat', encoding='utf-8') as f:
            state = f.read().rpartition(')')[2].split()[0]
    except OSError:
        return False
    return state != 'Z'


def poll(sandbox_id: str) -> Optional[int]:
    """None while running, as Sandbox.poll does."""
    return None if _alive(_record(sandbox_id)['pid']) else 0


def list_ids(app_id: str, tags: Dict[str, str]) -> List[str]:
    """Running sandboxes with the tags, as Sandbox.list does by default."""
    parent = os.path.join(_state_dir(), 'sandboxes')
    found = []
    for sandbox_id in sorted(
            os.listdir(parent) if os.path.isdir(parent) else []):
        if not is_process_sandbox(sandbox_id):
            continue
        record = _record(sandbox_id)
        if (record['app_id'] == app_id and _alive(record['pid']) and
                tags.items() <= record['tags'].items()):
            found.append(sandbox_id)
    return found


def terminate(sandbox_id: str) -> None:
    pid = _record(sandbox_id)['pid']
    try:
        # Everything in the sandbox dies with the PID namespace's init.
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.time() + 30
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.02)
    directory = _sandbox_dir(sandbox_id)
    for name in ('upper', 'work', 'root'):
        shutil.rmtree(os.path.join(directory, name), ignore_errors=True)


def exec_args(sandbox_id: str, args: tuple, workdir: Optional[str],
              env: Optional[Dict[str, str]]) -> List[str]:
    """The host command that runs args in the sandbox, as exec does."""
    pid = _record(sandbox_id)['pid']
    if not _alive(pid):
        raise ProcessLookupError(f'{sandbox_id} has finished')
    merged = dict(_ENV, **(env or {}))
    variables = [f'{key}={value}' for key, value in merged.items()]
    chdir = ['-C', workdir] if workdir else []
    return [
        'nsenter', '-t',
        str(pid), '-p', '-m', '-u', '-i', '-r', '-w', '/usr/bin/env', '-i',
        *chdir, *variables, *args
    ]
