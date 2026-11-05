"""Drives a Modal launch through the real backend: test_modal_launch_e2e.py.

Runs as its own process, against the fake Modal SDK, in a temporary HOME. Takes
the path to write one JSON record per step to, and the path of its own stdout.
"""
import base64
import io
import json
import os
import sys
import tarfile
import tempfile
import time
import traceback
from typing import Any, Callable, Dict, List, Tuple

import modal

import sky
from sky import core
from sky import execution
from sky import global_user_state
from sky import models
from sky.provision.modal import utils as modal_utils
from sky.utils import common_utils

CLUSTER = 'e2e-modal'
LOST_CLUSTER = 'e2e-modal-lost'
AUTODOWN_CLUSTER = 'e2e-modal-autodown'
# Over the 2**16 bytes that one Modal exec can carry.
LARGE_SCRIPT_BYTES = 100 * 1024

_results: List[Dict[str, Any]] = []
_results_path = sys.argv[1]
_log_path = sys.argv[2]


def _step(name: str, function: Callable[[], Any]) -> Tuple[bool, Any]:
    print(f'===== STEP {name}', flush=True)
    log_start = os.path.getsize(_log_path)
    started = time.time()
    record: Dict[str, Any] = {'step': name}
    try:
        result = function()
        record['ok'] = True
    except BaseException as e:  # pylint: disable=broad-except
        traceback.print_exc()
        result = e
        record['ok'] = False
    record['seconds'] = round(time.time() - started, 1)
    record['result'] = repr(result)[:600]
    sys.stdout.flush()
    with open(_log_path, encoding='utf-8', errors='replace') as f:
        f.seek(log_start)
        record['log'] = f.read()[-20000:]
    _results.append(record)
    print(
        f'===== RESULT {name} ok={record["ok"]} {record["seconds"]}s '
        f'{record["result"]}',
        flush=True)
    with open(_results_path, 'w', encoding='utf-8') as f:
        json.dump(_results, f, indent=1)
    return record['ok'], result


def _task(name: str, run: str, **kwargs: Any) -> sky.Task:
    task = sky.Task(name=name, run=run, **kwargs)
    task.set_resources(sky.Resources(infra='modal', cpus='2+'))
    return task


def _launch(cluster: str, task: sky.Task) -> int:
    job_id, _ = execution.launch(task,
                                 cluster_name=cluster,
                                 _disable_controller_check=True)
    return job_id


def _wait_for_job(job_id: int, cluster: str = CLUSTER) -> str:
    deadline = time.time() + 120
    while True:
        status = core.job_status(cluster, [job_id])[job_id]
        if status is not None and status.is_terminal():
            return status.value
        if time.time() > deadline:
            raise TimeoutError(f'Job {job_id} is still {status}')
        time.sleep(1)


def _sandbox(cluster: str) -> Any:
    handle = global_user_state.get_handle_from_cluster_name(cluster)
    sandboxes = modal_utils.list_cluster_sandboxes(handle.cluster_name_on_cloud)
    assert len(sandboxes) == 1, sandboxes
    return sandboxes[0]


def _in_sandbox(cluster: str, command: str) -> str:
    process = _sandbox(cluster).exec('bash', '-c', command)
    output = ''.join(process.stdout)
    process.wait()
    return output


def _install_fake_sdk(cluster: str) -> str:
    """Puts this fake in place of the modal package in the cluster's runtime.

    The runtime installs the real SDK; autodown on the sandbox must call the
    fake's control plane instead of Modal.
    """
    python = '~/skypilot-runtime/bin/python'
    package = _in_sandbox(
        cluster, f'{python} -c "import modal, os; '
        'print(os.path.dirname(modal.__file__))"').strip()
    assert package.endswith('/modal'), package
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as tar:
        tar.add(os.path.dirname(modal.__file__),
                arcname='.',
                filter=lambda info: None
                if '__pycache__' in info.name else info)
    encoded = base64.b64encode(archive.getvalue()).decode()
    return _in_sandbox(
        cluster, f'rm -rf {package} && mkdir {package} && '
        f'echo {encoded} | base64 -d | tar -xz -C {package} && '
        f'{python} -c "import modal; print(modal.__file__)"')


def _wait_for_no_sandbox(cluster: str) -> str:
    handle = global_user_state.get_handle_from_cluster_name(cluster)
    deadline = time.time() + 300
    while modal_utils.list_cluster_sandboxes(handle.cluster_name_on_cloud):
        if time.time() > deadline:
            raise TimeoutError(f'{cluster} still has its sandbox')
        time.sleep(2)
    return 'gone'


def _main() -> None:
    assert 'fake_modal' in modal.__file__, modal.__file__
    # Sandboxes created from here on get its URL in their environment.
    os.environ['FAKE_MODAL_SERVER'], served = modal.serve_control_plane()
    global_user_state.add_or_update_user(
        models.User(id=common_utils.get_user_hash(), name='e2e'))

    workdir = tempfile.mkdtemp()
    with open(os.path.join(workdir, 'from_workdir.txt'), 'w',
              encoding='utf-8') as f:
        f.write('WORKDIR-CONTENT\n')
    mounted = os.path.join(workdir, 'mounted.txt')
    with open(mounted, 'w', encoding='utf-8') as f:
        f.write('MOUNT-CONTENT\n')
    first = _task('hello',
                  'echo RUN-RAN; echo stderr-line >&2; cat from_workdir.txt; '
                  'cat /root/mounted.txt; echo "ENV=$E2E_VAR"; '
                  'echo "RANK=$SKYPILOT_NODE_RANK"; sleep 1; echo RUN-DONE',
                  workdir=workdir,
                  setup='echo SETUP-RAN',
                  envs={'E2E_VAR': 'env-value'},
                  file_mounts={'/root/mounted.txt': mounted})
    launched, job_id = _step('launch', lambda: _launch(CLUSTER, first))
    if launched:
        _step('job_succeeded', lambda: _wait_for_job(job_id))
        _step('tail_logs', lambda: core.tail_logs(CLUSTER, job_id, follow=True))
        _step(
            'sandbox_processes', lambda: _in_sandbox(
                CLUSTER, 'ps -eo stat,args | grep -E "gcs_server|raylet/raylet|'
                'skylet" | grep -v grep | cut -c1-90'))
        sandbox_id = _sandbox(CLUSTER).object_id
        _step(
            'status', lambda: [
                r['status'].value
                for r in core.status(cluster_names=[CLUSTER],
                                     refresh=sky.StatusRefreshMode.FORCE)
            ])
        _step(
            'queue', lambda: [(job['job_id'], job['status'].value)
                              for job in core.queue(CLUSTER)])

        failing = _task('fail', 'echo ABOUT-TO-FAIL; exit 7')
        _, failing_id = _step(
            'exec_failing',
            lambda: execution.exec(failing, cluster_name=CLUSTER)[0])
        _step('job_failed', lambda: _wait_for_job(failing_id))
        _step('tail_logs_failed',
              lambda: core.tail_logs(CLUSTER, failing_id, follow=True))

        filler = '# ' + 'x' * 78 + '\n'
        large_script = (filler * (LARGE_SCRIPT_BYTES // len(filler)) +
                        'echo LARGE-SCRIPT-RAN')
        _, large_id = _step(
            'exec_large_script',
            lambda: execution.exec(_task('large', large_script),
                                   cluster_name=CLUSTER)[0])
        _step('job_large_succeeded', lambda: _wait_for_job(large_id))
        _step('tail_logs_large',
              lambda: core.tail_logs(CLUSTER, large_id, follow=True))

        noisy = _task(
            'noisy',
            "head -c 2000000 /dev/zero | base64; printf '\\377\\376\\n';"
            ' echo NOISY-DONE')
        _, noisy_id = _step(
            'exec_noisy',
            lambda: execution.exec(noisy, cluster_name=CLUSTER)[0])
        _step('job_noisy_succeeded', lambda: _wait_for_job(noisy_id))
        _step('tail_logs_noisy',
              lambda: core.tail_logs(CLUSTER, noisy_id, follow=True))

        running = _task('long', 'echo LONG-START; sleep 987')
        _, running_id = _step(
            'exec_long',
            lambda: execution.exec(running, cluster_name=CLUSTER)[0])

        def _wait_for_sleep() -> str:
            deadline = time.time() + 60
            while time.time() < deadline:
                found = _in_sandbox(CLUSTER, 'pgrep -f "^sleep 987" || true')
                if found.strip():
                    return found
                time.sleep(1)
            raise TimeoutError('The job did not start')

        _step('job_running', _wait_for_sleep)
        _step('tail_logs_running',
              lambda: core.tail_logs(CLUSTER, running_id, follow=False))
        _step('cancel', lambda: core.cancel(CLUSTER, job_ids=[running_id]))
        _step('job_cancelled', lambda: _wait_for_job(running_id))

        def _job_process_gone() -> str:
            deadline = time.time() + 30
            found = ''
            while time.time() < deadline:
                found = _in_sandbox(CLUSTER, 'pgrep -f "^sleep 987" || true')
                if not found.strip():
                    return 'gone'
                time.sleep(1)
            raise AssertionError(f'Still running after cancel: {found}')

        _step('job_process_gone', _job_process_gone)

        _step('download_logs',
              lambda: core.download_logs(CLUSTER, [str(job_id)]))

        _, again_id = _step(
            'relaunch_reuses_sandbox',
            lambda: _launch(CLUSTER, _task('again', 'echo AGAIN-RAN')))
        _step('same_sandbox',
              lambda: _sandbox(CLUSTER).object_id == sandbox_id or 1 / 0)
        _step('job_again_succeeded', lambda: _wait_for_job(again_id))

        bad_setup = _task('badsetup',
                          'echo NOT-REACHED',
                          setup='echo SETUP-FAILS; exit 3')
        _, bad_setup_id = _step('launch_failing_setup',
                                lambda: _launch(CLUSTER, bad_setup))
        _step('job_failed_setup', lambda: _wait_for_job(bad_setup_id))
    _step('down', lambda: core.down(CLUSTER))
    _step(
        'sandboxes_after_down', lambda: [
            sandbox.object_id for sandbox in modal.Sandbox.list(
                app_id=modal.App.lookup('skypilot').app_id)
        ])
    _step('status_after_down', lambda: core.status(cluster_names=[CLUSTER]))

    # A sandbox that ends on its own, as at its timeout: the cluster goes away.
    lost, lost_job = _step('launch_second',
                           lambda: _launch(LOST_CLUSTER, _task('x', 'echo X')))
    if lost:
        _step('second_job', lambda: _wait_for_job(lost_job, LOST_CLUSTER))
        _step('end_sandbox', lambda: _sandbox(LOST_CLUSTER).terminate())
        _step(
            'exec_after_sandbox_ended',
            lambda: execution.exec(_task('y', 'echo Y'),
                                   cluster_name=LOST_CLUSTER))
        _step(
            'status_after_sandbox_ended',
            lambda: core.status(cluster_names=[LOST_CLUSTER],
                                refresh=sky.StatusRefreshMode.FORCE))

    # Autodown: the sandbox's skylet ends its own sandbox, through the SDK on
    # the sandbox and the token mounted there.
    auto, auto_job = _step(
        'launch_autodown',
        lambda: _launch(AUTODOWN_CLUSTER, _task('z', 'echo Z')))
    if auto:
        _step('autodown_job', lambda: _wait_for_job(auto_job, AUTODOWN_CLUSTER))
        _step('install_fake_sdk', lambda: _install_fake_sdk(AUTODOWN_CLUSTER))
        _step(
            'set_autodown',
            lambda: core.autostop(AUTODOWN_CLUSTER, idle_minutes=1, down=True))
        _step('autodown_ended_sandbox',
              lambda: _wait_for_no_sandbox(AUTODOWN_CLUSTER))
        _step('control_plane_calls', lambda: list(served))
        _step(
            'status_after_autodown',
            lambda: core.status(cluster_names=[AUTODOWN_CLUSTER],
                                refresh=sky.StatusRefreshMode.FORCE))


if __name__ == '__main__':
    _main()
