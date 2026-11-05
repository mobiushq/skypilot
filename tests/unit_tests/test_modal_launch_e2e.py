"""A Modal launch through the real backend, with a fake Modal SDK.

Provision, runtime setup, Ray, skylet, job submission, job status, logs and
teardown all really run, in a sandbox that is a process tree in Linux namespaces
on a local Ubuntu root filesystem (see fake_modal/modal/_sandbox_process.py).
There are no Modal credentials and no calls to Modal. It says nothing about
Modal's own runtime, such as Ray under gVisor.

It needs root, the network (the cluster installs uv, Python and Ray) and about
a minute of setup, so it runs only when given a root filesystem:

    sudo debootstrap --variant=minbase jammy /tmp/jammy \
        http://archive.ubuntu.com/ubuntu/
    sudo SKY_MODAL_E2E_ROOTFS=/tmp/jammy pytest \
        tests/unit_tests/test_modal_launch_e2e.py
"""
import json
import os
import pathlib
import signal
import subprocess
import sys
from typing import Any, Dict, List

import pytest

_ROOTFS = os.environ.get('SKY_MODAL_E2E_ROOTFS')
_FAKE_MODAL = pathlib.Path(__file__).parent / 'test_sky/utils/fake_modal'

pytestmark = pytest.mark.skipif(
    not _ROOTFS or os.geteuid() != 0,
    reason='Needs root and SKY_MODAL_E2E_ROOTFS; see the module docstring.')

_SYSTEM_PATH = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'


def _end_sandboxes(state: pathlib.Path) -> None:
    """So that a failed run doesn't leave Ray running on the host's ports."""
    for record in state.glob('sandboxes/*/sandbox.json'):
        pid = json.loads(record.read_text())['pid']
        try:
            if os.readlink(f'/proc/{pid}/root').startswith(str(state)):
                os.kill(pid, signal.SIGKILL)
        except OSError:
            pass  # Already ended.


def _run_driver(tmp_path: pathlib.Path,
                **extra_env: str) -> Dict[str, Dict[str, Any]]:
    home = tmp_path / 'home'
    (home / '.sky').mkdir(parents=True)
    (home / '.sky/config.yaml').write_text('allowed_clouds: [modal]\n')
    results = tmp_path / 'results.json'
    log = tmp_path / 'driver.log'
    # A clean environment: only these fake values reach SkyPilot and the fake.
    env = {
        'PATH': f'{os.path.dirname(sys.executable)}:{_SYSTEM_PATH}',
        'HOME': str(home),
        'PYTHONPATH': str(_FAKE_MODAL),
        'MODAL_TOKEN_ID': 'fake-id',
        'MODAL_TOKEN_SECRET': 'fake-secret',
        # With gRPC on, Modal must still take the legacy command-runner path.
        'SKYPILOT_ENABLE_GRPC': '1',
        'FAKE_MODAL_ROOTFS': str(_ROOTFS),
        'FAKE_MODAL_STATE': str(tmp_path / 'state'),
        'FAKE_MODAL_LOG': str(tmp_path / 'exec.log'),
        **extra_env,
    }
    try:
        with open(log, 'wb') as output:
            subprocess.run([
                sys.executable,
                str(_FAKE_MODAL / 'launch_e2e_driver.py'),
                str(results),
                str(log)
            ],
                           env=env,
                           cwd=tmp_path,
                           stdout=output,
                           stderr=subprocess.STDOUT,
                           timeout=1500,
                           check=True)
    finally:
        print(log.read_text(errors='replace')[-20000:])
        _end_sandboxes(tmp_path / 'state')
    records: List[Dict[str, Any]] = json.loads(results.read_text())
    return {record['step']: record for record in records}


def _assert_ok(steps: Dict[str, Dict[str, Any]], *names: str) -> None:
    for name in names:
        assert steps[name]['ok'], f'{name}: {steps[name]["result"]}'


@pytest.fixture(scope='module')
def steps(tmp_path_factory) -> Dict[str, Dict[str, Any]]:
    return _run_driver(tmp_path_factory.mktemp('modal-e2e'))


def test_launch_runs_setup_and_job(steps):
    _assert_ok(steps, 'launch', 'job_succeeded', 'tail_logs')
    assert "'SUCCEEDED'" in steps['job_succeeded']['result']
    log = steps['tail_logs']['log']
    for expected in ('SETUP-RAN', 'RUN-RAN', 'stderr-line', 'WORKDIR-CONTENT',
                     'MOUNT-CONTENT', 'ENV=env-value', 'RANK=0', 'RUN-DONE'):
        assert expected in log


def test_ray_and_skylet_outlive_the_execs_that_started_them(steps):
    _assert_ok(steps, 'sandbox_processes')
    processes = steps['sandbox_processes']['result']
    assert 'gcs_server' in processes
    assert 'raylet' in processes
    assert 'skylet' in processes
    assert 'defunct' not in processes


def test_status_and_queue(steps):
    _assert_ok(steps, 'status', 'queue')
    assert steps['status']['result'] == "['UP']"
    assert steps['queue']['result'] == "[(1, 'SUCCEEDED')]"


def test_failed_job_reports_failure(steps):
    _assert_ok(steps, 'exec_failing', 'job_failed', 'tail_logs_failed')
    assert "'FAILED'" in steps['job_failed']['result']
    assert 'ABOUT-TO-FAIL' in steps['tail_logs_failed']['log']
    # 100 is SkyPilot's exit code for a job that failed.
    assert steps['tail_logs_failed']['result'] == '100'


def test_script_over_modals_argument_limit_runs(steps):
    _assert_ok(steps, 'exec_large_script', 'job_large_succeeded',
               'tail_logs_large')
    assert 'LARGE-SCRIPT-RAN' in steps['tail_logs_large']['log']


def test_large_and_undecodable_output(steps):
    _assert_ok(steps, 'exec_noisy', 'job_noisy_succeeded', 'tail_logs_noisy')
    assert 'NOISY-DONE' in steps['tail_logs_noisy']['log']


def test_cancel_kills_the_running_job(steps):
    _assert_ok(steps, 'exec_long', 'job_running', 'tail_logs_running', 'cancel',
               'job_cancelled', 'job_process_gone')
    assert 'LONG-START' in steps['tail_logs_running']['log']
    assert "'CANCELLED'" in steps['job_cancelled']['result']


def test_download_logs_over_rsync(steps):
    _assert_ok(steps, 'download_logs')


def test_relaunch_reuses_the_sandbox(steps):
    _assert_ok(steps, 'relaunch_reuses_sandbox', 'same_sandbox',
               'job_again_succeeded')


def test_failing_setup_marks_the_job(steps):
    _assert_ok(steps, 'launch_failing_setup', 'job_failed_setup')
    assert "'FAILED_SETUP'" in steps['job_failed_setup']['result']


def test_down_ends_the_sandbox(steps):
    _assert_ok(steps, 'down', 'sandboxes_after_down', 'status_after_down')
    assert steps['sandboxes_after_down']['result'] == '[]'
    assert steps['status_after_down']['result'] == '[]'


def test_cluster_goes_away_when_its_sandbox_ends(steps):
    _assert_ok(steps, 'launch_second', 'second_job', 'end_sandbox',
               'status_after_sandbox_ended')
    assert steps['status_after_sandbox_ended']['result'] == '[]'
    assert not steps['exec_after_sandbox_ended']['ok']
