"""A script that generates the Modal catalog.

Usage:
    python fetch_modal.py

Modal (https://modal.com) has no public pricing API, so the per-second list
prices from https://modal.com/pricing are recorded below. Sandboxes are
billed per second for GPU, CPU and memory; CPU and memory on sandboxes are
non-preemptible and carry Modal's 3x multiplier over function prices. GPU
prices are the same as for functions.

GPU instance types are generated per GPU type and count using fixed
per-GPU CPU and memory requests, which the provisioner passes to Modal as
the sandbox's `cpu` and `memory` reservation. Modal schedules sandboxes
across its own fleet, represented by the single `global` region. Modal
sandboxes cannot be preempted, so there is no spot price.

The catalog is bundled with the package rather than fetched from the hosted
catalog repository; this script rewrites the bundled copy.
"""

import csv
import io
import json
import os
from typing import Dict, List, NamedTuple, Optional

REGION = 'global'

# Sandbox CPU and memory, per second (non-preemptible, 3x multiplier).
# A Modal CPU core is a physical core; the catalog's vCPUs column holds
# Modal cores so that `--cpus N` maps directly to `cpu=N` on Modal.
CPU_CORE_PRICE_PER_SECOND = 0.00003942
MEMORY_GIB_PRICE_PER_SECOND = 0.00000667


class GpuSpec(NamedTuple):
    # Value passed as `gpu=` to modal.Sandbox.create (with `:<count>`).
    modal_name: str
    price_per_second: float
    gpu_memory_gib: int
    max_count: int
    cpus_per_gpu: int
    memory_gib_per_gpu: int


# Keyed by SkyPilot accelerator name. CPU and memory reservations are
# billed at sandbox rates, so they are kept modest: a Modal core is a
# physical core (2 vCPUs).
GPUS: Dict[str, GpuSpec] = {
    'T4': GpuSpec('T4', 0.000164, 16, 8, 2, 16),
    'L4': GpuSpec('L4', 0.000222, 24, 8, 2, 16),
    'A10G': GpuSpec('A10G', 0.000306, 24, 4, 2, 16),
    'L40S': GpuSpec('L40S', 0.000542, 48, 8, 4, 64),
    'A100': GpuSpec('A100-40GB', 0.000583, 40, 8, 4, 64),
    'A100-80GB': GpuSpec('A100-80GB', 0.000694, 80, 8, 4, 64),
    'H100': GpuSpec('H100', 0.001097, 80, 8, 4, 64),
    'H200': GpuSpec('H200', 0.001261, 141, 8, 4, 64),
    'B200': GpuSpec('B200', 0.001736, 180, 8, 4, 64),
}

# CPU-only sandbox shapes: (cores, memory GiB).
CPU_SHAPES = ((1, 4), (2, 8), (4, 16), (8, 32), (16, 64), (32, 128))

COLUMNS = [
    'InstanceType',
    'AcceleratorName',
    'AcceleratorCount',
    'vCPUs',
    'MemoryGiB',
    'Price',
    'SpotPrice',
    'Region',
    'GpuInfo',
]


def gpu_instance_type(acc_name: str, count: int) -> str:
    return f'{count}x-{acc_name}'


def cpu_instance_type(cpus: int, memory_gib: int) -> str:
    return f'cpu-{cpus}x-{memory_gib}gb'


def modal_gpu_string(acc_name: str, count: int) -> str:
    """Returns the `gpu=` argument for modal.Sandbox.create."""
    modal_name = GPUS[acc_name].modal_name
    return modal_name if count == 1 else f'{modal_name}:{count}'


def _hourly(cpus: float, memory_gib: float, gpu: Optional[GpuSpec],
            count: int) -> float:
    per_second = (cpus * CPU_CORE_PRICE_PER_SECOND +
                  memory_gib * MEMORY_GIB_PRICE_PER_SECOND)
    if gpu is not None:
        per_second += count * gpu.price_per_second
    return round(per_second * 3600, 6)


def _gpu_info(acc_name: str, count: int, gpu_memory_gib: int) -> str:
    gpu_memory_mib = gpu_memory_gib * 1024
    gpu_info = {
        'Gpus': [{
            'Name': acc_name,
            'Manufacturer': 'NVIDIA',
            'Count': count,
            'MemoryInfo': {
                'SizeInMiB': gpu_memory_mib
            },
        }],
        'TotalGpuMemoryInMiB': gpu_memory_mib * count,
    }
    return json.dumps(gpu_info).replace('"', '\'')


def catalog_rows() -> List[List[object]]:
    rows: List[List[object]] = []
    for acc_name, gpu in GPUS.items():
        for count in range(1, gpu.max_count + 1):
            cpus = gpu.cpus_per_gpu * count
            memory = gpu.memory_gib_per_gpu * count
            rows.append([
                gpu_instance_type(acc_name, count),
                acc_name,
                count,
                cpus,
                memory,
                _hourly(cpus, memory, gpu, count),
                '',
                REGION,
                _gpu_info(acc_name, count, gpu.gpu_memory_gib),
            ])
    for cpus, memory in CPU_SHAPES:
        rows.append([
            cpu_instance_type(cpus, memory),
            '',
            '',
            cpus,
            memory,
            _hourly(cpus, memory, None, 0),
            '',
            REGION,
            '',
        ])
    return rows


def catalog_csv() -> str:
    """Returns the Modal catalog as CSV text."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=',', quotechar='"', lineterminator='\n')
    writer.writerow(COLUMNS)
    writer.writerows(catalog_rows())
    return buf.getvalue()


BUNDLED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'bundled', 'v8', 'modal', 'vms.csv')


def create_catalog(output_path: str = BUNDLED_PATH) -> None:
    """Writes the Modal catalog CSV file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, mode='w', encoding='utf-8') as f:
        f.write(catalog_csv())


if __name__ == '__main__':
    create_catalog()
    print(f'Modal catalog saved to {BUNDLED_PATH}')
