"""A script that generates the Modal catalog.

Modal publishes its prices on https://modal.com/pricing rather than through
an API, so the rates below are copied from that page and must be updated by
hand when they change. Sandboxes are billed per second for the GPU plus the
CPU cores and memory requested; sandbox CPU and memory cost
SANDBOX_CPU_MEMORY_MULTIPLIER times the base rate.

Usage:
    python fetch_modal.py   # writes modal/vms.csv
"""

import csv
import json
import os
from typing import Dict, Tuple

# Read from https://modal.com/pricing on 2026-11-05.
GPU_USD_PER_SECOND: Dict[str, float] = {
    'B300': 0.001972,
    'B200': 0.001736,
    'H200': 0.001261,
    'H100': 0.001097,
    'RTX-PRO-6000': 0.000842,
    'A100-80GB': 0.000694,
    'A100-40GB': 0.000583,
    'L40S': 0.000542,
    'A10': 0.000306,
    'L4': 0.000222,
    'T4': 0.000164,
}
CPU_CORE_USD_PER_SECOND = 0.0000131
MEMORY_GIB_USD_PER_SECOND = 0.00000222
SANDBOX_CPU_MEMORY_MULTIPLIER = 3

# Modal GPU type -> (SkyPilot accelerator name, GPU memory GiB, vCPUs per
# GPU, memory GiB per GPU, most GPUs per sandbox). One Modal CPU core is two
# vCPUs. Modal takes any GPU count up to the maximum.
GPU_TYPES: Dict[str, Tuple[str, int, int, int, int]] = {
    'B300': ('B300', 288, 8, 32, 8),
    'B200': ('B200', 180, 8, 32, 8),
    'H200': ('H200', 141, 8, 32, 8),
    'H100': ('H100', 80, 8, 32, 8),
    'RTX-PRO-6000': ('RTXPRO6000', 96, 8, 32, 8),
    'A100-80GB': ('A100-80GB', 80, 8, 32, 8),
    'A100-40GB': ('A100', 40, 8, 32, 8),
    'L40S': ('L40S', 48, 8, 32, 8),
    'A10': ('A10', 24, 4, 16, 4),
    'L4': ('L4', 24, 4, 16, 8),
    'T4': ('T4', 16, 4, 16, 8),
}
# Larger CPU and memory for a GPU sandbox, as multiples of the shape above.
# Modal bills the cores and memory a sandbox requests, so each size is its own
# row and price, named `<count>x-<ACC>--<vcpus>cpu-<mem>gb`; the base size
# keeps its plain name. Five times the base memory is the 8 x B300 trainer of
# a DeepSeek-V4.1-Flash LoRA run (64 cores, 1280 GiB), which Modal accepts.
CPU_MULTIPLES = (1, 2)
MEMORY_MULTIPLES = (1, 2, 3, 4, 5)
SIZE_SEPARATOR = '--'

# Other SkyPilot names for a Modal GPU type: Modal's A10 is the A10G that
# AWS sells, and SkyPilot users ask for it by that name.
ALIASES: Dict[str, str] = {'A10G': 'A10'}

# CPU-only sandbox shapes: (vCPUs, memory GiB).
CPU_SHAPES = ((2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256))

# Modal places sandboxes itself; pinning a region costs extra and is not
# offered here.
REGION = 'global'


def usd_per_hour(gpu_type: str, gpu_count: int, vcpus: int,
                 memory_gib: int) -> float:
    gpu = GPU_USD_PER_SECOND[gpu_type] * gpu_count if gpu_type else 0.0
    cpu = (vcpus / 2) * CPU_CORE_USD_PER_SECOND
    memory = memory_gib * MEMORY_GIB_USD_PER_SECOND
    per_second = gpu + SANDBOX_CPU_MEMORY_MULTIPLIER * (cpu + memory)
    return round(per_second * 3600, 6)


def _gpu_info(acc_name: str, count: int, gpu_memory_gib: int) -> str:
    info = {
        'Gpus': [{
            'Name': acc_name,
            'Manufacturer': 'NVIDIA',
            'Count': count,
            'MemoryInfo': {
                'SizeInMiB': gpu_memory_gib * 1024
            },
        }],
        'TotalGpuMemoryInMiB': gpu_memory_gib * 1024 * count,
    }
    return json.dumps(info).replace('"', '\'')


def create_catalog(output_path: str = 'modal/vms.csv') -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        writer.writerow([
            'InstanceType',
            'AcceleratorName',
            'AcceleratorCount',
            'vCPUs',
            'MemoryGiB',
            'Price',
            'SpotPrice',
            'Region',
            'GpuInfo',
        ])
        gpu_types = [(gpu_type, *spec) for gpu_type, spec in GPU_TYPES.items()]
        gpu_types += [(gpu_type, alias, *GPU_TYPES[gpu_type][1:])
                      for alias, gpu_type in ALIASES.items()]
        for (gpu_type, acc_name, gpu_memory, vcpus_per_gpu, memory_per_gpu,
             max_count) in gpu_types:
            for count in range(1, max_count + 1):
                for cpu_multiple in CPU_MULTIPLES:
                    for memory_multiple in MEMORY_MULTIPLES:
                        vcpus = vcpus_per_gpu * count * cpu_multiple
                        memory = memory_per_gpu * count * memory_multiple
                        name = f'{count}x-{acc_name}'
                        if (cpu_multiple, memory_multiple) != (1, 1):
                            name += f'{SIZE_SEPARATOR}{vcpus}cpu-{memory}gb'
                        writer.writerow([
                            name,
                            acc_name,
                            count,
                            vcpus,
                            memory,
                            usd_per_hour(gpu_type, count, vcpus, memory),
                            '',
                            REGION,
                            _gpu_info(acc_name, count, gpu_memory),
                        ])
        for vcpus, memory in CPU_SHAPES:
            writer.writerow([
                f'cpu-{vcpus}x-{memory}gb',
                '',
                '',
                vcpus,
                memory,
                usd_per_hour('', 0, vcpus, memory),
                '',
                REGION,
                '',
            ])


if __name__ == '__main__':
    create_catalog()
    print('Modal catalog saved to modal/vms.csv')
