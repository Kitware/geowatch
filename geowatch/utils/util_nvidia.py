"""
Ported from netharn.device, previously called gpu_infos
"""
import ubelt as ub
import os
import warnings


class NvidiaSMIError(Exception):
    pass


def nvidia_smi(ignore_environ=False):
    """
    Run nvidia-smi and parse output

    Args:
        new_mode: internal argument that changes the underlying implementation

        ignore_environ (bool): if True respects
            CUDA_VISIBLE_DEVICES environment variable, otherwise returns
            data corresponding to physical GPU indexes.  Defaults to False.

    Returns:
        dict: info about each nvidia GPU indexed by gpu number

    Note:
        Not gaurenteed to work if CUDA is not installed.

    Warnings:
        if nvidia-smi is not found

    Example:
        >>> # xdoctest: +REQUIRES(env:HAS_CUDA)
        >>> from geowatch.utils.util_nvidia import *  # NOQA
        >>> gpus = nvidia_smi()
        >>> # xdoctest: +IGNORE_WANT
        >>> import torch
        >>> print('gpus = {}'.format(ub.repr2(gpus, nl=4)))
        >>> assert len(gpus) == torch.cuda.device_count()
        gpus = {
            0: {
                'gpu_uuid': 'GPU-348ebe36-252b-46fa-8a97-477ae331f6f4',
                'index': '0',
                'mem_avail': 10013.0,
                'mem_total': 11170.0,
                'mem_used': 1157.0,
                'memory.free': '10013 MiB',
                'memory.total': '11170 MiB',
                'memory.used': '1157 MiB',
                'name': 'GeForce GTX 1080 Ti',
                'num': 0,
                'num_compute_procs': 1,
                'procs': [
                    {
                        'gpu_num': 0,
                        'gpu_uuid': 'GPU-348ebe36-252b-46fa-8a97-477ae331f6f4',
                        'name': '/usr/bin/python',
                        'pid': '19912',
                        'type': 'C',
                        'used_memory': '567 MiB',
                    },
                ],
            },
        }
    """
    # Note: the old netharn implementation has an xml mode and an "old" mode We
    # just kept the "new" mode here, but the xml might be worth revisiting that
    # is in the notes on the bottom of this file.

    # This is slightly more robust than the old mode, but it also makes
    # more than one call to nvidia-smi and cannot return information about
    # graphics processes.
    fields = ['index', 'memory.total', 'memory.used', 'memory.free',
              'name', 'gpu_uuid']
    mode = 'query-gpu'
    try:
        gpu_rows = _query_nvidia_smi(mode, fields)
    except FileNotFoundError:
        warnings.warn('nvidia-smi not found. There are likely no nvidia gpus')
        # Lkely no GPUS
        return {}
    except Exception as ex:
        warnings.warn('Problem running nvidia-smi: {!r}'.format(ex))
        raise NvidiaSMIError

    fields = ['pid', 'name', 'gpu_uuid', 'used_memory']
    mode = 'query-compute-apps'
    proc_rows = _query_nvidia_smi(mode, fields)

    # Coerce into the old-style format for backwards compatibility
    gpus = {}
    for row in gpu_rows:
        gpu = row.copy()
        num = int(gpu['index'])
        gpu['num'] = num
        gpu['mem_used'] = float(gpu['memory.used'].strip().replace('MiB', ''))
        gpu['mem_total'] = float(gpu['memory.total'].strip().replace('MiB', ''))
        gpu['mem_avail'] = gpu['mem_total'] - gpu['mem_used']
        gpu['procs'] = []
        gpus[num] = gpu

    gpu_uuid_to_num = {gpu['gpu_uuid']: gpu['num'] for gpu in gpus.values()}

    for row in proc_rows:
        # Give each GPU info on which processes are using it
        proc = row.copy()
        proc['type'] = 'C'
        proc['gpu_num'] = gpu_uuid_to_num[proc['gpu_uuid']]
        num = proc['gpu_num']
        gpus[num]['procs'].append(proc)

    WITH_GPU_PROCS = False
    if WITH_GPU_PROCS:
        # Hacks in gpu-procs if enabled
        import re
        info = ub.cmd('nvidia-smi pmon -c 1')
        for line in info['out'].split('\n'):
            line = line.strip()
            if line and not line.startswith("#"):
                parts = re.split(r'\s+', line, maxsplit=7)
                if parts[1] != '-':
                    header = [
                        'gpu_num', 'pid', 'type', 'sm', 'mem', 'enc',
                        'dec', 'name']
                    proc = ub.dzip(header, parts)
                    proc['gpu_num'] = int(proc['gpu_num'])
                    if proc['type'] == 'G':
                        gpu = gpus[proc['gpu_num']]
                        gpu['procs'].append(proc)
                        proc['gpu_uuid'] = gpu['gpu_uuid']

    for gpu in gpus.values():
        # Let each GPU know how many processes are currently using it
        num_compute_procs = 0
        num_graphics_procs = 0
        for proc in gpu['procs']:
            if proc['type'] == 'C':
                num_compute_procs += 1
            elif proc['type'] == 'G':
                num_graphics_procs += 1
            else:
                raise NotImplementedError(proc['type'])

        # NOTE calling nvidia-smi in query mode does not seem to have
        # support for getting info about graphics procs.
        gpu['num_compute_procs'] = num_compute_procs
        if WITH_GPU_PROCS:
            gpu['num_graphics_procs'] = num_graphics_procs

    if not ignore_environ:
        # Respect CUDA_VISIBLE_DEVICES, nvidia-smi does not respect this by
        # default so remap to gain the appropriate effect.
        val = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        parts = (p.strip() for p in val.split(','))
        visible_devices = [int(p) for p in parts if p]

        if visible_devices:
            remapped = {}
            for visible_idx, real_idx in enumerate(visible_devices):
                gpu = remapped[visible_idx] = gpus[real_idx]
                gpu['index'] = str(visible_idx)
                gpu['num'] = visible_idx
                gpu['real_num'] = real_idx
            gpus = remapped

    return gpus


def _query_nvidia_smi(mode, fields):
    """
    Runs nvidia smi in query mode

    Args:
        mode (str): the query cli flag to pass to nvidia-smi
        fields (List[str]): csv header fields to query

    Returns:
        List[Dict[str, str]]: parsed csv output
    """
    header = ','.join(fields)
    command = ['nvidia-smi', f'--{mode}={header}', '--format=csv,noheader']
    info = ub.cmd(command)
    if info['ret'] != 0:
        print(info['out'])
        print(info['err'])
        raise NvidiaSMIError('unable to call nvidia-smi: ret={}'.format(
            info['ret']))
    rows = []
    for line in info['out'].split('\n'):
        line = line.strip()
        if line:
            parts = [p.strip() for p in line.split(',')]
            row = ub.dzip(fields, parts)
            rows.append(row)
    return rows


__notes__ = """
Ignore:

    # official nvidia-smi python bindings
    pip install nvidia-ml-py

    import pynvml

    # TODO: make more efficient calls to nvidia-smi

    utilization.gpu
    utilization.memory
    compute_mode
    memory.total
    memory.used
    memory.free
    index
    name
    count

    nvidia-smi pmon --count 1

    nvidia-smi  -h
    nvidia-smi  --help-query-compute-apps
    nvidia-smi  --help-query-gpu

    nvidia-smi --help-query-accounted-apps
    nvidia-smi --help-query-supported-clocks
    nvidia-smi --help-query-retired-pages
    nvidia-smi --query-accounted-apps="pid" --format=csv

    nvidia-smi  --query-gpu="index,memory.total,memory.used,memory.free,count,name,gpu_uuid" --format=csv
    nvidia-smi  --query-compute-apps="pid,name,gpu_uuid,used_memory" --format=csv
    nvidia-smi  --query-accounted-apps="gpu_name,pid" --format=csv

    import timerit
    ti = timerit.Timerit(40, bestof=5, verbose=2)
    for timer in ti.reset('new1'):
        with timer:
            gpu_info(True)
    for timer in ti.reset('old'):
        with timer:
            gpu_info(False)
    for timer in ti.reset('xml'):
        with timer:
            gpu_info('xml')

    xdev.profile_now(gpu_info)('xml')

    for timer in ti.reset('cmd'):
        with timer:
            ub.cmd(['nvidia-smi', '--query', '--xml-format'])

    for timer in ti.reset('check_output'):
        with timer:
            import subprocess
            subprocess.check_output(['nvidia-smi', '--query', '--xml-format'])


if new_mode == 'xml':
    # Parse info out of the nvidia xml query
    # note, that even though this has less calls to nvidia-smi, there
    # is a lot more output, which makes it the slowest method especially
    # for multi-gpu systems
    import xml.etree.ElementTree as ET

    info = ub.cmd(['nvidia-smi', '--query', '--xml-format'])
    if info['ret'] != 0:
        print(info['out'])
        print(info['err'])
        warnings.warn('Problem running nvidia-smi: ret={}'.format(info['ret']))
        raise NvidiaSMIError
    xml_string = info['out']
    root = ET.fromstring(xml_string)

    gpus = {}
    for gpu_elem in root.findall('gpu'):
        gpu = {}
        gpu['uuid'] = gpu_elem.find('uuid').text
        gpu['name'] = gpu_elem.find('product_name').text
        gpu['num'] = int(gpu_elem.find('minor_number').text)
        gpu['procs'] = [
            {item.tag: item.text for item in proc_elem}
            for proc_elem in gpu_elem.find('processes')
        ]

        for item in gpu_elem.find('fb_memory_usage'):
            gpu['memory.' + item.tag] = item.text

        gpu['mem_used'] = float(gpu['memory.used'].strip().replace('MiB', ''))
        gpu['mem_total'] = float(gpu['memory.total'].strip().replace('MiB', ''))
        gpu['mem_avail'] = gpu['mem_total'] - gpu['mem_used']
        gpus[gpu['num']] = gpu

        # Let each GPU know how many processes are currently using it
        num_compute_procs = 0
        num_graphics_procs = 0
        for proc in gpu['procs']:
            if proc['type'] == 'C':
                num_compute_procs += 1
            elif proc['type'] == 'G':
                num_graphics_procs += 1
            else:
                raise NotImplementedError(proc['type'])
        gpu['num_compute_procs'] = num_compute_procs
        gpu['num_graphics_procs'] = num_graphics_procs

"""
