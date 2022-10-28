import platform
import socket
import sys
import os
import ubelt as ub
import uuid


class ProcessContext:
    """
    Context manager to track the context under which a result was computed

    Example:
        >>> from watch.utils.process_context import *
        >>> self = ProcessContext(track_emissions=True)
        >>> obj = self.start().stop()
        >>> self.add_disk_info('.')
        >>> import torch
        >>> if torch.cuda.is_available():
        >>>     device = torch.device(0)
        >>>     self.add_device_info(device)
        >>> print('obj = {}'.format(ub.repr2(obj, nl=3)))
    """

    def __init__(self, name=None, type='process', args=None, config=None,
                 extra=None, track_emissions=False):
        if args is None:
            args = sys.argv

        self.properties = {
            "name": name,
            "args": args,
            "config": config,
            "machine": None,
            "start_timestamp": None,
            "stop_timestamp": None,
            "uuid": str(uuid.uuid4()),
            "extra": extra,
        }
        self.obj = {
            "type": type,
            "properties": self.properties
        }
        self.track_emissions = track_emissions
        self.emissions_tracker = None

    def _timestamp(self):
        timestamp = ub.timestamp()
        return timestamp

    def _hostinfo(self):
        return {
            "host": socket.gethostname(),
            "user": ub.Path.home().name,
            'cwd': os.fspath(ub.Path.cwd()),
            "userhome": os.fspath(ub.Path.home()),
        }

    def _osinfo(self):
        (
            uname_system,
            _,
            uname_release,
            uname_version,
            _,
            uname_processor,
        ) = platform.uname()
        return {
            "os_name": uname_system,
            "os_release": uname_release,
            "os_version": uname_version,
            "arch": uname_processor,
        }

    def _pyinfo(self):
        return {
            "py_impl": platform.python_implementation(),
            "py_version": sys.version.replace("\n", ""),
        }

    def _meminfo(self):
        import psutil

        # TODO: could collect memory info at start and stop and intermediate
        # stages.  Here we just want info that is static wrt to the machine.
        # For now, just get the total available.
        svmem_info = psutil.virtual_memory()
        return {
            "mem_total": svmem_info.total,
        }

    def _cpuinfo(self):
        import cpuinfo

        _cpu_info = cpuinfo.get_cpu_info()
        cpu_info = {
            "cpu_brand": _cpu_info["brand_raw"],
        }
        return cpu_info

    def _machine(self):
        return ub.dict_union(
            self._hostinfo(),
            self._meminfo(),
            self._cpuinfo(),
            self._osinfo(),
            self._pyinfo(),
        )

    def start(self):
        self.properties.update({
            "machine": self._machine(),
            "start_timestamp": self._timestamp(),
            "stop_timestamp": None,
        })
        if self.track_emissions:
            self._start_emissions_tracker()

        return self

    def stop(self):
        self.properties["stop_timestamp"] = self._timestamp()
        if self.emissions_tracker is not None:
            self._stop_emissions_tracker()
        return self.obj

    def __enter__(self):
        return self.start()

    def __exit__(self, a, b, c):
        self.stop()

    def _start_emissions_tracker(self):
        try:
            from codecarbon import EmissionsTracker
            """
            # emissions_tracker = EmissionsTracker(log_level='info')
            """
            emissions_tracker = EmissionsTracker(log_level='error')
            emissions_tracker.start()
        except Exception as ex:
            # Pretend
            try:
                from codecarbon import OfflineEmissionsTracker
                print('Unable to online emissions ex = {!r}'.format(ex))
                emissions_tracker = OfflineEmissionsTracker(
                    country_iso_code='USA',
                    region='Virginia',
                    cloud_provider='aws',
                    cloud_region='us-east-1',
                    # country_2letter_iso_code='us'
                )
                emissions_tracker.start()
            except Exception as ex:
                print('Unable to track emissions ex = {!r}'.format(ex))
            else:
                self.emissions_tracker = emissions_tracker

    def _stop_emissions_tracker(self):
        if self.emissions_tracker is None:
            self.properties['emissions'] = None
            return
        self.emissions_tracker.stop()
        summary = self.emissions_tracker.final_emissions_data
        co2_kg = summary.emissions
        total_kWH = summary.energy_consumed
        # summary.cloud_provider
        # summary.cloud_region
        # summary.duration
        # summary.emissions_rate
        # summary.cpu_power
        # summary.gpu_power
        # summary.ram_power
        # summary.cpu_energy
        # summary.gpu_energy
        # summary.ram_energy
        emissions = {
            'co2_kg': co2_kg,
            'total_kWH': total_kWH,
            'run_id': str(self.emissions_tracker.run_id),
        }
        try:
            import pint
        except Exception as ex:
            print('ex = {!r}'.format(ex))
        else:
            reg = pint.UnitRegistry()
            if co2_kg is None:
                co2_kg = float('nan')
            co2_ton = (co2_kg * reg.kg).to(reg.metric_ton)
            dollar_per_ton = 15 / reg.metric_ton  # cotap rate
            emissions['co2_ton'] = co2_ton.m
            emissions['est_dollar_to_offset'] = (co2_ton * dollar_per_ton).m
        self.properties['emissions'] = emissions

    def add_device_info(self, device):
        import torch
        try:
            device_info = {
                'device_index': device.index,
                'device_type': device.type,
            }
            try:
                device_props = torch.cuda.get_device_properties(device)
                capabilities = (device_props.multi_processor_count, device_props.minor)
                device_info.update({
                    'device_name': device_props.name,
                    'total_vram': device_props.total_memory,
                    'reserved_vram': torch.cuda.memory_reserved(device),
                    'allocated_vram': torch.cuda.memory_allocated(device),
                    'device_capabilities': capabilities,
                    'device_multi_processor_count': device_props.multi_processor_count,
                })
            except Exception:
                pass
        except Exception as ex:
            print('ex = {!r}'.format(ex))
            device_info = str(ex)
        self.properties['device_info'] = device_info

    def add_disk_info(self, path):
        try:
            from watch.utils import util_hardware
            # Get information about disk used in this process
            disk_info = util_hardware.disk_info_of_path(path)
        except Exception as ex:
            print('ex = {!r}'.format(ex))
            print('ex = {!r}'.format(ex))
            disk_info = str(ex)
        self.properties['disk_info'] = disk_info
