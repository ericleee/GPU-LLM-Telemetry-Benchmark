"""
GPU telemetry exporter — polls NVML and exposes Prometheus metrics on :9100.

Reads the NVIDIA driver directly through NVML (the same interface `nvidia-smi`
uses) every 250 ms and serves the values at http://localhost:9100/metrics for
Prometheus to scrape.

Exposed metrics
    gpu_utilization_percent          GPU core utilization (%)
    gpu_memory_utilization_percent   Memory-controller utilization (%)
    gpu_power_watts                  Board power draw (W)
    gpu_temperature_celsius          Core temperature (C)
    gpu_memory_used_bytes            VRAM in use (bytes)
    gpu_memory_total_bytes           VRAM total (bytes)
    gpu_clock_sm_mhz                 SM / shader clock (MHz)
    gpu_clock_mem_mhz                Memory clock (MHz)
    gpu_throttle_reason{reason=...}  1 when that clock-throttle reason is active, else 0
    gpu_poll_errors_total            NVML reads that raised (so a silent N/A can't look like 0)
"""

import logging
import time

import pynvml
from prometheus_client import Counter, Gauge, start_http_server

EXPORTER_PORT = 9100
POLL_INTERVAL_S = 0.25        # 250 ms -> 4 Hz
GPU_INDEX = 0

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("gpu_exporter")

# ── Prometheus metrics ───────────────────────────────────────────────────────
g_util      = Gauge("gpu_utilization_percent",        "GPU core utilization (percent)")
g_mem_util  = Gauge("gpu_memory_utilization_percent", "GPU memory-controller utilization (percent)")
g_power     = Gauge("gpu_power_watts",                "GPU board power draw (watts)")
g_temp      = Gauge("gpu_temperature_celsius",        "GPU core temperature (Celsius)")
g_mem_used  = Gauge("gpu_memory_used_bytes",          "GPU memory used (bytes)")
g_mem_total = Gauge("gpu_memory_total_bytes",         "GPU memory total (bytes)")
g_clock_sm  = Gauge("gpu_clock_sm_mhz",               "GPU SM / shader clock (MHz)")
g_clock_mem = Gauge("gpu_clock_mem_mhz",              "GPU memory clock (MHz)")
g_throttle  = Gauge("gpu_throttle_reason",            "Active clock-throttle reason (1=active, 0=inactive)", ["reason"])
c_errors    = Counter("gpu_poll_errors_total",        "NVML reads that raised an exception")


# ── Throttle-reason bitmask map ──────────────────────────────────────────────
# nvmlDeviceGetCurrentClocksThrottleReasons() returns a bitmask; each bit means a
# different reason the GPU is holding clocks below max. We decode it into one 0/1
# series per reason so each is independently graph-able (e.g. thermal slowdown).
#
# Newer nvidia-ml-py renamed the `nvmlClocksThrottleReason*` constants to
# `nvmlClocksEventReason*` (old names kept as aliases). Resolve both, then fall
# back to the stable NVML bit value so this works across library versions.
def _bit(default, *names):
    for name in names:
        value = getattr(pynvml, name, None)
        if value is not None:
            return value
    return default

THROTTLE_REASONS = {
    "gpu_idle":           _bit(0x001, "nvmlClocksThrottleReasonGpuIdle",                 "nvmlClocksEventReasonGpuIdle"),
    "app_clocks_setting": _bit(0x002, "nvmlClocksThrottleReasonApplicationsClocksSetting", "nvmlClocksEventReasonApplicationsClocksSetting"),
    "sw_power_cap":       _bit(0x004, "nvmlClocksThrottleReasonSwPowerCap",              "nvmlClocksEventReasonSwPowerCap"),
    "hw_slowdown":        _bit(0x008, "nvmlClocksThrottleReasonHwSlowdown",              "nvmlClocksEventReasonHwSlowdown"),
    "sync_boost":         _bit(0x010, "nvmlClocksThrottleReasonSyncBoost",               "nvmlClocksEventReasonSyncBoost"),
    "sw_thermal":         _bit(0x020, "nvmlClocksThrottleReasonSwThermalSlowdown",       "nvmlClocksEventReasonSwThermalSlowdown"),
    "hw_thermal":         _bit(0x040, "nvmlClocksThrottleReasonHwThermalSlowdown",       "nvmlClocksEventReasonHwThermalSlowdown"),
    "hw_power_brake":     _bit(0x080, "nvmlClocksThrottleReasonHwPowerBrakeSlowdown",    "nvmlClocksEventReasonHwPowerBrakeSlowdown"),
    "display_clocks":     _bit(0x100, "nvmlClocksThrottleReasonDisplayClockSetting",     "nvmlClocksEventReasonDisplayClockSetting"),
}

# The reasons-query function was also renamed; bind whichever this version ships.
_get_throttle_mask = (
    getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
    or getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
)


def poll_once(handle) -> None:
    """Read every metric once. Each read is isolated so one unsupported field
    can't stop the others (it just bumps gpu_poll_errors_total)."""
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        g_util.set(util.gpu)
        g_mem_util.set(util.memory)
    except pynvml.NVMLError:
        c_errors.inc()

    try:
        g_power.set(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)   # mW -> W
    except pynvml.NVMLError:
        c_errors.inc()

    try:
        g_temp.set(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except pynvml.NVMLError:
        c_errors.inc()

    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        g_mem_used.set(mem.used)
        g_mem_total.set(mem.total)
    except pynvml.NVMLError:
        c_errors.inc()

    try:
        g_clock_sm.set(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
    except pynvml.NVMLError:
        c_errors.inc()

    try:
        g_clock_mem.set(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
    except pynvml.NVMLError:
        c_errors.inc()

    if _get_throttle_mask is not None:
        try:
            mask = _get_throttle_mask(handle)
            for reason, bit in THROTTLE_REASONS.items():
                g_throttle.labels(reason=reason).set(1 if mask & bit else 0)
        except pynvml.NVMLError:
            c_errors.inc()


def main() -> None:
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        log.info("NVML initialized - GPU %d: %s", GPU_INDEX, name)

        # Pre-create every throttle series at 0 so they all show up immediately,
        # even before any reason ever fires.
        for reason in THROTTLE_REASONS:
            g_throttle.labels(reason=reason).set(0)

        start_http_server(EXPORTER_PORT)
        log.info("Serving metrics on http://localhost:%d/metrics  (polling every %.0f ms)",
                 EXPORTER_PORT, POLL_INTERVAL_S * 1000)

        while True:
            poll_once(handle)
            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
