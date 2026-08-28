"""Optional, portable inference-only environmental-footprint telemetry.

NVML's cumulative GPU-energy counter is the primary measurement.  CodeCarbon
is retained as a secondary, broader tracked-process cross-check.  Importing
this module does not import either optional dependency.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any


CAMPAIGN_CARBON_PROFILE = "accepted-campaign"
CAMPAIGN_CARBON_INTENSITY_G_PER_KWH = 59.0
WATER_LOW_L_PER_KWH = 1.8
WATER_HIGH_L_PER_KWH = 4.0


def _optional_module(name: str, extra: str) -> Any:
    """Load an optional dependency only when telemetry is requested."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"Telemetry requires optional dependency '{name}'. Install with `uv sync --group {extra}`."
        ) from error


def _as_text(value: object) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _normalize_uuid(value: object | None) -> str | None:
    return None if value is None else _as_text(value).removeprefix("GPU-")


def _numeric(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def nvml_energy(start_mj: float | None, end_mj: float | None, duration_seconds: float | None = None) -> dict[str, Any]:
    """Convert NVML cumulative millijoules to joules, Wh, and kWh.

    A decreasing counter is unavailable, matching the accepted campaign; no
    counter-wrap correction is inferred.
    """
    if start_mj is None or end_mj is None:
        return {"status": "unavailable", "measurement_type": "direct_hardware_counter"}
    delta_mj = end_mj - start_mj
    if delta_mj < 0:
        return {
            "status": "unavailable",
            "measurement_type": "direct_hardware_counter",
            "error": "NVML cumulative energy counter decreased.",
        }
    joules = delta_mj / 1_000.0
    result: dict[str, Any] = {
        "status": "measured",
        "measurement_type": "direct_hardware_counter",
        "start_mj": start_mj,
        "end_mj": end_mj,
        "delta_mj": delta_mj,
        "joules": joules,
        "wh": joules / 3_600.0,
        "kwh": joules / 3_600_000.0,
    }
    if duration_seconds is not None and duration_seconds > 0:
        result["average_power_w"] = joules / duration_seconds
    return result


def carbon_kg(energy_kwh: float | None, intensity_g_per_kwh: float | None) -> float | None:
    """Return operational CO2e in kg from kWh and gCO2e/kWh."""
    if energy_kwh is None or intensity_g_per_kwh is None:
        return None
    return energy_kwh * intensity_g_per_kwh / 1_000.0


def carbon_provenance(*, carbon_profile: str | None, carbon_intensity_g_per_kwh: float | None) -> dict[str, Any]:
    """Resolve an explicit campaign profile or a caller-provided intensity."""
    if carbon_profile is not None and carbon_intensity_g_per_kwh is not None:
        raise ValueError("Choose either a carbon profile or a caller-provided carbon intensity, not both.")
    if carbon_profile == CAMPAIGN_CARBON_PROFILE:
        return {
            "intensity_g_per_kwh": CAMPAIGN_CARBON_INTENSITY_G_PER_KWH,
            "units": "gCO2e/kWh",
            "source": "Ontario/ECCC",
            "profile": CAMPAIGN_CARBON_PROFILE,
            "location_confirmation": "location-unconfirmed",
        }
    if carbon_profile is not None:
        raise ValueError(f"Unknown carbon profile: {carbon_profile}")
    if carbon_intensity_g_per_kwh is not None:
        if carbon_intensity_g_per_kwh < 0:
            raise ValueError("Carbon intensity must be non-negative.")
        return {
            "intensity_g_per_kwh": carbon_intensity_g_per_kwh,
            "units": "gCO2e/kWh",
            "source": "caller-provided",
            "profile": None,
            "location_confirmation": "caller-responsibility",
        }
    return {
        "intensity_g_per_kwh": None,
        "units": "gCO2e/kWh",
        "source": "not-configured",
        "profile": None,
        "location_confirmation": "not-configured",
    }


def water_estimate(
    energy_kwh: float | None, *, evaluated_items: int | None, scope: str, energy_basis: str
) -> dict[str, Any]:
    """Return an explicitly estimated DIA-style water range."""
    if energy_kwh is None:
        return {
            "status": "unavailable",
            "quality": "ESTIMATED",
            "low_liters": None,
            "high_liters": None,
            "per_item_low_liters": None,
            "per_item_high_liters": None,
            "factor_low_l_per_kwh": WATER_LOW_L_PER_KWH,
            "factor_high_l_per_kwh": WATER_HIGH_L_PER_KWH,
            "energy_kwh": None,
            "energy_basis": energy_basis,
            "scope": scope,
        }
    low = energy_kwh * WATER_LOW_L_PER_KWH
    high = energy_kwh * WATER_HIGH_L_PER_KWH
    per_item_low = low / evaluated_items if evaluated_items is not None and evaluated_items > 0 else None
    per_item_high = high / evaluated_items if evaluated_items is not None and evaluated_items > 0 else None
    return {
        "status": "estimated" if evaluated_items is None or evaluated_items > 0 else "estimated_per_item_unavailable",
        "quality": "ESTIMATED",
        "low_liters": low,
        "high_liters": high,
        "per_item_low_liters": per_item_low,
        "per_item_high_liters": per_item_high,
        "factor_low_l_per_kwh": WATER_LOW_L_PER_KWH,
        "factor_high_l_per_kwh": WATER_HIGH_L_PER_KWH,
        "energy_kwh": energy_kwh,
        "energy_basis": energy_basis,
        "scope": scope,
    }


def _sync_cuda(torch: Any, warnings: list[str]) -> None:
    try:
        torch.cuda.synchronize()
    except Exception as error:
        warnings.append(f"CUDA synchronization unavailable: {type(error).__name__}: {error}")


def _occupancy(pynvml: Any, handle: Any) -> dict[str, Any]:
    """Record only public-safe aggregate process-occupancy evidence."""
    try:
        getter = getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None) or getattr(
            pynvml, "nvmlDeviceGetComputeRunningProcesses", None
        )
        if getter is None:
            raise RuntimeError("No NVML compute-process enumeration API is available.")
        current_pid = os.getpid()
        pids = {int(process.pid) for process in getter(handle) if getattr(process, "pid", None) is not None}
        other_count = sum(pid != current_pid for pid in pids)
        return {
            "status": "measured",
            "active_compute_process_count": len(pids),
            "other_compute_process_count": other_count,
            "competing_process_detected": other_count > 0,
        }
    except Exception as error:
        return {
            "status": "unavailable",
            "active_compute_process_count": None,
            "other_compute_process_count": None,
            "competing_process_detected": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _nvml_device(torch: Any, pynvml: Any, requested_cuda_index: int | None) -> tuple[dict[str, Any], Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    cuda_index = int(torch.cuda.current_device())
    if requested_cuda_index is not None and requested_cuda_index != cuda_index:
        raise RuntimeError(
            f"Requested CUDA device {requested_cuda_index} does not match active CUDA device {cuda_index}."
        )
    cuda_uuid = _normalize_uuid(getattr(torch.cuda.get_device_properties(cuda_index), "uuid", None))
    pynvml.nvmlInit()
    devices: list[tuple[dict[str, Any], Any]] = []
    for index in range(int(pynvml.nvmlDeviceGetCount())):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        devices.append(
            (
                {
                    "index": index,
                    "uuid": _normalize_uuid(pynvml.nvmlDeviceGetUUID(handle)),
                    "name": _as_text(pynvml.nvmlDeviceGetName(handle)),
                },
                handle,
            )
        )
    match = next(
        ((identity, handle) for identity, handle in devices if cuda_uuid and identity["uuid"] == cuda_uuid), None
    )
    if match is None and len(devices) == 1:
        identity, handle = devices[0]
        identity = {**identity, "mapping_warning": "CUDA UUID unavailable; sole visible NVML GPU selected."}
    elif match is not None:
        identity, handle = match
    else:
        raise RuntimeError("Unable to map the active CUDA GPU to an NVML device.")
    identity.update(
        {
            "cuda_index": cuda_index,
            "cuda_uuid": cuda_uuid,
            "nvml_version": _as_text(pynvml.nvmlSystemGetNVMLVersion()),
        }
    )
    return identity, handle


def _codecarbon_tracker(codecarbon: Any, gpu_index: int, country_iso_code: str | None, region: str | None) -> Any:
    tracker_type = getattr(codecarbon, "OfflineEmissionsTracker", None)
    if tracker_type is None:
        raise RuntimeError("CodeCarbon OfflineEmissionsTracker is unavailable.")
    kwargs: dict[str, Any] = {
        "project_name": "eval-efficiency",
        "measure_power_secs": 1,
        "tracking_mode": "process",
        "gpu_ids": [gpu_index],
        "save_to_file": False,
        "save_to_api": False,
        "save_to_logger": False,
        "log_level": "error",
    }
    if country_iso_code is not None:
        kwargs["country_iso_code"] = country_iso_code
    if region is not None:
        kwargs["region"] = region
    return tracker_type(**kwargs)


def _codecarbon_identity(pynvml: Any, gpu_index: int | None, nvml_uuid: str | None) -> dict[str, Any]:
    if gpu_index is None:
        return {"status": "unavailable", "same_physical_gpu": None}
    try:
        uuid = _normalize_uuid(pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(gpu_index)))
        return {
            "status": "resolved",
            "gpu_index": gpu_index,
            "uuid": uuid,
            "same_physical_gpu": uuid == nvml_uuid if uuid is not None and nvml_uuid is not None else None,
        }
    except Exception as error:
        return {"status": "unavailable", "same_physical_gpu": None, "error": f"{type(error).__name__}: {error}"}


def _extract_codecarbon(tracker: Any, codecarbon: Any, pynvml: Any, nvml_uuid: str | None) -> dict[str, Any]:
    """Stop CodeCarbon and expose its documented kWh fields."""
    result: dict[str, Any] = {
        "status": "unavailable",
        "version": getattr(codecarbon, "__version__", None),
        "scope": "CodeCarbon tracked process/system energy; not equivalent to NVML GPU-only energy.",
    }
    if tracker is None:
        return result
    try:
        emissions_kg = tracker.stop()
        data = dataclasses.asdict(tracker.final_emissions_data)
        selected = data.get("gpu_ids")
        gpu_index = int(selected[0]) if isinstance(selected, (list, tuple)) and len(selected) == 1 else None
        result.update(
            {
                "status": "measured",
                "gpu_energy_kwh": _numeric(data.get("gpu_energy")),
                "cpu_energy_kwh": _numeric(data.get("cpu_energy")),
                "ram_energy_kwh": _numeric(data.get("ram_energy")),
                "total_energy_kwh": _numeric(data.get("energy_consumed")),
                "native_emissions_kg": _numeric(emissions_kg),
                "duration_seconds": _numeric(data.get("duration")),
                "native_wue": _numeric(data.get("wue")),
                "native_water_liters": _numeric(data.get("water_consumed")),
                "tracking_mode": data.get("tracking_mode"),
                "gpu_identity": _codecarbon_identity(pynvml, gpu_index, nvml_uuid),
            }
        )
    except Exception as error:
        result.update({"status": "partial", "error": f"{type(error).__name__}: {error}"})
    return result


def _difference(value: float | None, reference: float | None) -> dict[str, float] | None:
    if value is None or reference is None or reference == 0:
        return None
    signed = value - reference
    return {
        "absolute_difference_kwh": abs(signed),
        "signed_difference_kwh": signed,
        "percent_difference_vs_nvml": signed / reference * 100.0,
    }


def _telemetry_versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("codecarbon", "nvidia-ml-py"):
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = None
    return values


def start_measurement(
    *,
    device_index: int | None = None,
    carbon_profile: str | None = None,
    carbon_intensity_g_per_kwh: float | None = None,
) -> dict[str, Any]:
    """Prepare trackers, then start the inference-only measurement boundary."""
    carbon = carbon_provenance(carbon_profile=carbon_profile, carbon_intensity_g_per_kwh=carbon_intensity_g_per_kwh)
    torch = _optional_module("torch", "campaign-reproduction")
    pynvml = _optional_module("pynvml", "telemetry")
    codecarbon = _optional_module("codecarbon", "telemetry")
    warnings: list[str] = []
    identity: dict[str, Any] = {
        "status": "unavailable",
        "cuda_index": None,
        "cuda_uuid": None,
        "index": None,
        "uuid": None,
        "name": None,
    }
    nvml_handle: Any | None = None
    try:
        identity, nvml_handle = _nvml_device(torch, pynvml, device_index)
    except Exception as error:
        warnings.append(f"NVML unavailable: {type(error).__name__}: {error}")
    if identity.get("mapping_warning"):
        warnings.append(str(identity["mapping_warning"]))
    country = "CAN" if carbon_profile == CAMPAIGN_CARBON_PROFILE else None
    region = "ontario" if carbon_profile == CAMPAIGN_CARBON_PROFILE else None
    tracker: Any | None = None
    try:
        tracker = _codecarbon_tracker(codecarbon, int(identity.get("index") or 0), country, region)
        tracker.get_detected_hardware()
    except Exception as error:
        warnings.append(f"CodeCarbon preparation unavailable: {type(error).__name__}: {error}")
    _sync_cuda(torch, warnings)
    start_monotonic = time.monotonic()
    start_mj: float | None = None
    if nvml_handle is not None:
        try:
            start_mj = _numeric(pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml_handle))
        except Exception as error:
            warnings.append(f"NVML start counter unavailable: {type(error).__name__}: {error}")
    if tracker is not None:
        try:
            tracker.start()
        except Exception as error:
            warnings.append(f"CodeCarbon start unavailable: {type(error).__name__}: {error}")
            tracker = None
    occupancy_start = (
        _occupancy(pynvml, nvml_handle)
        if nvml_handle is not None
        else {
            "status": "unavailable",
            "active_compute_process_count": None,
            "other_compute_process_count": None,
            "competing_process_detected": None,
        }
    )
    if occupancy_start["competing_process_detected"] is True:
        warnings.append(
            "Other GPU compute processes were present at measurement start; energy attribution may be contaminated."
        )
    return {
        "_torch": torch,
        "_pynvml": pynvml,
        "_nvml_handle": nvml_handle,
        "_tracker": tracker,
        "_codecarbon": codecarbon,
        "started_monotonic": start_monotonic,
        "start_mj": start_mj,
        "identity": identity,
        "occupancy_start": occupancy_start,
        "carbon": carbon,
        "warnings": warnings,
    }


def stop_measurement(handle: dict[str, Any], *, evaluated_items: int | None) -> dict[str, Any]:
    """Stop telemetry after prediction generation and build a public-safe artifact."""
    warnings: list[str] = list(handle["warnings"])
    pynvml = handle["_pynvml"]
    nvml_handle = handle["_nvml_handle"]
    _sync_cuda(handle["_torch"], warnings)
    codecarbon = _extract_codecarbon(handle["_tracker"], handle["_codecarbon"], pynvml, handle["identity"].get("uuid"))
    stopped_monotonic = time.monotonic()
    end_mj: float | None = None
    if nvml_handle is not None:
        try:
            end_mj = _numeric(pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml_handle))
        except Exception as error:
            warnings.append(f"NVML end counter unavailable: {type(error).__name__}: {error}")
    duration_seconds = stopped_monotonic - float(handle["started_monotonic"])
    nvml = {
        **handle["identity"],
        **nvml_energy(handle["start_mj"], end_mj, duration_seconds),
    }
    occupancy_stop = (
        _occupancy(pynvml, nvml_handle)
        if nvml_handle is not None
        else {
            "status": "unavailable",
            "active_compute_process_count": None,
            "other_compute_process_count": None,
            "competing_process_detected": None,
        }
    )
    if occupancy_stop["competing_process_detected"] is True:
        warnings.append(
            "Other GPU compute processes were present at measurement stop; energy attribution may be contaminated."
        )
    if nvml_handle is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception as error:
            warnings.append(f"NVML shutdown failed: {type(error).__name__}: {error}")
    nvml_kwh = _numeric(nvml.get("kwh"))
    codecarbon_gpu_kwh = _numeric(codecarbon.get("gpu_energy_kwh"))
    comparison = _difference(codecarbon_gpu_kwh, nvml_kwh)
    codecarbon_identity = codecarbon.get("gpu_identity", {})
    cuda_matches_nvml = (
        handle["identity"].get("cuda_uuid") == handle["identity"].get("uuid")
        if handle["identity"].get("cuda_uuid") and handle["identity"].get("uuid")
        else None
    )
    if codecarbon_identity.get("same_physical_gpu") is False:
        warnings.append("CodeCarbon GPU identity differs from the NVML-mapped CUDA device.")
    if cuda_matches_nvml is False:
        warnings.append("CUDA and NVML GPU identities differ.")
    carbon = handle["carbon"]
    intensity = _numeric(carbon["intensity_g_per_kwh"])
    primary_water = water_estimate(
        _numeric(codecarbon.get("total_energy_kwh")),
        evaluated_items=evaluated_items,
        scope="inference_only; CodeCarbon tracked total-energy scope (GPU, CPU, RAM, and its treatment).",
        energy_basis="codecarbon_total_energy_kwh",
    )
    secondary_water = water_estimate(
        nvml_kwh,
        evaluated_items=evaluated_items,
        scope="inference_only; GPU-attributed energy only.",
        energy_basis="nvml_gpu_energy_kwh",
    )
    return {
        "schema_version": "public-footprint-v1",
        "measurement_status": (
            "measured"
            if nvml["status"] == "measured"
            else "partial"
            if codecarbon["status"] == "measured"
            else "unavailable"
        ),
        "boundary": "inference_only",
        "boundary_scope": "model already loaded; prediction generation only; excludes loading, data preparation, metrics, serialization, and judge execution.",
        "evaluated_items": evaluated_items,
        "primary_measurement": "nvml_total_energy_counter",
        "gpu": {
            "cuda_index": handle["identity"].get("cuda_index"),
            "nvml_index": handle["identity"].get("index"),
            "name": handle["identity"].get("name"),
            "uuid": handle["identity"].get("uuid"),
        },
        "nvml": nvml,
        "codecarbon": codecarbon,
        "energy_comparison": {
            "nvml_gpu_kwh": nvml_kwh,
            "codecarbon_gpu_kwh": codecarbon_gpu_kwh,
            "codecarbon_gpu_minus_nvml": comparison,
            "scope_note": "CodeCarbon total energy is broader than NVML GPU-attributed energy.",
        },
        "carbon": {
            **carbon,
            "energy_basis": "nvml_gpu_energy_kwh",
            "scope": "GPU-attributed operational CO2e",
            "gpu_attributed_operational_co2e_kg": carbon_kg(nvml_kwh, intensity),
            "codecarbon_native_emissions_kg": codecarbon.get("native_emissions_kg"),
            "codecarbon_scope_note": "CodeCarbon native emissions follow CodeCarbon's tracked-process/system scope.",
        },
        "water": {
            "quality": "ESTIMATED",
            "primary_codecarbon_total": primary_water,
            "secondary_nvml_gpu": secondary_water,
            "method": "DIA-style estimated water from tracked energy and total WUE range.",
            "native_codecarbon_context": {
                "wue": codecarbon.get("native_wue"),
                "water_liters": codecarbon.get("native_water_liters"),
                "warning": "CodeCarbon WUE/water zero or absence is not evidence of zero physical water use.",
            },
        },
        "measurement_quality": {
            "cuda_matches_nvml": cuda_matches_nvml,
            "codecarbon_matches_nvml": codecarbon_identity.get("same_physical_gpu"),
            "occupancy_start": handle["occupancy_start"],
            "occupancy_stop": occupancy_stop,
            "competing_process_detected": (
                True
                if handle["occupancy_start"].get("competing_process_detected") is True
                or occupancy_stop.get("competing_process_detected") is True
                else False
                if handle["occupancy_start"].get("competing_process_detected") is False
                and occupancy_stop.get("competing_process_detected") is False
                else None
            ),
            "warnings": warnings,
        },
        "software": _telemetry_versions(),
    }
