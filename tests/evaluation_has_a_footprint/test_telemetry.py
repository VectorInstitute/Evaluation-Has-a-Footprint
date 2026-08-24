"""CPU-only tests for portable telemetry arithmetic and lifecycle."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from evaluation_has_a_footprint import telemetry


@dataclasses.dataclass
class _EmissionsData:
    gpu_energy: float = 0.001
    cpu_energy: float = 0.0002
    ram_energy: float = 0.0001
    energy_consumed: float = 0.0013
    duration: float = 2.0
    wue: float = 0.0
    water_consumed: float = 0.0
    tracking_mode: str = "process"
    gpu_ids: list[int] = dataclasses.field(default_factory=lambda: [0])


class _FakeTorchCuda:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return 0

    def get_device_properties(self, index: int) -> Any:
        assert index == 0
        return type("Properties", (), {"uuid": "GPU-unit-test"})()

    def synchronize(self) -> None:
        self._events.append("sync")


class _FakeTorch:
    def __init__(self, events: list[str]) -> None:
        self.cuda = _FakeTorchCuda(events)


class _FakeNVML:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._energy = iter((1_000, 4_600))

    def __getattr__(self, name: str) -> Any:
        methods = {
            "nvmlInit": self._init,
            "nvmlShutdown": self._shutdown,
            "nvmlSystemGetNVMLVersion": self._version,
            "nvmlDeviceGetCount": self._count,
            "nvmlDeviceGetHandleByIndex": self._handle,
            "nvmlDeviceGetUUID": self._uuid,
            "nvmlDeviceGetName": self._name,
            "nvmlDeviceGetTotalEnergyConsumption": self._energy_counter,
            "nvmlDeviceGetComputeRunningProcesses_v3": self._processes,
        }
        try:
            return methods[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def _init(self) -> None:
        self._events.append("nvml.init")

    def _shutdown(self) -> None:
        self._events.append("nvml.shutdown")

    def _version(self) -> bytes:
        return b"test-nvml"

    def _count(self) -> int:
        return 1

    def _handle(self, index: int) -> str:
        assert index == 0
        return "gpu0"

    def _uuid(self, handle: str) -> bytes:
        assert handle == "gpu0"
        return b"GPU-unit-test"

    def _name(self, handle: str) -> bytes:
        assert handle == "gpu0"
        return b"Mock GPU"

    def _energy_counter(self, handle: str) -> int:
        assert handle == "gpu0"
        value = next(self._energy)
        self._events.append(f"energy:{value}")
        return value

    def _processes(self, handle: str) -> list[Any]:
        assert handle == "gpu0"
        return []


class _FakeTracker:
    def __init__(self, events: list[str], **kwargs: Any) -> None:
        self._events = events
        self.kwargs = kwargs
        self.final_emissions_data = _EmissionsData()

    def get_detected_hardware(self) -> dict[str, list[int]]:
        self._events.append("tracker.prepare")
        return {"gpu_ids": [0]}

    def start(self) -> None:
        self._events.append("tracker.start")

    def stop(self) -> float:
        self._events.append("tracker.stop")
        return 0.00007


class _FakeCodeCarbon:
    __version__ = "3.3.0"

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def __getattr__(self, name: str) -> Any:
        if name == "OfflineEmissionsTracker":
            return self._tracker
        raise AttributeError(name)

    def _tracker(self, **kwargs: Any) -> _FakeTracker:
        return _FakeTracker(self._events, **kwargs)


def _patched_measurement(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    events: list[str] = []
    modules = {
        "torch": _FakeTorch(events),
        "pynvml": _FakeNVML(events),
        "codecarbon": _FakeCodeCarbon(events),
    }
    monkeypatch.setattr(telemetry, "_optional_module", lambda name, _extra: modules[name])
    return events


def test_nvml_conversion_matches_campaign_arithmetic() -> None:
    result = telemetry.nvml_energy(242_331_092_432, 242_983_404_030)
    assert result["delta_mj"] == 652_311_598
    assert result["joules"] == pytest.approx(652_311.598)
    assert result["wh"] == pytest.approx(181.19766611111112)
    assert result["kwh"] == pytest.approx(0.1811976661111111)


@pytest.mark.parametrize(
    ("delta_mj", "expected_kwh"),
    [
        (652_311_598, 0.1811976661111111),
        (999_957_524, 0.2777659788888889),
        (1_625_573_975, 0.4515483263888889),
        (2_709_868_049, 0.7527411247222222),
    ],
)
def test_nvml_four_campaign_style_values(delta_mj: int, expected_kwh: float) -> None:
    assert telemetry.nvml_energy(0, delta_mj)["kwh"] == pytest.approx(expected_kwh)


def test_decreasing_nvml_counter_is_unavailable() -> None:
    assert telemetry.nvml_energy(10, 9)["status"] == "unavailable"


def test_unavailable_nvml_hardware_preserves_partial_codecarbon_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_measurement(monkeypatch)
    monkeypatch.setattr(telemetry, "_nvml_device", lambda *_args: (_ for _ in ()).throw(RuntimeError("no counter")))
    footprint = telemetry.stop_measurement(telemetry.start_measurement(), evaluated_items=1)
    assert footprint["nvml"]["status"] == "unavailable"
    assert footprint["codecarbon"]["status"] == "measured"
    assert footprint["measurement_status"] == "partial"
    assert any("NVML unavailable" in warning for warning in footprint["measurement_quality"]["warnings"])


def test_carbon_profiles_and_arithmetic() -> None:
    campaign = telemetry.carbon_provenance(carbon_profile="accepted-campaign", carbon_intensity_g_per_kwh=None)
    assert campaign["intensity_g_per_kwh"] == 59.0
    assert campaign["location_confirmation"] == "location-unconfirmed"
    assert telemetry.carbon_kg(0.1811976661111111, campaign["intensity_g_per_kwh"]) == pytest.approx(
        0.010690662300555554
    )
    caller = telemetry.carbon_provenance(carbon_profile=None, carbon_intensity_g_per_kwh=400.0)
    assert caller["source"] == "caller-provided"
    assert telemetry.carbon_kg(1.0, 400.0) == 0.4


def test_water_is_estimated_and_normalized() -> None:
    primary = telemetry.water_estimate(
        0.21630707721148806,
        evaluated_items=2_000,
        scope="total",
        energy_basis="codecarbon_total_energy_kwh",
    )
    assert primary["quality"] == "ESTIMATED"
    assert primary["low_liters"] == pytest.approx(0.3893527389806785)
    assert primary["high_liters"] == pytest.approx(0.8652283088459523)
    assert primary["per_item_low_liters"] == pytest.approx(0.00019467636949033925)
    assert primary["per_item_high_liters"] == pytest.approx(0.00043261415442297614)
    zero_items = telemetry.water_estimate(1.0, evaluated_items=0, scope="total", energy_basis="total")
    assert zero_items["per_item_low_liters"] is None
    assert zero_items["status"] == "estimated_per_item_unavailable"


def test_measurement_order_identity_quality_and_public_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _patched_measurement(monkeypatch)
    handle = telemetry.start_measurement(carbon_profile="accepted-campaign")
    footprint = telemetry.stop_measurement(handle, evaluated_items=2)
    assert (
        events.index("tracker.prepare")
        < events.index("sync")
        < events.index("energy:1000")
        < events.index("tracker.start")
    )
    assert (
        events.index("tracker.stop") < events.index("sync", events.index("tracker.stop")) < events.index("energy:4600")
    )
    assert footprint["boundary"] == "inference_only"
    assert footprint["nvml"]["kwh"] == pytest.approx(1e-6)
    assert footprint["codecarbon"]["gpu_energy_kwh"] == 0.001
    assert footprint["carbon"]["gpu_attributed_operational_co2e_kg"] == pytest.approx(59e-9)
    assert footprint["measurement_quality"]["cuda_matches_nvml"] is True
    assert footprint["measurement_quality"]["codecarbon_matches_nvml"] is True
    encoded = json.dumps(footprint)
    assert "current_pid" not in encoded
    assert "compute_pids" not in encoded
    assert "not evidence of zero physical water use" in footprint["water"]["native_codecarbon_context"]["warning"]


def test_identity_mismatch_and_contamination_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _patched_measurement(monkeypatch)
    monkeypatch.setattr(
        telemetry,
        "_codecarbon_identity",
        lambda *_args: {"status": "resolved", "same_physical_gpu": False},
    )
    monkeypatch.setattr(
        telemetry,
        "_occupancy",
        lambda *_args: {
            "status": "measured",
            "active_compute_process_count": 2,
            "other_compute_process_count": 1,
            "competing_process_detected": True,
        },
    )
    footprint = telemetry.stop_measurement(telemetry.start_measurement(), evaluated_items=1)
    warnings = footprint["measurement_quality"]["warnings"]
    assert footprint["measurement_quality"]["competing_process_detected"] is True
    assert any("identity differs" in warning for warning in warnings)
    assert any("contaminated" in warning for warning in warnings)
    assert events


def test_missing_telemetry_dependency_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_: str) -> Any:
        raise ModuleNotFoundError("missing")

    monkeypatch.setattr(telemetry.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="optional dependency 'pynvml'"):
        telemetry._optional_module("pynvml", "telemetry")
