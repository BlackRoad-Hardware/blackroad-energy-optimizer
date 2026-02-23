"""Tests for blackroad-energy-optimizer."""
import pytest
from datetime import datetime, date, timedelta
from energy_optimizer import EnergyOptimizer, Device


@pytest.fixture
def opt(tmp_path):
    o = EnergyOptimizer(db_path=str(tmp_path / "test.db"), rate_per_kwh=0.12)
    fridge = Device("d1", "Fridge", "appliance", "kitchen", power_watts=150)
    ac     = Device("d2", "AC Unit", "hvac", "bedroom", power_watts=1500)
    o.add_device(fridge)
    o.add_device(ac)
    return o


def _inject_readings(opt, device_id, count=20, watts=150.0):
    base = datetime.utcnow() - timedelta(hours=count)
    for i in range(count):
        ts = (base + timedelta(hours=i)).isoformat()
        opt.track_consumption(device_id, watts, ts)


def test_track_consumption(opt):
    r = opt.track_consumption("d1", 150.0)
    assert r.watts == 150.0
    assert r.device_id == "d1"


def test_kwh_accumulates(opt):
    base = datetime.utcnow() - timedelta(hours=2)
    opt.track_consumption("d1", 1000.0, (base).isoformat())
    opt.track_consumption("d1", 1000.0, (base + timedelta(hours=2)).isoformat())
    r2 = opt._get_prev_reading("d1")
    assert r2["kwh_accumulated"] == pytest.approx(2.0, rel=0.01)


def test_daily_kwh(opt):
    _inject_readings(opt, "d1", count=24, watts=1000.0)
    kwh = opt.get_daily_kwh("d1")
    assert kwh > 0


def test_estimate_monthly_cost(opt):
    _inject_readings(opt, "d1", count=24, watts=1000.0)
    summary = opt.estimate_monthly_cost("d1", rate_per_kwh=0.10)
    assert summary.cost_usd > 0
    assert summary.total_kwh > 0
    assert summary.co2_kg > 0


def test_co2_equivalent(opt):
    co2 = opt.co2_equivalent(10.0)
    assert co2 == pytest.approx(3.86, rel=0.01)


def test_find_peak_hours(opt):
    _inject_readings(opt, "d1", count=48, watts=200.0)
    peaks = opt.find_peak_hours("d1")
    assert len(peaks) > 0
    assert "hour" in peaks[0]
    assert "avg_watts" in peaks[0]


def test_suggest_off_peak_schedule(opt):
    _inject_readings(opt, "d1", count=48, watts=200.0)
    suggestion = opt.suggest_off_peak_schedule("d1")
    assert "recommended_start_hour" in suggestion
    assert 0 <= suggestion["recommended_start_hour"] <= 23


def test_invalid_watts_raises(opt):
    with pytest.raises(ValueError, match="watts must be >= 0"):
        opt.track_consumption("d1", -50.0)


def test_unknown_device_raises(opt):
    with pytest.raises(ValueError, match="not found"):
        opt.track_consumption("unknown", 100.0)


def test_fleet_summary(opt):
    _inject_readings(opt, "d1", count=5, watts=150.0)
    _inject_readings(opt, "d2", count=5, watts=1500.0)
    summary = opt.get_fleet_summary()
    assert summary["device_count"] == 2
    assert summary["total_daily_kwh"] >= 0
