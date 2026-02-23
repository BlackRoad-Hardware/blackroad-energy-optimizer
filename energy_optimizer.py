"""
blackroad-energy-optimizer — Energy Optimization
Production: consumption tracking, daily kWh, monthly cost estimation,
peak-hour analysis, off-peak scheduling suggestions, CO2 equivalent.
"""

from __future__ import annotations
import sqlite3
import json
import math
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

DB_PATH = "energy_optimizer.db"
_LOCK = threading.Lock()

# CO2 intensity (kg CO2 per kWh) — US average grid
DEFAULT_CO2_KG_PER_KWH = 0.386
DEFAULT_RATE_PER_KWH = 0.12  # USD


# ─────────────────────────── Dataclasses ────────────────────────────

@dataclass
class Device:
    id: str
    name: str
    type: str                    # appliance / hvac / lighting / ev_charger / server
    room: str
    power_watts: float           # rated power consumption
    standby_watts: float = 0.0
    smart: bool = True           # can be programmed
    active: bool = True
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def hourly_kwh(self) -> float:
        return self.power_watts / 1000.0


@dataclass
class EnergyReading:
    device_id: str
    watts: float
    kwh_accumulated: float
    timestamp: str
    source: str = "realtime"   # realtime / estimated / manual


@dataclass
class CostSummary:
    device_id: str
    period_start: str
    period_end: str
    total_kwh: float
    cost_usd: float
    co2_kg: float
    rate_per_kwh: float
    readings_count: int


# ─────────────────────────── Database ───────────────────────────────

def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    with _get_conn(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS devices (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            type            TEXT NOT NULL,
            room            TEXT NOT NULL,
            power_watts     REAL NOT NULL,
            standby_watts   REAL NOT NULL DEFAULT 0.0,
            smart           INTEGER NOT NULL DEFAULT 1,
            active          INTEGER NOT NULL DEFAULT 1,
            created_at      TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS readings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id       TEXT NOT NULL,
            watts           REAL NOT NULL,
            kwh_accumulated REAL NOT NULL DEFAULT 0.0,
            timestamp       TEXT NOT NULL,
            source          TEXT NOT NULL DEFAULT 'realtime',
            FOREIGN KEY(device_id) REFERENCES devices(id)
        );
        CREATE INDEX IF NOT EXISTS idx_readings_device_ts
            ON readings(device_id, timestamp);
        CREATE TABLE IF NOT EXISTS costs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id       TEXT NOT NULL,
            period_start    TEXT NOT NULL,
            period_end      TEXT NOT NULL,
            total_kwh       REAL NOT NULL,
            cost_usd        REAL NOT NULL,
            co2_kg          REAL NOT NULL,
            rate_per_kwh    REAL NOT NULL,
            readings_count  INTEGER NOT NULL,
            computed_at     TEXT NOT NULL,
            FOREIGN KEY(device_id) REFERENCES devices(id)
        );
        """)
    logger.info("energy_optimizer DB initialised at %s", db_path)


# ─────────────────────────── Optimizer ──────────────────────────────

class EnergyOptimizer:
    def __init__(self, db_path: str = DB_PATH,
                 rate_per_kwh: float = DEFAULT_RATE_PER_KWH,
                 co2_per_kwh: float = DEFAULT_CO2_KG_PER_KWH):
        self.db_path = db_path
        self.rate_per_kwh = rate_per_kwh
        self.co2_per_kwh = co2_per_kwh
        init_db(db_path)

    # ── Device CRUD ───────────────────────────────────────────────────

    def add_device(self, device: Device) -> Device:
        with _LOCK, _get_conn(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO devices VALUES (?,?,?,?,?,?,?,?,?)",
                (device.id, device.name, device.type, device.room,
                 device.power_watts, device.standby_watts,
                 int(device.smart), int(device.active), device.created_at)
            )
        return device

    def get_device(self, device_id: str) -> Optional[Device]:
        with _get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM devices WHERE id=?", (device_id,)
            ).fetchone()
        if not row:
            return None
        return Device(
            id=row["id"], name=row["name"], type=row["type"],
            room=row["room"], power_watts=row["power_watts"],
            standby_watts=row["standby_watts"],
            smart=bool(row["smart"]), active=bool(row["active"]),
            created_at=row["created_at"]
        )

    def list_devices(self, active_only: bool = True) -> List[Device]:
        q = "SELECT * FROM devices"
        if active_only:
            q += " WHERE active=1"
        with _get_conn(self.db_path) as conn:
            rows = conn.execute(q).fetchall()
        return [
            Device(id=r["id"], name=r["name"], type=r["type"],
                   room=r["room"], power_watts=r["power_watts"],
                   standby_watts=r["standby_watts"],
                   smart=bool(r["smart"]), active=bool(r["active"]),
                   created_at=r["created_at"])
            for r in rows
        ]

    # ── Tracking ──────────────────────────────────────────────────────

    def track_consumption(self, device_id: str, watts: float,
                          timestamp: Optional[str] = None) -> EnergyReading:
        if watts < 0:
            raise ValueError("watts must be >= 0")
        device = self.get_device(device_id)
        if not device:
            raise ValueError(f"Device {device_id!r} not found")
        ts = timestamp or datetime.utcnow().isoformat()

        # compute accumulated kWh from last reading
        prev = self._get_prev_reading(device_id)
        kwh_delta = 0.0
        if prev:
            prev_ts = datetime.fromisoformat(prev["timestamp"])
            cur_ts  = datetime.fromisoformat(ts)
            hours = max(0.0, (cur_ts - prev_ts).total_seconds() / 3600.0)
            avg_watts = (prev["watts"] + watts) / 2.0
            kwh_delta = avg_watts / 1000.0 * hours
            kwh_acc = prev["kwh_accumulated"] + kwh_delta
        else:
            kwh_acc = 0.0

        reading = EnergyReading(
            device_id=device_id, watts=watts,
            kwh_accumulated=kwh_acc, timestamp=ts
        )
        with _LOCK, _get_conn(self.db_path) as conn:
            conn.execute(
                "INSERT INTO readings (device_id, watts, kwh_accumulated, timestamp) "
                "VALUES (?,?,?,?)",
                (device_id, watts, kwh_acc, ts)
            )
        return reading

    def _get_prev_reading(self, device_id: str) -> Optional[Dict[str, Any]]:
        with _get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT watts, kwh_accumulated, timestamp FROM readings "
                "WHERE device_id=? ORDER BY timestamp DESC LIMIT 1",
                (device_id,)
            ).fetchone()
        return dict(row) if row else None

    # ── Analytics ─────────────────────────────────────────────────────

    def get_daily_kwh(self, device_id: str,
                      target_date: Optional[date] = None) -> float:
        d = target_date or date.today()
        day_start = datetime.combine(d, datetime.min.time()).isoformat()
        day_end   = datetime.combine(d + timedelta(days=1),
                                     datetime.min.time()).isoformat()
        with _get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT watts, timestamp FROM readings "
                "WHERE device_id=? AND timestamp>=? AND timestamp<? "
                "ORDER BY timestamp ASC",
                (device_id, day_start, day_end)
            ).fetchall()
        if len(rows) < 2:
            if len(rows) == 1:
                # single reading: estimate
                dev = self.get_device(device_id)
                return (dev.power_watts / 1000.0) if dev else 0.0
            return 0.0
        kwh = 0.0
        for i in range(1, len(rows)):
            t1 = datetime.fromisoformat(rows[i-1]["timestamp"])
            t2 = datetime.fromisoformat(rows[i]["timestamp"])
            hours = (t2 - t1).total_seconds() / 3600.0
            avg_w = (rows[i-1]["watts"] + rows[i]["watts"]) / 2.0
            kwh += avg_w / 1000.0 * hours
        return round(kwh, 6)

    def estimate_monthly_cost(self, device_id: str,
                              rate_per_kwh: Optional[float] = None,
                              days_in_month: int = 30) -> CostSummary:
        rate = rate_per_kwh if rate_per_kwh is not None else self.rate_per_kwh
        device = self.get_device(device_id)
        if not device:
            raise ValueError(f"Device {device_id!r} not found")

        today = date.today()
        # get last 7 days of data for rolling average
        kwh_days = []
        for offset in range(7):
            d = today - timedelta(days=offset)
            kw = self.get_daily_kwh(device_id, d)
            if kw > 0:
                kwh_days.append(kw)

        if kwh_days:
            avg_daily_kwh = sum(kwh_days) / len(kwh_days)
        else:
            # fallback: use rated power × 24h
            avg_daily_kwh = device.power_watts / 1000.0 * 24.0

        monthly_kwh = avg_daily_kwh * days_in_month
        cost = monthly_kwh * rate
        co2 = self.co2_equivalent(monthly_kwh)

        period_start = today.isoformat()
        period_end = (today + timedelta(days=days_in_month)).isoformat()
        summary = CostSummary(
            device_id=device_id,
            period_start=period_start,
            period_end=period_end,
            total_kwh=round(monthly_kwh, 4),
            cost_usd=round(cost, 4),
            co2_kg=round(co2, 4),
            rate_per_kwh=rate,
            readings_count=len(kwh_days)
        )
        with _LOCK, _get_conn(self.db_path) as conn:
            conn.execute(
                "INSERT INTO costs "
                "(device_id, period_start, period_end, total_kwh, cost_usd, "
                "co2_kg, rate_per_kwh, readings_count, computed_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (device_id, period_start, period_end, summary.total_kwh,
                 summary.cost_usd, summary.co2_kg, rate,
                 summary.readings_count, datetime.utcnow().isoformat())
            )
        return summary

    def find_peak_hours(self, device_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Return average consumption per hour-of-day bucket over last N days."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with _get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT watts, timestamp FROM readings "
                "WHERE device_id=? AND timestamp>=? ORDER BY timestamp ASC",
                (device_id, since)
            ).fetchall()

        hourly: Dict[int, List[float]] = {h: [] for h in range(24)}
        for row in rows:
            try:
                hour = datetime.fromisoformat(row["timestamp"]).hour
                hourly[hour].append(row["watts"])
            except Exception:
                continue

        result = []
        for hour in range(24):
            vals = hourly[hour]
            if vals:
                avg = sum(vals) / len(vals)
                result.append({
                    "hour": hour,
                    "avg_watts": round(avg, 2),
                    "sample_count": len(vals)
                })
        result.sort(key=lambda x: x["avg_watts"], reverse=True)
        return result

    def suggest_off_peak_schedule(self, device_id: str,
                                  days: int = 7) -> Dict[str, Any]:
        """Identify the cheapest 8-hour window for device operation."""
        peak_hours = self.find_peak_hours(device_id, days=days)
        if not peak_hours:
            return {"device_id": device_id, "suggestion": "no_data"}

        hourly_map = {h["hour"]: h["avg_watts"] for h in peak_hours}
        # fill missing hours with zero
        for h in range(24):
            if h not in hourly_map:
                hourly_map[h] = 0.0

        # sliding 8-hour window sum
        window = 8
        best_start = 0
        best_sum = float("inf")
        for start in range(24):
            total = sum(hourly_map[(start + i) % 24] for i in range(window))
            if total < best_sum:
                best_sum = total
                best_start = start

        hours_in_window = [(best_start + i) % 24 for i in range(window)]
        return {
            "device_id": device_id,
            "recommended_start_hour": best_start,
            "recommended_end_hour": (best_start + window) % 24,
            "window_hours": hours_in_window,
            "avg_consumption_in_window": round(best_sum / window, 2),
            "rationale": "lowest average consumption window over last {} days".format(days)
        }

    def co2_equivalent(self, kwh: float,
                       co2_per_kwh: Optional[float] = None) -> float:
        """Convert kWh to kg CO2 equivalent."""
        factor = co2_per_kwh if co2_per_kwh is not None else self.co2_per_kwh
        return round(kwh * factor, 6)

    def get_fleet_summary(self) -> Dict[str, Any]:
        devices = self.list_devices()
        summaries = []
        total_kwh = 0.0
        for d in devices:
            daily = self.get_daily_kwh(d.id)
            total_kwh += daily
            summaries.append({
                "device_id": d.id, "name": d.name,
                "daily_kwh": round(daily, 4),
                "daily_cost_usd": round(daily * self.rate_per_kwh, 4)
            })
        return {
            "device_count": len(devices),
            "total_daily_kwh": round(total_kwh, 4),
            "total_daily_cost_usd": round(total_kwh * self.rate_per_kwh, 4),
            "total_daily_co2_kg": round(self.co2_equivalent(total_kwh), 4),
            "devices": sorted(summaries, key=lambda x: x["daily_kwh"], reverse=True)
        }


def demo() -> None:
    import os, random
    os.remove(DB_PATH) if os.path.exists(DB_PATH) else None
    opt = EnergyOptimizer()

    fridge = Device("d-fridge", "Refrigerator", "appliance", "kitchen", power_watts=150)
    washer = Device("d-washer", "Washing Machine", "appliance", "laundry", power_watts=500)
    opt.add_device(fridge)
    opt.add_device(washer)

    base = datetime.utcnow() - timedelta(hours=48)
    for i in range(200):
        ts = (base + timedelta(minutes=i * 14)).isoformat()
        opt.track_consumption("d-fridge", 150 + random.gauss(0, 10), ts)
    for i in range(40):
        ts = (base + timedelta(minutes=i * 70)).isoformat()
        opt.track_consumption("d-washer",
                              500 if 8 <= (base + timedelta(minutes=i*70)).hour <= 11 else 0,
                              ts)

    daily = opt.get_daily_kwh("d-fridge")
    print(f"Fridge daily kWh: {daily}")

    summary = opt.estimate_monthly_cost("d-fridge")
    print(f"Fridge monthly cost: ${summary.cost_usd} / {summary.co2_kg} kg CO2")

    peaks = opt.find_peak_hours("d-washer")
    print(f"Washer top peak hour: {peaks[0] if peaks else 'N/A'}")

    schedule = opt.suggest_off_peak_schedule("d-washer")
    print(f"Off-peak suggestion: start={schedule['recommended_start_hour']}h")

    print(opt.get_fleet_summary())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
