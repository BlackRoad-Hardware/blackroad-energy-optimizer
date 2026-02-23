# blackroad-energy-optimizer

Part of [BlackRoad-Hardware](https://github.com/BlackRoad-Hardware) — IoT & hardware intelligence platform.

## Overview

| Repo | Description |
|------|-------------|
| blackroad-smart-home | Smart home controller: scenes, scheduling, device groups |
| blackroad-sensor-network | IoT sensor aggregator with Z-score anomaly detection |
| blackroad-automation-hub | Rules engine: triggers, conditions, actions |
| blackroad-energy-optimizer | Energy tracking, peak analysis, CO2 equivalent |
| blackroad-fleet-tracker | Fleet GPS tracking, geofencing, idle detection |

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python energy-optimizer.py   # runs demo
```

## Tests

```bash
pytest test_energy-optimizer.py -v
```

## Architecture

- Pure Python with SQLite persistence (WAL mode)
- Thread-safe with per-operation locks
- Self-initializing database on first run
- Dataclass-based domain model

## License

© BlackRoad OS, Inc. All rights reserved.
