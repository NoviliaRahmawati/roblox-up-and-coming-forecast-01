# Roblox Up-and-Coming Forecast

Deterministic Python 3 forecaster for make.fun's `roblox-up-and-coming-forecast-01` bounty.

## Run

The competition sandbox runs `python3 /bot/main.py`. The program reads `/input/task.json` and `/input/history.ndjson`, then writes `/output/forecast.json`. For local rehearsal:

```sh
MAKEFUN_INPUT_DIR=../roblox-up-and-coming-forecast-01-kit/input \
MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The implementation uses only Python's standard library and no network. It estimates rank momentum from timestamped historical observations, averages place probabilities across the scored horizon, and leaves probability unnamed for games absent from the history.
