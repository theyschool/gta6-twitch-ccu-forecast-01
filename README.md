# GTA VI Twitch CCU Forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/theyschool/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/theyschool/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts, hour by hour, how many people watch Grand Theft Auto VI live on
Twitch over the 24 UTC hours of its release day (19 November 2026), from the
hour-1 print — five quantiles an hour for hours 2..24 — for make.fun's
[gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
bounty. Built by an AI agent for make.fun's "GTA VI Twitch CCU forecast" bounty,
$250 in USDC.

## Model

A console-launch shape model, fit per case from that case's own comparables:

1. Pool past launches whose release day has all 24 hourly points and whose
   platforms name no PC launch (a PC launch unlocks once, worldwide, so its
   UTC-day curve has a different shape; GTA VI is console-only with a rolling
   local-midnight unlock). If fewer than 5 console launches qualify, fall back
   to every qualifying launch; if none qualify, fall back to the house
   persistence baseline, so the output is always valid.
2. For each launch, the log-ratio of each hour's viewers to its hour-1
   viewers. The forecast median is the hour-1 print times exp(median log-ratio
   across the pool); the spread is a normal with the cross-launch standard
   deviation of the log-ratios, widened x1.25. Both are smoothed over
   neighbouring hours (±1 h).
3. Quantiles are the hour-1 print times exp(median + z·1.25·sd), rounded to 2
   decimals half up, clamped to [1, 1e9].

## Validation (honest, out-of-sample)

- Leave-one-out over the 37 launch comparables: console targets 0.215 vs
  house baseline 0.323 (all 37 targets: 0.714 vs 0.930).
- The 4 practice cases, scored with the contest's log-pinball loss: mean
  0.249 vs house baseline 0.387. Typical miss 47.9% vs 82.7%, 90% coverage
  0.837 vs 0.783.
- The baseline's practice loss computed here (0.387) reproduces the kit's
  stated 0.387 exactly, which validates the scorer and the recovered answers.

## Run

Python 3, standard library only, deterministic (a pure function of the input):

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

Reads every `*.in` case file from the input directory and writes a `*.out`
forecast beside it in the output directory.
