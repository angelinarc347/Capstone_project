# Edge–Fog Digital Twin for Traffic Management

A three-tier digital twin of a real urban freeway network, built on real
detector data, with every architectural claim measured rather than asserted.

```
python run.py              # full pipeline; runtime depends on the host CPU
python build_dashboard.py  # writes dashboard.html
python -m tests.test_features
```

## The data

**METR-LA / los-loop**: 207 inductive loop detectors on the Los Angeles
freeway network, 5-minute mean speeds in mph, 2016 consecutive intervals
(7 days, March 2012), plus a 207×207 weighted adjacency matrix built from
road-network distance between detectors.

Downloaded in `data/` from the T-GCN repository:

```
https://raw.githubusercontent.com/lehaifeng/T-GCN/master/T-GCN/T-GCN-PyTorch/data/los_speed.csv
https://raw.githubusercontent.com/lehaifeng/T-GCN/master/T-GCN/T-GCN-PyTorch/data/los_adj.csv
```

The adjacency matrix is what makes this a *graph* twin rather than 207
independent sensors, and it is what the fog tier exists to exploit.

First 70% of the week trains, the remaining 30% (605 intervals ≈ 50 hours)
is held out for every number reported below.

## Architecture

```
                        ┌─────────────────────────────┐
                        │  CLOUD  — city twin          │
                        │  full 207-node replica       │
                        │  30-min forecast             │
                        │  what-if scenarios           │
                        │  inter-zone state sync       │
                        └──────────┬──────────────────┘
                          45 ms WAN │ zone summaries
              ┌────────────┬────────┴───────┬────────────┐
        ┌─────┴─────┐┌─────┴─────┐    ┌─────┴─────┐
        │   FOG 1   ││   FOG 2   │ …  │   FOG 8   │  one per road zone
        │ sub-twin  ││ sub-twin  │    │ sub-twin  │  5 & 15-min forecast
        │ corridor  ││ corridor  │    │ corridor  │  congestion alarms
        │ green     ││ budget    │    │ budget    │  green-time budget
        └─────┬─────┘└───────────┘    └───────────┘
      8 ms LAN │ event-triggered reports
    ┌──────────┼──────────┬───────────┐
 ┌──┴──┐   ┌──┴──┐    ┌──┴──┐    ┌──┴──┐
 │EDGE │   │EDGE │    │EDGE │ …  │EDGE │  one per detector (207)
 │ KF  │   │ KF  │    │ KF  │    │ KF  │  denoise, fault-detect,
 └──┬──┘   └──┬──┘    └──┬──┘    └──┬──┘  1-step forecast, local split
    │ raw 5-min speed readings (noisy, with dropouts and spikes)
```

Fog zones are formed by **spectral clustering on the road adjacency**, so
physically connected corridors land in the same fog node — which is the
precondition for coordinating them.

### What each tier does

| Tier | Model | Job | Why it has to live there |
|---|---|---|---|
| Edge | 2-state Kalman filter (level + damped trend) | denoise, reject gross outliers, diagnose stuck detectors, 5-min local forecast | needs every reading at full rate; must keep working when the network doesn't |
| Fog | Gradient-boosted trees on graph-diffused features, per zone | 5 and 15-min forecast, congestion-onset alarms, corridor green-time budget | needs the *neighbouring roads*, which a detector cannot see; needs to be close enough to close a control loop |
| Cloud | Same model, city-wide, 30-min horizon | full replica, what-if simulation, keeping fog zones aware of each other | needs the whole graph; latency doesn't matter for planning |

## Method

**Speed → control → outcome.** The dataset gives speed only, so the chain is
explicit:

1. **Greenshields** relation turns speed into density and flow:
   `k = k_j(1 − v/v_f)`, `q = kv`
2. **Webster's split rule** picks the green ratio from *predicted* demand:
   `λ = y/(y + y_opp)`
3. **Webster's delay formula** charges the controller for that choice using
   *actual* demand.

This is what makes forecast accuracy cash out as a traffic outcome: a
controller acting on a stale or noisy speed picks the wrong split and pays
for it in seconds of delay.

**Raw stream.** The published dataset is already cleaned, so the simulation
injects sensor noise (σ = 1.5 mph), gross outliers (0.4%) and dropouts
(0.3%). Without this the edge filtering tier would have nothing to do.
Every architecture sees the identical feed.

**Event-triggered reporting.** An edge node transmits only when its estimate
drifts more than δ from what the fog node last heard, plus a heartbeat every
30 minutes. Silence is information: "nothing has changed by more than δ".

**Hierarchical fusion.** The edge filter is hard to beat on its own road at
five minutes — it sees every reading. The fog model is the only thing that
sees the neighbours. Neither dominates, so the control input is
`w·edge + (1−w)·fog`, with **w measured on a validation slice of the
training week**, not guessed. It comes out at w = 0.20.

**Model tuning.** Kalman process and measurement noise were grid-searched on
training data. The search surfaced a real tension: high process noise gives
the best *state estimate* (what the replica needs, 1.55 mph MAE) while low
process noise gives the best *forecast* (what control needs, 2.45 mph MAE).
The deployed setting is the balanced point, and the grid is in the write-up
rather than hidden.

## Results

The tables below preserve the project's original reported baseline snapshot.
The dashboard and `results.json` display values from the most recent
`python run.py` execution; small numerical differences can occur between
snapshots. The evaluation split and metric definitions have not been changed.

All figures on the held-out window.

### Forecast error (MAE, mph)

| Model | 5 min | 15 min | 30 min |
|---|---|---|---|
| Historical average | 5.01 | 5.02 | 5.03 |
| Persistence | 2.72 | 3.44 | 4.15 |
| Graph ridge (lightweight) | 2.88 | 3.53 | 4.02 |
| **Graph GBT (deployed)** | **2.66** | **3.28** | **3.75** |

Persistence is a famously strong baseline at five minutes and a weak one at
thirty. The gap the graph model opens up grows with horizon — 2% better at
5 minutes, 10% at 30 — which is exactly the shape you would expect if the
spatial information is doing real work.

### Congestion onset, 15 minutes ahead

Of the roads running above 42 mph right now, which drop below it within 15
minutes? 2,407 such events in the held-out window.

| | Precision | Recall | F1 |
|---|---|---|---|
| Assume no change | 0.000 | 0.000 | 0.000 |
| Detector alone (Kalman) | 0.264 | 0.132 | 0.176 |
| **Fog, with the road graph** | **0.446** | **0.242** | **0.314** |

**This is the result that justifies the fog tier**: +79% F1 over an isolated
detector. Onset is a spatial question — the queue arrives from a
neighbouring link — so a detector reasoning about itself is structurally
handicapped no matter how good its filter is. Persistence scores zero by
construction, which is the point.

### Architecture comparison

| | Delay/veh (s) | Control input MAE | p95 loop latency | Uplink (kB/h) | Uncontrolled intervals under outage |
|---|---|---|---|---|---|
| Cloud only | 15.83 | 3.20 | 184.7 ms | 58.1 | 39 |
| Edge only | 15.65 | 2.87 | 3.8 ms | 9.6 | 0 |
| **Edge + fog + cloud** | **15.40** | **2.86** | 26.1 ms | 23.3 | **0** |

All three land on nearly the same signal splits — the delay spread is under
3%. **Reporting this honestly matters more than inflating it.** At a
five-minute control horizon, on freeway data, the choice of architecture is
not primarily an accuracy story. What it is:

- **7× faster control loop** than routing every decision through the cloud
- **60% less uplink traffic**, with the replica still within 4.0 mph RMSE
- **degrades gracefully**: during simulated backhaul outages (5% of
  intervals, in bursts), the centralised controller reverts to fixed timing
  for 39 intervals; the tiered twin keeps running on the local link
- and the onset detection above, which the edge tier simply cannot do alone

### Bandwidth vs. replica fidelity

| δ (mph) | Transmit duty cycle | Replica RMSE | Delay/veh |
|---|---|---|---|
| 0.5 | 68.0% | 3.77 | 15.43 s |
| 1.0 | 46.9% | 3.81 | 15.42 s |
| **2.0** | **29.8%** | **3.97** | **15.40 s** |
| 4.0 | 21.5% | 4.24 | 15.41 s |
| 8.0 | 19.5% | 4.48 | 15.45 s |

Replica error is **filter-limited, not trigger-limited** — going from δ=8 to
δ=0.5 costs 3.5× the bandwidth and recovers only 0.7 mph. δ=2 is the knee.
The floor at ~20% is the heartbeat plus fault reports.

### What-if

Forcing a lane-blocking incident onto the slowest detector and its three
closest links, projected 30 minutes out: 20 roads affected, **16 of them
spillover onto links the incident never touched**, 8.3 mph mean speed loss,
network travel time +1.9%.

This is the capability that separates a twin from a dashboard — the replica
can be driven into a state the real network is not in.

## Layout

```
src/config.py     every tunable in one dataclass
src/data.py       loading, graph construction, spectral zone partition
src/models.py     Kalman filter, graph ridge/GBT, shared feature assembly
src/traffic.py    Greenshields, Webster delay, adaptive split
src/infra.py      message bus with per-hop latency and byte accounting
src/nodes.py      EdgeNode, FogNode, CloudTwin
src/simulate.py   orchestrator, three architectures, all experiments
build_dashboard.py builds the offline interactive dashboard and forecast payload
dashboard_template.html responsive dashboard UI and client-side interactions
tests/            feature-parity test (training path == online path)
```

`tests/test_features.py` asserts that the features built during training and
the features the fog node builds at run time from its *reconstructed*
replica are identical. If those drift, the twin degrades silently — it keeps
returning confident numbers computed from columns that no longer mean what
the model was trained on. Worth a test.

## Interactive dashboard

The dashboard is a self-contained HTML file generated from `results.json`
and the included METR-LA data. It does not need Flask, a separate web server,
or a third-party JavaScript package. Dashboard generation also calculates
held-out per-detector GraphGBT forecasts using the same training split,
configuration, and model factory as the simulation.

### Install and run

Use Python 3.10 or newer. From the project folder in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m tests.test_features
python run.py
python build_dashboard.py
```

The dashboard is written to `dashboard.html` beside the scripts. Open that
file directly in a current browser; no local server is needed. If an included
virtual environment is not available, create one with
`python -m venv venv` before activating it. The simulation and build commands
resolve dataset and output paths relative to the project files, so they can
also be launched from another working directory.

### Dashboard sections

- **Overview:** held-out mean speed, detector count, congestion state,
  forecasting MAE, measured simulated bandwidth/latency/replica error,
  alert samples, trend chart and speed heatmap.
- **Road network:** interactive nodes and edges from the supplied adjacency
  matrix, with detector/zone/condition filters, neighbor highlighting and
  zoom controls. The graph is deliberately schematic: detector coordinates
  are not provided and none are invented.
- **Edge · Fog · Cloud:** describes the work done at each tier and reports
  communication, control-loop and replica-fidelity results from the
  simulation.
- **Forecast analytics:** select a detector, zone, horizon and test-window
  range; compare the actual target with its stored held-out GraphGBT
  prediction and calculate detector-level MAE/RMSE. Network-wide model
  comparison, fog-zone pooled errors, and congestion-onset
  precision/recall/F1 use the original simulation outputs.
- **What-if lab:** displays the original trained-model 30-minute scenario
  and offers an interactive reduced-order counterfactual. The interactive
  run reduces the selected detector's latest observed speed and diffuses a
  weighted impact through adjacency links. It reports affected links and a
  travel-time proxy; it does **not** retrain or re-run the forecast model.
- **Architecture lab:** compares cloud-only, edge-only and hierarchical
  results, the existing bandwidth/fidelity sweep and the simulated
  backhaul-outage test.
- **Sensor health:** screens the synthetic noisy feed for injected missing
  readings, candidate spikes and consecutive repeated readings, and displays
  the simulated outage results. These are not diagnoses of actual detector
  hardware.
- **Reports & export:** download filtered detector/horizon CSV, the original
  simulation results as JSON, or a concise text experiment report.

For a demonstration screenshot, generate and open `dashboard.html`, choose
the section to present, and capture the browser window at desktop width.
Browser zoom and window size can be adjusted for a narrower tablet/mobile
view; graph node positions remain schematic at every size.

The dashboard labels archived forecasts as historical, not real-time. Sensor
faults and communication outages are simulated. Traffic delay is a
Greenshields/Webster proxy, not a microsimulation or observed trip-time
measurement. The interactive graph-diffusion incident tool is intentionally
separate from the saved model-driven CloudTwin counterfactual; duration,
arbitrary sensor-noise and configurable outage runs are not implemented in
the offline dashboard.

### Troubleshooting

- **Missing `results.json`:** run `python run.py` successfully before
  `python build_dashboard.py`.
- **Missing Python packages:** activate the project environment and run
  `python -m pip install -r requirements.txt`.
- **Browser shows a build-data warning:** regenerate the dashboard after
  running the simulation; do not open `dashboard_template.html` directly.
- **Blank chart after resizing:** reload the generated HTML in a current
  browser. All chart data is embedded and the page requires no network access.

## Limitations, stated plainly

- **Delay is a physics proxy, not a microsimulation.** Webster's formula on
  Greenshields-derived flow is defensible and standard, but SUMO or VISSIM
  would model queues, turning movements and platoon dynamics that this does
  not. The delay numbers should be read as relative, not absolute.
- **This is freeway data, not signalised arterials.** METR-LA detectors sit
  on freeways, which have no traffic signals. The signal-control layer is
  applied as if each detector governed an approach. The forecasting and
  architecture results stand on their own; the control results are
  illustrative.
- **The opposing approach is synthetic.** Cross-street demand is set at half
  the detector's median flow, since the dataset has no such measurement.
- **Everything is one process.** Latency and bandwidth are charged from a
  model, not measured over real hardware. The numbers are consistent and
  comparable across architectures, but they are not a field measurement.
- **The fusion weight sees the training week.** It is calibrated on a
  validation slice of training, but the underlying models were fit on the
  full training set, so that single scalar has mild leakage.

## Where to take it next

1. **Replace the proxy with SUMO.** Feed the predicted speeds into a real
   microsimulation of an arterial corridor and measure actual queue lengths.
2. **Swap in a spatio-temporal GNN** (DCRNN, Graph WaveNet) at the cloud
   tier and check whether the onset F1 moves — the feature pipeline and
   evaluation harness are already in place, so only `models.py` changes.
3. **Run it on real hardware.** The edge tier is a Kalman filter and a
   threshold; it fits on an ESP32. Two Raspberry Pis as fog nodes over
   actual Wi-Fi would turn the modelled latencies into measured ones.
4. **Close the loop properly.** Right now control does not feed back into
   traffic state, because the dataset is a fixed recording. A simulator
   would let the twin's decisions actually change what happens next.

## Run the interactive web app

The web app serves the existing dashboard template and runs the existing
Python simulation in a background thread when a user selects **Run
Simulation**. The controls set the edge reporting threshold, injected sensor
noise, dropout rate, and congestion speed threshold. Once the run and forecast
payload are ready, the browser reloads with the newly calculated results.
Simulations use the included METR-LA data; no external data service is used.
Web runs are held in the app process and do not overwrite `results.json`.
The `dashboard.html` file remains an offline snapshot; use `app.py` to run
new simulations from the browser.
Opening `dashboard.html` with VS Code Live Server (often port 5500) only
serves the static snapshot; the simulation API is available at port 5000
while `python app.py` is running.
The full replay can take 10+ minutes on CPU-limited hosts. It runs as a
background job, so the initial HTTP request does not have to stay open for
the full calculation.

From the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. The existing `python run.py` and
`python build_dashboard.py` commands remain available for the offline results
file and self-contained HTML snapshot.

### Deploy on Render

1. Push this project folder (the folder containing `render.yaml`) to a GitHub
   repository.
2. In Render, choose **New > Blueprint**, connect the repository, and apply
   the included `render.yaml` configuration.
3. Render installs `requirements.txt`, starts the Flask app with Gunicorn,
   and provides the public service URL when deployment finishes.

The Blueprint uses Render's free web service plan for a student demo. Its
in-memory run status and latest results reset when the service restarts; each
new run recomputes results from the included dataset. CPU-limited plans may
take longer than the local run, so choose a larger plan if the simulation is
too slow or is stopped by resource limits. Render's official
[Flask deployment guide](https://render.com/docs/deploy-flask) documents the
same build and start commands used here.
