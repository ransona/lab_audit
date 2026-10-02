# Lab Audit

A standalone desktop tool for reporting approximate microscope usage from raw
Timeline files. It is intentionally outside `lab_pipeline` and never writes to
experiment data.

## What it counts

An experiment is included only when its raw experiment folder contains both:

- ScanImage TIFF data, either directly (standard microscope) or in `P*/R*`
  folders (mesoscope), and
- `<expID>_Timeline.mat`.

For each included experiment, duration is simply the final Timeline timestamp
minus the initial Timeline timestamp. It does not inspect microscope-frame
triggers or try to infer acquisition segments.

The owner is read from the PQE-written
`<expID>_experiment_metadata.json` file's `user` field. Missing values are
reported as `Unknown`.

## Run

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lab_pipeline
python /home/adamranson/code/labaudit/labaudit.py
```

Choose the raw repository (normally `/data/Remote_Repository`), click **Scan
timelines**, filter the resulting report, and use **Export filtered CSV** when
needed.

The app stores a local SQLite cache at `~/.local/share/labaudit/usage.sqlite`.
It shows the cached report immediately, then updates that cache from the raw
repository in the background on startup. **Scan / update database** starts the
same refresh manually.

The top panel includes a daily-hours plot. Choose a rolling window from the
week/month/three-month/six-month/year buttons, use the arrows to move by that
same interval, and select a PQE user to plot only that user's usage.

## Server resource monitor

The **Server resources** tab records and visualises aggregate CPU use plus
compute and memory use for each NVIDIA GPU every 20 seconds. It stores samples
in `~/.local/share/labaudit/server_monitor.sqlite`. The tab offers rolling
day/week/two-week/month/year windows, matching previous/next navigation, and
a custom start/end date-time range.

The installed `labmonitor.service` is a per-user systemd service. It starts at
boot (with user lingering enabled), restarts after failure, and continues after
logout:

```bash
systemctl --user status labmonitor.service
python /home/adamranson/code/labaudit/labmonitor.py
```
