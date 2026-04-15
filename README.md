# ndbc-xml

Converts OOI surface mooring buoy data (JSON) into NDBC XML submission files.
Runs hourly via cron for the four OOI Endurance Array Coastal Surface Moorings.

| OOI Site | NDBC ID | Location                                   |
|----------|---------|--------------------------------------------|
| CE02     | 46097   | Oregon Shelf (44.639°N, 124.304°W)         |
| CE04     | 46098   | Oregon Offshore (44.381°N, 124.956°W)      |
| CE07     | 46099   | Washington Shelf (46.986°N, 124.566°W)     |
| CE09     | 46100   | Washington Offshore (46.851°N, 124.972°W)  |

## Requirements

- Python 3.12+
- conda/miniforge (`ooi` environment)

Key dependencies: `pandas`, `numpy`, `gsw` (Gibbs SeaWater), `ppigrf` (IGRF-14).

## Installation

```bash
conda env create -f environment.yml
conda activate ooi
conda develop .
```

## Usage

### Command line

```bash
python -m ndbc_xml.ndbc <SITE> <DEPLOYMENT_DIR> [options]
```

**Positional arguments:**

| Argument | Description |
|----------|-------------|
| `SITE` | OOI site code: `CE02`, `CE04`, `CE07`, or `CE09` |
| `DEPLOYMENT_DIR` | Root deployment directory (e.g. `/data/parsed/ce02shsm/D00020`). METBK data are read from `<DEPLOYMENT_DIR>/buoy/metbk` and WAVSS from `<DEPLOYMENT_DIR>/buoy/wavss` |

**Optional arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--xml-out DIR` | `<DEPLOYMENT_DIR>/buoy/metbk/xml` | Output directory for XML files |
| `--state-file PATH` | `<xml-out>/<SITE>_position.json` | State file for incremental processing |
| `--alpha-date YYYY-MM-DD` | mid-point of processing window | Override date for IGRF declination calculation |
| `--sensor-depth METERS` | `1.15` | CTD sensor depth below surface (m, positive down) |
| `--reprocess` | off | Delete state file and reprocess all data from scratch |
| `--log-level LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--log-file PATH` | stderr only | Append log output to this file |

**Examples:**

```bash
# Normal hourly run
python -m ndbc_xml.ndbc CE02 /data/parsed/ce02shsm/D00020

# Specify output directory
python -m ndbc_xml.ndbc CE02 /data/parsed/ce02shsm/D00020 --xml-out /data/xml/CE02

# Force full reprocess after a bug fix
python -m ndbc_xml.ndbc CE02 /data/parsed/ce02shsm/D00020 --reprocess

# Back-process with a fixed declination date
python -m ndbc_xml.ndbc CE02 /data/parsed/ce02shsm/D00020 --alpha-date 2025-06-01
```

### Cron deployment

Use `utilities/ndbc.sh`, which sets the explicit Python path and `PYTHONPATH`
required for cron's minimal environment. Edit the path variables at the top of
the script to match the deployment system, then add one line per mooring:

```
0 * * * * /home/ooiuser/code/ndbc-xml/utilities/ndbc.sh CE02 /home/ooiuser/data/parsed/ce02shsm/D00020
0 * * * * /home/ooiuser/code/ndbc-xml/utilities/ndbc.sh CE04 /home/ooiuser/data/parsed/ce04ossm/D00019
0 * * * * /home/ooiuser/code/ndbc-xml/utilities/ndbc.sh CE07 /home/ooiuser/data/parsed/ce07shsm/D00020
0 * * * * /home/ooiuser/code/ndbc-xml/utilities/ndbc.sh CE09 /home/ooiuser/data/parsed/ce09ossm/D00020
```

Log files are written to `LOG_DIR` (configured in `ndbc.sh`), one per site.

### New deployment

Switch the `DEPLOYMENT_DIR` path (e.g. `D00020` → `D00021`). No state file
exists for the new directory, so the pipeline automatically starts from the
beginning of the available data.

## Data flow

```
OOI JSON files
  └─ ingest.py      load_metbk / load_wavss → raw DataFrames (Unix epoch seconds)
       └─ process.py   rotate_wind, rotate_wave_dir, calc_salinity, calc_rain_rate
            └─ process.py   bin_observations (10-min means)
                 └─ process.py   apply_qc (out-of-range → NaN)
                      └─ xml_writer.py   write_xml → NDBC XML file
                           └─ state.py   save_state (persist last_bin_end)
```

## Output format

Files are named `HH-DD-Mon-YYYY-<NDBC_ID>.xml` (e.g. `14-30-Mar-2026-46097.xml`).
Spans longer than one day (reprocess or first run) produce one file per UTC calendar day.

Each file begins with a WMO bulletin header required by NDBC:
```
SXML99 KWBC DDHHMM
<?xml version="1.0" encoding="ISO-8859-1"?>
<message>
  <station>46097</station>
  <date>03/30/2026 14:00:00</date>
  ...
</message>
```

Missing or QC-failed values are written as `-9999`.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success — XML file written |
| 1 | Error — see log output |
| 2 | No new data to process |

## Development

```bash
# Run all tests
pytest ndbc_xml/tests/test_ndbc_xml.py -v

# Run a single test class
pytest ndbc_xml/tests/test_ndbc_xml.py::TestBinObservations -v

# Run a single test
pytest ndbc_xml/tests/test_ndbc_xml.py::TestBinObservations::test_wind_speed_correct -v
```
