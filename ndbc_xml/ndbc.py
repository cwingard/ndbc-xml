"""
Command-line entry point for the NDBC XML generation pipeline.

Intended to be invoked from a crontab entry, one line per mooring::

    # /etc/cron.d/ndbc_xml  — runs hourly
    0 * * * * python -m ndbc_xml.main CE02 /home/ooiuser/data/parsed/ce02shsm/D00020
    0 * * * * python -m ndbc_xml.main CE04 /home/ooiuser/data/parsed/ce04ossm/D00019
    0 * * * * python -m ndbc_xml.main CE07 /home/ooiuser/data/parsed/ce07shsm/D00020
    0 * * * * python -m ndbc_xml.main CE09 /home/ooiuser/data/parsed/ce09ossm/D00020

A new deployment is signaled implicitly by passing a new
DEPLOYMENT_DIR (e.g. ``D00021`` instead of ``D00020``). No state file
will exist for that directory, so the pipeline automatically starts
from the beginning of the available data.

Positional arguments
--------------------
site
    OOI site code: CE02, CE04, CE07, or CE09.
deployment_dir
    Root directory for this deployment, e.g.
    ``/home/ooiuser/data/parsed/ce02shsm/D00020``.
    METBK data are read from ``<deployment_dir>/buoy/metbk`` and
    WAVSS data from ``<deployment_dir>/buoy/wavss``.

Optional arguments
------------------
--xml-out DIR
    Directory for output XML files. Default:
    ``<deployment_dir>/buoy/metbk/xml``.
--state-file PATH
    JSON file for run-continuity tracking. Default:
    ``<deployment_dir>/buoy/metbk/xml/<SITE>_position.json``.
--alpha-date DATE
    ISO-8601 date (YYYY-MM-DD) used for the IGRF magnetic
    declination calculation. Defaults to the mid-point of the
    processing window (auto). Override when back-processing a
    deployment period that differs significantly from today.
--sensor-depth METERS
    Depth of the CTD/temperature sensor below the surface in
    meters (positive down). Used for GSW pressure calculation.
    Default: 1.25.
--reprocess
    Delete the state file and reprocess all available data from
    scratch. Use after a bug fix or data correction when the
    existing XML output needs to be regenerated.
--log-level LEVEL
    Logging verbosity: DEBUG, INFO, WARNING, ERROR.
    Default: INFO.
--log-file PATH
    Append log output to this file in addition to stderr.
    Useful for cron jobs where stdout is discarded.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import StationConfig, SITES
from .pipeline import run_station
from .state import clear_state

def _build_config(args: argparse.Namespace) -> StationConfig:
    """Construct a :class:`StationConfig` from parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from :func:`_parse_args`.

    Returns
    -------
    StationConfig
        Fully populated configuration ready for :func:`run_station`.
    """
    site = args.site.upper()
    meta = SITES[site]
    deploy = Path(args.deployment_dir)

    return StationConfig(
        site=site,
        ndbc_id=meta["ndbc_id"],
        latitude=meta["latitude"],
        longitude=meta["longitude"],
        metbk_dir=deploy / "buoy" / "metbk",
        wavss_dir=deploy / "buoy" / "wavss",
        xml_out_dir=Path(args.xml_out) if args.xml_out else deploy / "buoy" / "metbk" / "xml",
        state_file=(
            Path(args.state_file)
            if args.state_file
            else deploy / "buoy" / "metbk" / "xml" / f"{site}_position.json"
        ),
        sensor_depth_m=args.sensor_depth,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str or None
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog="python -m ndbc_xml.main",
        description=(
            "Generate NDBC XML submission files from OOI buoy JSON data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "site",
        choices=list(SITES.keys()),
        metavar="SITE",
        help="OOI mooring site code: %(choices)s.",
    )
    parser.add_argument(
        "deployment_dir",
        metavar="DEPLOYMENT_DIR",
        help=(
            "Root directory for this deployment "
            "(e.g. /home/ooiuser/data/parsed/ce02shsm/D00020). "
            "METBK and WAVSS subdirectories are derived automatically. "
            "Switching to a new deployment directory (e.g. D00021) "
            "automatically triggers a fresh start."
        ),
    )
    parser.add_argument(
        "--xml-out",
        metavar="DIR",
        default=None,
        help=(
            "Output directory for XML files. "
            "Defaults to <DEPLOYMENT_DIR>/xml."
        ),
    )
    parser.add_argument(
        "--state-file",
        metavar="PATH",
        default=None,
        help=(
            "JSON file for run-continuity tracking. "
            "Defaults to <DEPLOYMENT_DIR>/<SITE>_position.json."
        ),
    )
    parser.add_argument(
        "--alpha-date",
        metavar="YYYY-MM-DD",
        default=None,
        help=(
            "Date for IGRF magnetic declination calculation. "
            "Defaults to the mid-point of the processing window. "
            "Override for back-processing historical deployments."
        ),
    )
    parser.add_argument(
        "--sensor-depth",
        type=float,
        default=1.5,
        metavar="meters",
        help="CTD/temperature sensor depth below surface (m, positive down).",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        default=False,
        help=(
            "Delete the state file and reprocess all available data from "
            "scratch. Use after a bug fix or data correction."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        default=None,
        help="Append log output to this file in addition to stderr.",
    )

    return parser.parse_args(argv)


def _configure_logging(level: str, log_file: str | None) -> None:
    """Set up root logger for stderr (and optionally a file).

    Parameters
    ----------
    level : str
        Logging level name, e.g. ``'INFO'``.
    log_file : str or None
        Optional path to append log records to.
    """
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%SZ"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI.

    Parameters
    ----------
    argv : list of str or None
        Override ``sys.argv[1:]`` for testing.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error, 2 if no new data.
    """
    args = _parse_args(argv)
    _configure_logging(args.log_level, args.log_file)
    log = logging.getLogger(__name__)

    # Validate alpha_date if supplied
    alpha_date: datetime | None = None
    if args.alpha_date:
        try:
            alpha_date = datetime.strptime(
                args.alpha_date, "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            log.error(
                "--alpha-date '%s' is not a valid YYYY-MM-DD date.",
                args.alpha_date,
            )
            return 1

    # Validate deployment_dir exists
    deploy = Path(args.deployment_dir)
    if not deploy.is_dir():
        log.error("DEPLOYMENT_DIR does not exist: %s", deploy)
        return 1

    config = _build_config(args)

    # --reprocess: clear state before running so the pipeline starts fresh
    if args.reprocess:
        clear_state(config.state_file)

    # Log the effective configuration so cron output is self-documenting
    log.info("Site:           %s (NDBC %s)", config.site, config.ndbc_id)
    log.info("Deployment dir: %s", deploy)
    log.info("METBK dir:      %s", config.metbk_dir)
    log.info("WAVSS dir:      %s", config.wavss_dir)
    log.info("XML output:     %s", config.xml_out_dir)
    log.info("State file:     %s", config.state_file)
    log.info("Reprocess:      %s", args.reprocess)
    log.info(
        "Alpha date:     %s",
        alpha_date.strftime("%Y-%m-%d") if alpha_date
        else "mid-deployment (auto)",
    )
    log.info("Sensor depth:   %.2f m", config.sensor_depth_m)

    # If an explicit alpha_date was supplied, fix the date used for
    # declination by temporarily overriding the pipeline's reference.
    if alpha_date is not None:
        import ndbc_xml.pipeline as _pipeline
        from .declination import get_declination as _get_decl

        _original = _pipeline.get_declination

        def _fixed_date_decl(latitude, longitude, date):
            return _get_decl(latitude, longitude, alpha_date)

        _pipeline.get_declination = _fixed_date_decl
        log.debug("Declination fixed to alpha_date %s", alpha_date.date())

    try:
        result = run_station(config)
    except FileNotFoundError as exc:
        log.error("Data directory not found: %s", exc)
        return 1
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        return 1
    finally:
        if alpha_date is not None:
            _pipeline.get_declination = _original

    if result is None:
        log.info("No new data to process — exiting.")
        return 2

    log.info("XML written: %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
