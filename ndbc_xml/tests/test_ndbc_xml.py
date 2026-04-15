#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for ndbc_xml processing and XML writing.

Run with:  pytest tests/test_ndbc_xml.py -v
"""


import numpy as np
import pandas as pd
import pytest

from ndbc_xml.process import (
    rotate_wind,
    wind_speed,
    wind_direction,
    rotate_wave_dir,
    calc_rain_rate,
    make_bin_edges,
    bin_observations,
    apply_qc,
)
from ndbc_xml.xml_writer import (
    build_message,
    write_xml,
    xml_filename,
    _fmt,
    _tag,
)
from ndbc_xml.state import (
    load_state,
    save_state,
    clear_state,
    bin_start_from_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


# ---------------------------------------------------------------------------
# process.py
# ---------------------------------------------------------------------------

class TestRotateWind:
    def test_zero_alpha_no_change(self):
        u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        ur, vr = rotate_wind(u, v, alpha_deg=0.0)
        np.testing.assert_allclose(ur, u)
        np.testing.assert_allclose(vr, v)

    def test_90_deg_rotation(self):
        u, v = np.array([1.0]), np.array([0.0])
        ur, vr = rotate_wind(u, v, alpha_deg=90.0)
        np.testing.assert_allclose(ur, [0.0], atol=1e-10)
        np.testing.assert_allclose(vr, [-1.0], atol=1e-10)

    def test_magnitude_preserved(self):
        rng = np.random.default_rng(42)
        u, v = rng.standard_normal(50), rng.standard_normal(50)
        ur, vr = rotate_wind(u, v, alpha_deg=37.5)
        mag_orig = np.sqrt(u**2 + v**2)
        mag_rot = np.sqrt(ur**2 + vr**2)
        np.testing.assert_allclose(mag_rot, mag_orig, atol=1e-12)


class TestWindSpeed:
    def test_basic(self):
        np.testing.assert_allclose(
            wind_speed(np.array([3.0]), np.array([4.0])),
            [5.0],
        )

    def test_zero(self):
        assert wind_speed(np.array([0.0]), np.array([0.0]))[0] == 0.0


class TestWindDirection:
    def test_northward_is_180(self):
        # Wind blowing north (v > 0) → coming from south → 180°
        d = wind_direction(np.array([0.0]), np.array([5.0]))
        np.testing.assert_allclose(d, [180.0], atol=1e-10)

    def test_eastward_is_270(self):
        d = wind_direction(np.array([5.0]), np.array([0.0]))
        np.testing.assert_allclose(d, [270.0], atol=1e-10)

    def test_range_0_to_360(self):
        rng = np.random.default_rng(7)
        u, v = rng.standard_normal(200), rng.standard_normal(200)
        d = wind_direction(u, v)
        assert (d >= 0).all() and (d <= 360).all()


class TestRotateWaveDir:
    def test_output_unit_vectors(self):
        dirs = np.array([0.0, 90.0, 180.0, 270.0])
        u, v = rotate_wave_dir(dirs, alpha_deg=0.0)
        mag = np.sqrt(u**2 + v**2)
        np.testing.assert_allclose(mag, np.ones(4), atol=1e-12)


class TestCalcRainRate:
    def test_diff_clipping(self):
        precip = np.array([0.0, 1.0, 3.6, 4.0, 3.9])
        rate = calc_rain_rate(precip)
        assert np.isnan(rate[0])          # first always NaN
        assert np.isclose(rate[1], 1.0)
        assert np.isnan(rate[2])          # >2.5 spike masked
        assert np.isclose(rate[3], 0.4)
        assert np.isnan(rate[4])          # negative masked

    def test_first_element_nan(self):
        precip = np.array([5.0, 6.0])
        assert np.isnan(calc_rain_rate(precip)[0])


class TestBinObservations:
    """Tests for resample-based bin_observations."""

    _t0 = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

    def _metbk(self, times_s, u=3.0, v=4.0, precip=None):
        n = len(times_s)
        if precip is None:
            precip = np.arange(n, dtype=float)
        return pd.DataFrame({
            "time":                    np.array(times_s, dtype=float),
            "eastward_wind_velocity":  np.full(n, u),
            "northward_wind_velocity": np.full(n, v),
            "barometric_pressure":     np.full(n, 1013.0),
            "air_temperature":         np.full(n, 15.0),
            "relative_humidity":       np.full(n, 75.0),
            "shortwave_irradiance":    np.full(n, 200.0),
            "longwave_irradiance":     np.full(n, 350.0),
            "sea_surface_temperature": np.full(n, 12.0),
            "sea_surface_conductivity":np.full(n, 3.5),
            "precipitation_level":     np.asarray(precip, dtype=float),
        })

    def _wavss(self, times_s):
        n = len(times_s)
        return pd.DataFrame({
            "time":                    np.array(times_s, dtype=float),
            "significant_wave_height": np.full(n, 1.5),
            "maximum_wave_height":     np.full(n, 2.0),
            "peak_period":             np.full(n, 10.0),
            "average_wave_period":     np.full(n, 8.0),
            "mean_wave_direction":     np.full(n, 270.0),
        })

    def _ts(self, hhmm: str) -> int:
        return int(pd.Timestamp(f"2026-01-01 {hhmm}", tz="UTC").timestamp())

    def test_output_shape(self):
        # Samples at :00/:10/:20; after +5 min shift → :05/:15/:25, one per bin.
        edges = make_bin_edges(self._t0, self._t0 + pd.Timedelta(minutes=30))
        times_s = [self._ts("00:00"), self._ts("00:10"), self._ts("00:20")]
        df = bin_observations(self._metbk(times_s), self._wavss(times_s),
                              edges, alpha_deg=0.0)
        assert len(df) == 3

    def test_empty_bin_is_nan(self):
        edges = make_bin_edges(self._t0, self._t0 + pd.Timedelta(minutes=30))
        # Samples at :00 and :20; after the +5 min METBK shift they land at
        # :05 and :25, falling in bins [00:00, 00:10) and [00:20, 00:30).
        # The middle bin [00:10, 00:20) is empty → NaN.
        times_s = [self._ts("00:00"), self._ts("00:20")]
        df = bin_observations(self._metbk(times_s), self._wavss(times_s),
                              edges, alpha_deg=0.0)
        assert len(df) == 3
        assert np.isnan(df["wind_speed"].iloc[1])

    def test_nan_excluded_from_mean(self):
        edges = make_bin_edges(self._t0, self._t0 + pd.Timedelta(minutes=10))
        # Samples at :00 and :04; after +5 min shift → :05 and :09, both in
        # bin [00:00, 00:10). First sample has NaN wind; mean should ignore it.
        times_s = [self._ts("00:00"), self._ts("00:04")]
        metbk = self._metbk(times_s)
        metbk.loc[0, "eastward_wind_velocity"] = np.nan
        metbk.loc[0, "northward_wind_velocity"] = np.nan
        df = bin_observations(metbk, self._wavss(times_s), edges, alpha_deg=0.0)
        # Only the second sample (u=3, v=4) contributes → speed=5
        np.testing.assert_allclose(df["wind_speed"].iloc[0], 5.0)

    def test_wind_speed_correct(self):
        """u=3, v=4, alpha=0 → speed=5.0.

        Sample at :00; after +5 min METBK shift → :05, inside bin [00:00, 00:10).
        """
        edges = make_bin_edges(self._t0, self._t0 + pd.Timedelta(minutes=10))
        times_s = [self._ts("00:00")]
        df = bin_observations(self._metbk(times_s), self._wavss(times_s),
                              edges, alpha_deg=0.0)
        np.testing.assert_allclose(df["wind_speed"].iloc[0], 5.0)

    def test_bin_left_edge_timestamps(self):
        """time column holds bin left-edge timestamps (00:00, 00:10, …)."""
        edges = make_bin_edges(self._t0, self._t0 + pd.Timedelta(minutes=20))
        # Sample at :00; after +5 min shift → :05, in first bin.
        times_s = [self._ts("00:00")]
        df = bin_observations(self._metbk(times_s), self._wavss(times_s),
                              edges, alpha_deg=0.0)
        assert df["time"].iloc[0] == pd.Timestamp("2026-01-01 00:00", tz="UTC")
        assert df["time"].iloc[1] == pd.Timestamp("2026-01-01 00:10", tz="UTC")


class TestApplyQC:
    def _df(self, **overrides):
        defaults = {
            "time":        [pd.Timestamp("2026-01-01", tz="UTC")],
            "wind_dir":    [180.0],
            "wind_speed":  [5.0],
            "baro":        [1013.0],
            "air_temp":    [15.0],
            "rel_humidity":[75.0],
            "shortwave":   [200.0],
            "longwave":    [400.0],
            "sst":         [12.0],
            "salinity":    [33.0],
            "sig_wave_hgt":[1.5],
            "max_wave_hgt":[2.0],
            "peak_period": [10.0],
            "avg_period":  [8.0],
            "wave_dir":    [270.0],
        }
        defaults.update(overrides)
        return pd.DataFrame(defaults)

    def test_valid_passes(self):
        df = self._df()
        result = apply_qc(df)
        assert not result["wind_speed"].isna().any()

    def test_out_of_range_becomes_nan(self):
        df = self._df(wind_speed=[70.0])
        result = apply_qc(df)
        assert np.isnan(result["wind_speed"].iloc[0])

    def test_input_unchanged(self):
        df = self._df(baro=[500.0])
        _ = apply_qc(df)
        assert df["baro"].iloc[0] == 500.0  # original not mutated


# ---------------------------------------------------------------------------
# xml_writer.py
# ---------------------------------------------------------------------------

class TestFmt:
    def test_finite(self):
        assert _fmt(1.5) == "1.500000"

    def test_nan(self):
        assert _fmt(float("nan")) == "-9999"

    def test_inf(self):
        assert _fmt(float("inf")) == "-9999"


class TestTag:
    def test_structure(self):
        assert _tag("wspd1", 5.0) == "<wspd1>5.000000</wspd1>"

    def test_missing(self):
        assert _tag("wspd1", float("nan")) == "<wspd1>-9999</wspd1>"


class TestBuildMessage:
    def test_contains_station(self):
        msg = build_message(
            station_id="46097",
            timestamp=pd.Timestamp("2026-03-30 12:00:00", tz="UTC"),
            wind_dir=270.0, wind_speed=5.0, baro=1013.0,
            air_temp=12.0, rel_humidity=80.0, shortwave=200.0,
            longwave=350.0, sig_wave_hgt=1.2, peak_period=9.0,
            wave_dir=280.0, sst=11.5, salinity=32.8,
            sensor_depth_m=1.15,
        )
        assert "<station>46097</station>" in msg
        assert "<date>03/30/2026 12:00:00</date>" in msg
        assert "<wdir1>270.000000</wdir1>" in msg
        assert "<fm64iii>830</fm64iii>" in msg
        assert "<dp001>1.15</dp001>" in msg
        assert msg.startswith("<message>")
        assert msg.endswith("</message>")

    def test_nan_written_as_missing(self):
        msg = build_message(
            station_id="46097",
            timestamp=pd.Timestamp("2026-03-30 12:00:00", tz="UTC"),
            wind_dir=float("nan"), wind_speed=5.0, baro=1013.0,
            air_temp=12.0, rel_humidity=80.0, shortwave=200.0,
            longwave=350.0, sig_wave_hgt=1.2, peak_period=9.0,
            wave_dir=280.0, sst=11.5, salinity=32.8,
            sensor_depth_m=1.15,
        )
        assert "<wdir1>-9999</wdir1>" in msg


class TestWriteXml:
    def _sample_df(self):
        ts = pd.date_range("2026-03-30 00:05", periods=3,
                           freq="10min", tz="UTC")
        return pd.DataFrame({
            "time":        ts,
            "wind_dir":    [270.0, 265.0, 268.0],
            "wind_speed":  [5.0, 5.5, 4.8],
            "baro":        [1013.0, 1012.5, 1013.2],
            "air_temp":    [12.0, 12.1, 12.0],
            "rel_humidity":[80.0, 81.0, 79.5],
            "shortwave":   [200.0, 210.0, 205.0],
            "longwave":    [350.0, 355.0, 352.0],
            "sig_wave_hgt":[1.2, 1.3, 1.25],
            "peak_period": [9.0, 9.2, 9.1],
            "wave_dir":    [280.0, 275.0, 277.0],
            "sst":         [11.5, 11.4, 11.6],
            "salinity":    [32.8, 32.9, 32.7],
        })

    def test_writes_file(self, tmp_path):
        df = self._sample_df()
        out = write_xml(df, "46097", tmp_path / "test.xml", sensor_depth_m=1.15)
        assert out.exists()

    def test_header_line(self, tmp_path):
        df = self._sample_df()
        out = write_xml(df, "46097", tmp_path / "test.xml", sensor_depth_m=1.15)
        first_line = out.read_text(encoding="ISO-8859-1").splitlines()[0]
        assert first_line.startswith("SXML99 KWBC ")

    def test_xml_declaration(self, tmp_path):
        df = self._sample_df()
        out = write_xml(df, "46097", tmp_path / "test.xml", sensor_depth_m=1.15)
        content = out.read_text(encoding="ISO-8859-1")
        assert '<?xml version="1.0" encoding="ISO-8859-1"?>' in content

    def test_message_count(self, tmp_path):
        df = self._sample_df()
        out = write_xml(df, "46097", tmp_path / "test.xml", sensor_depth_m=1.15)
        content = out.read_text(encoding="ISO-8859-1")
        assert content.count("<message>") == 3

    def test_empty_df_raises(self, tmp_path):
        df = self._sample_df().iloc[0:0]
        with pytest.raises(ValueError):
            write_xml(df, "46097", tmp_path / "empty.xml", sensor_depth_m=1.15)

    def test_xml_filename_format(self):
        name = xml_filename("46097")
        assert name.endswith("-46097.xml")
        parts = name.split("-")
        assert len(parts) == 5  # HH-DD-Mon-YYYY-46097.xml


# ---------------------------------------------------------------------------
# state.py
# ---------------------------------------------------------------------------

class TestState:
    def test_roundtrip(self, tmp_path):
        f = tmp_path / "state.json"
        ts = _ts("2026-03-30T12:00:00+00:00")
        save_state(f, ts, station="CE02")
        loaded = load_state(f)
        assert loaded == ts

    def test_missing_file_returns_none(self, tmp_path):
        assert load_state(tmp_path / "nonexistent.json") is None

    def test_corrupt_file_returns_none(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json")
        assert load_state(f) is None

    def test_floor_to_10min(self):
        ts = _ts("2026-03-30T12:07:33+00:00")
        assert ts.floor("10min") == _ts("2026-03-30T12:00:00+00:00")

    def test_bin_start_no_state_uses_earliest(self, tmp_path):
        """No state file -> start 1 h after earliest data, floored."""
        state = tmp_path / "state.json"
        earliest = _ts("2026-03-01T00:00:00+00:00")
        result = bin_start_from_state(state, earliest)
        assert result == _ts("2026-03-01T01:00:00+00:00")

    def test_bin_start_resumes_from_state(self, tmp_path):
        """Existing state file -> resume from last_bin_end."""
        state = tmp_path / "state.json"
        last = _ts("2026-03-30T10:00:00+00:00")
        save_state(state, last)
        earliest = _ts("2026-03-01T00:00:00+00:00")
        result = bin_start_from_state(state, earliest)
        assert result == last

    def test_clear_state_removes_file(self, tmp_path):
        state = tmp_path / "state.json"
        save_state(state, _ts("2026-03-30T10:00:00+00:00"))
        assert state.exists()
        clear_state(state)
        assert not state.exists()

    def test_clear_state_missing_file_ok(self, tmp_path):
        """clear_state on a non-existent file should not raise."""
        clear_state(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# declination.py
# ---------------------------------------------------------------------------

class TestGetDeclination:
    """Tests for IGRF-based magnetic declination calculation."""

    def test_ce02_range(self):
        """Oregon coast should be ~14-16 deg east declination in 2026."""
        from datetime import datetime
        from ndbc_xml.declination import get_declination
        d = get_declination(44.639, -124.095, datetime(2026, 1, 1))
        assert 13.0 < d < 17.0

    def test_ce09_range(self):
        """Northern Oregon coast — slightly higher declination."""
        from datetime import datetime
        from ndbc_xml.declination import get_declination
        d = get_declination(46.859, -124.973, datetime(2026, 1, 1))
        assert 14.0 < d < 18.0

    def test_positive_east(self):
        """West coast USA: declination should be positive (east)."""
        from datetime import datetime
        from ndbc_xml.declination import get_declination
        d = get_declination(45.0, -124.0, datetime(2026, 1, 1))
        assert d > 0.0

    def test_secular_variation(self):
        """Declination should differ between 2010 and 2026 (secular variation)."""
        from datetime import datetime
        from ndbc_xml.declination import get_declination
        d_2010 = get_declination(44.639, -124.095, datetime(2010, 1, 1))
        d_2026 = get_declination(44.639, -124.095, datetime(2026, 1, 1))
        # ~0.1 deg/yr change — 16 years should give >0.5 deg difference
        assert abs(d_2026 - d_2010) > 0.5

    def test_returns_float(self):
        from datetime import datetime
        from ndbc_xml.declination import get_declination
        d = get_declination(44.639, -124.095, datetime(2026, 1, 1))
        assert isinstance(d, float)


# ---------------------------------------------------------------------------
# ndbc.py
# ---------------------------------------------------------------------------

class TestMain:
    """Tests for the CLI argument parsing and config building."""

    def test_help_exits_zero(self):
        from ndbc_xml.ndbc import main
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_invalid_site_rejected(self, tmp_path):
        from ndbc_xml.ndbc import main
        with pytest.raises(SystemExit) as exc:
            main(["CEXX", str(tmp_path)])
        assert exc.value.code != 0

    def test_missing_deployment_dir(self, tmp_path):
        from ndbc_xml.ndbc import main
        ret = main(["CE02", str(tmp_path / "nonexistent")])
        assert ret == 1

    def test_invalid_alpha_date(self, tmp_path):
        from ndbc_xml.ndbc import main
        ret = main(["CE02", str(tmp_path), "--alpha-date", "not-a-date"])
        assert ret == 1

    def test_build_config_defaults(self, tmp_path):
        """Default xml-out and state-file paths are derived from deployment_dir."""
        from ndbc_xml.ndbc import _parse_args, _build_config
        # Create minimal required subdirs so validation passes
        (tmp_path / "buoy" / "metbk").mkdir(parents=True)
        (tmp_path / "buoy" / "wavss").mkdir(parents=True)
        args = _parse_args(["CE02", str(tmp_path)])
        config = _build_config(args)
        assert config.xml_out_dir == tmp_path / "buoy" / "metbk" / "xml"
        assert config.state_file == tmp_path / "buoy" / "metbk" / "xml" / "CE02_position.json"
        assert config.metbk_dir == tmp_path / "buoy" / "metbk"
        assert config.wavss_dir == tmp_path / "buoy" / "wavss"

    def test_build_config_explicit_overrides(self, tmp_path):
        """Explicit --xml-out and --state-file are respected."""
        from ndbc_xml.ndbc import _parse_args, _build_config
        xml_dir = tmp_path / "myxml"
        state = tmp_path / "mystate.json"
        args = _parse_args([
            "CE04", str(tmp_path),
            "--xml-out", str(xml_dir),
            "--state-file", str(state),
            "--sensor-depth", "2.0",
        ])
        config = _build_config(args)
        assert config.xml_out_dir == xml_dir
        assert config.state_file == state
        assert config.sensor_depth_m == 2.0

    def test_reprocess_clears_state(self, tmp_path):
        """--reprocess deletes the state file before running."""
        from unittest.mock import patch
        from ndbc_xml.ndbc import main
        from ndbc_xml.state import save_state
        import pandas as pd
        state = tmp_path / "buoy" / "metbk" / "xml" / "CE02_position.json"
        save_state(state, pd.Timestamp("2026-01-01", tz="UTC"))
        assert state.exists()
        with patch("ndbc_xml.ndbc.run_station", return_value=None):
            main(["CE02", str(tmp_path), "--reprocess"])
        assert not state.exists()

    def test_site_metadata_complete(self):
        """All four sites have metadata entries."""
        from ndbc_xml.config import SITES
        for site in ("CE02", "CE04", "CE07", "CE09"):
            assert site in SITES
            assert "ndbc_id" in SITES[site]
            assert "latitude" in SITES[site]
            assert "longitude" in SITES[site]

    def test_exit_code_2_no_data(self, tmp_path):
        """Returns exit code 2 when no new data are available."""
        from unittest.mock import patch
        from ndbc_xml.ndbc import main
        (tmp_path / "buoy" / "metbk").mkdir(parents=True)
        (tmp_path / "buoy" / "wavss").mkdir(parents=True)
        with patch("ndbc_xml.ndbc.run_station", return_value=None):
            ret = main(["CE02", str(tmp_path)])
        assert ret == 2
