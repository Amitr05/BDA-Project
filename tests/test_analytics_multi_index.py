
import pandas as pd
import pytest

from src.analytics.event_study import (
    calculate_event_metrics,
    find_effective_trading_day
)


def make_frame(dates, closes):
    return pd.DataFrame({
        "Date": dates,
        "Close": closes
    })


def test_effective_day_uses_first_trading_day_on_or_after_event():

    dates = [
        "2024-01-04",
        "2024-01-05",
        "2024-01-08"
    ]

    assert find_effective_trading_day(
        "2024-01-06", dates
    ) == 2

    assert find_effective_trading_day(
        "2025-01-01", dates
    ) is None


def test_each_index_can_use_its_own_trading_calendar_and_long_format_identity():

    dates_a = (
        pd.date_range(
            "2024-01-01",
            periods=45,
            freq="B"
        )
        .strftime("%Y-%m-%d")
        .tolist()
    )

    dates_b = [
        d for i, d in enumerate(dates_a)
        if i != 12
    ]

    vix = make_frame(
        dates_a,
        [20.0] * len(dates_a)
    )

    event = {
        "event_name": "Test",
        "event_date": dates_a[10],
        "event_category": "Test"
    }

    a = calculate_event_metrics(
        event,
        make_frame(
            dates_a,
            [100 + i for i in range(len(dates_a))]
        ),
        vix,
        index_name="S&P 500",
        index_key="sp500"
    )

    b = calculate_event_metrics(
        event,
        make_frame(
            dates_b,
            [200 + i for i in range(len(dates_b))]
        ),
        vix,
        index_name="NASDAQ Composite",
        index_key="nasdaq"
    )

    assert a["index"] == "S&P 500"
    assert b["index"] == "NASDAQ Composite"

    assert a["ticker"] == "^GSPC"
    assert b["ticker"] == "^IXIC"

    assert (
        a["effective_trading_date"]
        == b["effective_trading_date"]
    )

    assert a["post_5d_date"] != b["post_5d_date"]


def test_incomplete_post_windows_are_not_clamped_to_last_row():

    dates = (
        pd.date_range(
            "2024-01-01",
            periods=12,
            freq="B"
        )
        .strftime("%Y-%m-%d")
        .tolist()
    )

    index = make_frame(
        dates,
        [100.0 + i for i in range(len(dates))]
    )

    vix = make_frame(
        dates,
        [20.0] * len(dates)
    )

    event = {
        "event_name": "Late event",
        "event_date": dates[9]
    }

    metrics = calculate_event_metrics(
        event,
        index,
        vix
    )

    assert metrics["post_5d_return_pct"] is None
    assert metrics["post_30d_return_pct"] is None
    assert metrics["max_drawdown_30d_pct"] is None

    assert metrics["post_5d_date"] is None
    assert metrics["post_30d_date"] is None

    assert (
        metrics["sp500_post_5d_return_pct"]
        is None
    )


def test_s_and_p_500_legacy_columns_match_generic_columns():

    dates = (
        pd.date_range(
            "2024-01-01",
            periods=45,
            freq="B"
        )
        .strftime("%Y-%m-%d")
        .tolist()
    )

    index = make_frame(
        dates,
        [100.0 + i for i in range(len(dates))]
    )

    vix = make_frame(
        dates,
        [20.0] * len(dates)
    )

    event = {
        "event_name": "Test",
        "event_date": dates[10]
    }

    metrics = calculate_event_metrics(
        event,
        index,
        vix
    )

    assert (
        metrics["sp500_price_t0"]
        == metrics["price_t0"]
    )

    assert (
        metrics["sp500_pre_5d_return_pct"]
        == metrics["pre_5d_return_pct"]
    )

    assert (
        metrics["sp500_post_5d_return_pct"]
        == metrics["post_5d_return_pct"]
    )

    assert (
        metrics["sp500_post_30d_return_pct"]
        == metrics["post_30d_return_pct"]
    )


def test_multi_index_pipeline_returns_one_row_per_event_and_index(
    tmp_path,
    monkeypatch
):

    from src.analytics import event_study

    dates = (
        pd.date_range(
            "2024-01-01",
            periods=50,
            freq="B"
        )
        .strftime("%Y-%m-%d")
        .tolist()
    )

    frames = {
        "sp": make_frame(
            dates, [100.0 + i for i in range(50)]
        ),
        "na": make_frame(
            dates, [200.0 + i for i in range(50)]
        ),
        "dj": make_frame(
            dates, [300.0 + i for i in range(50)]
        ),
        "ru": make_frame(
            dates, [400.0 + i for i in range(50)]
        ),
        "vix": make_frame(
            dates, [20.0] * 50
        )
    }

    path_to_frame = {
        str(tmp_path / f"{key}.parquet"): value
        for key, value in frames.items()
    }

    for path in path_to_frame:
        open(path, "w").close()

    events_path = tmp_path / "events.csv"

    pd.DataFrame([
        {
            "event_name": "A",
            "event_date": dates[10]
        },
        {
            "event_name": "B",
            "event_date": dates[11]
        }
    ]).to_csv(
        events_path,
        index=False
    )

    monkeypatch.setattr(
        event_study.pd,
        "read_parquet",
        lambda path: path_to_frame[str(path)].copy()
    )

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, **kwargs: open(
            path, "wb"
        ).close()
    )

    result = event_study.run_event_study_analysis(
        events_csv_path=str(events_path),

        sp500_parquet_path=str(
            tmp_path / "sp.parquet"
        ),

        vix_parquet_path=str(
            tmp_path / "vix.parquet"
        ),

        output_csv_path=str(
            tmp_path / "results.csv"
        ),

        output_parquet_path=str(
            tmp_path / "results.parquet"
        ),

        index_parquet_paths={
            "sp500": str(tmp_path / "sp.parquet"),
            "nasdaq": str(tmp_path / "na.parquet"),
            "dow_jones": str(tmp_path / "dj.parquet"),
            "russell2000": str(tmp_path / "ru.parquet")
        }
    )

    assert len(result) == 8

    assert set(result["index_key"]) == {
        "sp500",
        "nasdaq",
        "dow_jones",
        "russell2000"
    }

    assert (
        result.groupby("event_name")
        .size()
        .to_dict()
        == {"A": 4, "B": 4}
    )

    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "results.parquet").exists()


def test_original_sp500_calculation_values_are_preserved():

    dates = (
        pd.date_range(
            "2023-01-01",
            periods=45,
            freq="B"
        )
        .strftime("%Y-%m-%d")
        .tolist()
    )

    prices = [100.0] * len(dates)

    prices[5] = 100.0
    prices[10] = 110.0
    prices[15] = 104.5
    prices[40] = 121.0

    vix_values = [15.0] * len(dates)

    vix_values[10] = 20.0
    vix_values[12] = 35.0
    vix_values[15] = 22.0

    metrics = calculate_event_metrics(
        {
            "event_name": "Baseline",
            "event_date": dates[10]
        },
        make_frame(dates, prices),
        make_frame(dates, vix_values)
    )

    assert metrics["pre_5d_return_pct"] == pytest.approx(10.0)

    assert metrics["post_5d_return_pct"] == pytest.approx(-5.0)

    assert metrics["post_30d_return_pct"] == pytest.approx(10.0)

    assert metrics["vix_at_event"] == 20.0

    assert metrics["vix_post_5d"] == 22.0

    assert metrics["vix_max_30d"] == 35.0

    assert metrics["vix_5d_change"] == 2.0
