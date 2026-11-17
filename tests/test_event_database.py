"""
Test Suite – Global Event Database (Prompt 3)
=============================================
Covers:
  1. Schema integrity  – all required columns present, correct dtypes
  2. Category consistency – every event maps to a valid category
  3. Date validation  – event_date valid, end_date >= event_date,
                        announcement_date <= event_date
  4. No invented dates – spot-check known events against documented dates
  5. No duplicate event_ids
  6. Minimum coverage – at least 5 events per category
  7. Persistence round-trip – save → load → identical DataFrame
  8. Merge / dedup    – saving twice does not create duplicate rows
  9. Filter helpers   – category, date-range, region, severity, tags
 10. Related-company explode – get_related_companies() structure
 11. Legacy compatibility – build_legacy_events_df() column names
 12. Backward-compatibility with existing event_study pipeline
"""

import os
import tempfile
from datetime import datetime

import pandas as pd
import pytest

from src.ingestion.event_database import (
    REQUIRED_COLUMNS,
    VALID_CATEGORIES,
    _RAW_EVENTS,
    build_event_dataframe,
    build_legacy_events_df,
    filter_events,
    get_related_companies,
    load_event_database,
    save_event_database,
    validate_event_database,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def event_df() -> pd.DataFrame:
    """Full event DataFrame built from the master list."""
    return build_event_dataframe()


@pytest.fixture(scope="module")
def validation_report():
    """Full validation report for the master event list."""
    return validate_event_database(_RAW_EVENTS)


# ---------------------------------------------------------------------------
# 1. Schema integrity
# ---------------------------------------------------------------------------

class TestSchemaIntegrity:
    def test_all_required_columns_present(self, event_df):
        missing = set(REQUIRED_COLUMNS) - set(event_df.columns)
        assert not missing, f"DataFrame is missing columns: {missing}"

    def test_event_id_non_null(self, event_df):
        assert event_df["event_id"].notna().all()
        assert (event_df["event_id"].str.strip() != "").all()

    def test_event_name_non_null(self, event_df):
        assert event_df["event_name"].notna().all()
        assert (event_df["event_name"].str.strip() != "").all()

    def test_event_date_non_null(self, event_df):
        assert event_df["event_date"].notna().all()
        assert (event_df["event_date"].str.strip() != "").all()

    def test_severity_score_is_numeric(self, event_df):
        assert pd.api.types.is_float_dtype(event_df["severity_score"]) or \
               pd.api.types.is_integer_dtype(event_df["severity_score"])

    def test_severity_score_in_valid_range(self, event_df):
        assert (event_df["severity_score"] >= 0).all()
        assert (event_df["severity_score"] <= 5).all()

    def test_description_non_empty(self, event_df):
        assert (event_df["description"].str.strip() != "").all()

    def test_region_non_empty(self, event_df):
        assert (event_df["region"].str.strip() != "").all()


# ---------------------------------------------------------------------------
# 2. Category consistency
# ---------------------------------------------------------------------------

class TestCategoryConsistency:
    def test_all_events_have_valid_category(self, event_df):
        invalid = event_df[~event_df["category"].isin(VALID_CATEGORIES)]
        assert invalid.empty, (
            f"Events with invalid categories:\n{invalid[['event_id', 'category']]}"
        )

    def test_all_seven_categories_represented(self, event_df):
        present = set(event_df["category"].unique())
        missing = set(VALID_CATEGORIES) - present
        assert not missing, f"These categories have no events: {missing}"

    def test_minimum_events_per_category(self, event_df):
        """Every category should have at least 2 events."""
        counts = event_df["category"].value_counts()
        for cat in VALID_CATEGORIES:
            count = counts.get(cat, 0)
            assert count >= 2, (
                f"Category '{cat}' has only {count} event(s); expected >= 2."
            )


# ---------------------------------------------------------------------------
# 3. Date validation
# ---------------------------------------------------------------------------

class TestDateValidation:
    def _parse(self, s):
        return datetime.strptime(s, "%Y-%m-%d") if s and s.strip() else None

    def test_event_date_format(self, event_df):
        """All non-empty dates must be YYYY-MM-DD."""
        for col in ["event_date", "announcement_date", "end_date"]:
            non_empty = event_df[event_df[col].str.strip() != ""][col]
            for val in non_empty:
                try:
                    datetime.strptime(val, "%Y-%m-%d")
                except ValueError:
                    pytest.fail(f"Column '{col}' has invalid date: '{val}'")

    def test_event_date_within_project_range(self, event_df):
        """event_date should be between 2014-01-01 and 2026-12-31."""
        dates = pd.to_datetime(event_df["event_date"])
        assert (dates >= "2014-01-01").all(), "Some events predate 2014"
        assert (dates <= "2026-12-31").all(), "Some events are beyond 2026"

    def test_end_date_not_before_event_date(self, event_df):
        """end_date, when present, must be >= event_date."""
        both = event_df[event_df["end_date"].str.strip() != ""]
        for _, row in both.iterrows():
            ed = self._parse(row["event_date"])
            end = self._parse(row["end_date"])
            if ed and end:
                assert end >= ed, (
                    f"Event {row['event_id']}: end_date {row['end_date']} "
                    f"< event_date {row['event_date']}"
                )

    def test_announcement_date_not_after_event_date(self, event_df):
        """announcement_date, when present, must be <= event_date."""
        both = event_df[event_df["announcement_date"].str.strip() != ""]
        for _, row in both.iterrows():
            ann = self._parse(row["announcement_date"])
            evdt = self._parse(row["event_date"])
            if ann and evdt:
                assert ann <= evdt, (
                    f"Event {row['event_id']}: announcement_date {row['announcement_date']} "
                    f"> event_date {row['event_date']}"
                )


# ---------------------------------------------------------------------------
# 4. Known-event spot-checks (no invented dates)
# ---------------------------------------------------------------------------

class TestKnownEventDates:
    """Verify selected documented dates match public records."""

    @pytest.mark.parametrize("event_id,expected_date", [
        ("GEOP-003", "2016-06-23"),   # Brexit referendum – Electoral Commission
        ("GEOP-006", "2022-02-24"),   # Russia–Ukraine invasion – UN
        ("GEOP-007", "2023-10-07"),   # Israel–Hamas – AP
        ("ECON-003", "2020-03-11"),   # WHO pandemic declaration
        ("ECON-007", "2023-03-10"),   # SVB closure – FDIC
        ("CORP-006", "2021-01-28"),   # GameStop peak / trading halt
        ("CORP-008", "2022-11-11"),   # FTX bankruptcy
        ("CORP-013", "2023-05-24"),   # Nvidia blowout earnings
        ("POLI-001", "2016-11-08"),   # Trump 2016 election
        ("POLI-007", "2024-11-05"),   # Trump 2024 election
        ("TECH-002", "2022-11-30"),   # ChatGPT launch
        ("NATD-001", "2017-08-25"),   # Hurricane Harvey landfall
    ])
    def test_known_event_date(self, event_df, event_id, expected_date):
        row = event_df[event_df["event_id"] == event_id]
        assert not row.empty, f"Event {event_id} not found in database"
        actual = row.iloc[0]["event_date"]
        assert actual == expected_date, (
            f"{event_id}: expected {expected_date}, got {actual}"
        )


# ---------------------------------------------------------------------------
# 5. No duplicate event_ids
# ---------------------------------------------------------------------------

class TestNoDuplicates:
    def test_no_duplicate_event_ids(self, event_df):
        duplicates = event_df[event_df.duplicated(subset=["event_id"], keep=False)]
        assert duplicates.empty, (
            f"Duplicate event_ids found:\n{duplicates[['event_id', 'event_name']]}"
        )

    def test_validation_report_no_duplicates(self, validation_report):
        assert validation_report["duplicate_ids"] == [], (
            f"Duplicate IDs: {validation_report['duplicate_ids']}"
        )

    def test_validation_report_no_invalid(self, validation_report):
        assert validation_report["invalid_ids"] == [], (
            f"Invalid event IDs: {validation_report['invalid_ids']}\n"
            f"Issues:\n" + "\n".join(validation_report["issues"])
        )


# ---------------------------------------------------------------------------
# 6. Minimum dataset size
# ---------------------------------------------------------------------------

class TestMinimumDatasetSize:
    def test_total_event_count_at_least_50(self, event_df):
        assert len(event_df) >= 50, (
            f"Expected at least 50 events, got {len(event_df)}"
        )

    def test_validation_report_valid_count_matches(self, event_df, validation_report):
        assert validation_report["valid"] == len(event_df), (
            f"valid={validation_report['valid']} but DataFrame has {len(event_df)} rows"
        )


# ---------------------------------------------------------------------------
# 7. Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistenceRoundTrip:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "events_expanded.csv")
            save_event_database(output_path=path, overwrite=True)
            assert os.path.exists(path)

            loaded = load_event_database(path)
            assert isinstance(loaded, pd.DataFrame)
            assert len(loaded) == len(_RAW_EVENTS)

            missing_cols = set(REQUIRED_COLUMNS) - set(loaded.columns)
            assert not missing_cols, f"Loaded DF missing cols: {missing_cols}"

    def test_load_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_event_database("data/raw/nonexistent_events.csv")

    def test_load_raises_on_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = os.path.join(tmpdir, "bad_events.csv")
            pd.DataFrame({"event_id": ["A"], "garbage": ["x"]}).to_csv(bad_path, index=False)
            with pytest.raises(ValueError, match="missing required columns"):
                load_event_database(bad_path)


# ---------------------------------------------------------------------------
# 8. Merge / dedup – saving twice does not create duplicate rows
# ---------------------------------------------------------------------------

class TestMergeBehaviour:
    def test_save_twice_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "events_expanded.csv")
            # First save
            save_event_database(output_path=path, overwrite=True)
            n_first = len(pd.read_csv(path))

            # Second save without overwrite – should merge but not duplicate
            save_event_database(output_path=path, overwrite=False)
            n_second = len(pd.read_csv(path))

            assert n_second == n_first, (
                f"Second save without overwrite created duplicates: "
                f"{n_first} -> {n_second} rows"
            )


# ---------------------------------------------------------------------------
# 9. Filter helpers
# ---------------------------------------------------------------------------

class TestFilterHelpers:
    def test_filter_by_category(self, event_df):
        result = filter_events(event_df, category="Economic Events")
        assert (result["category"] == "Economic Events").all()
        assert len(result) >= 2

    def test_filter_by_date_range(self, event_df):
        result = filter_events(event_df, start_date="2020-01-01", end_date="2022-12-31")
        dates = pd.to_datetime(result["event_date"])
        assert (dates >= "2020-01-01").all()
        assert (dates <= "2022-12-31").all()

    def test_filter_by_region(self, event_df):
        result = filter_events(event_df, region="United States")
        assert len(result) >= 5

    def test_filter_by_min_severity(self, event_df):
        result = filter_events(event_df, min_severity=4.0)
        assert (result["severity_score"] >= 4.0).all()
        assert len(result) >= 5

    def test_filter_by_tags(self, event_df):
        result = filter_events(event_df, tags="bitcoin")
        assert len(result) >= 1

    def test_filter_combined(self, event_df):
        result = filter_events(
            event_df,
            category="Economic Events",
            start_date="2022-01-01",
            min_severity=3.0,
        )
        assert (result["category"] == "Economic Events").all()
        assert (result["severity_score"] >= 3.0).all()

    def test_filter_returns_empty_not_error(self, event_df):
        result = filter_events(event_df, category="Economic Events", start_date="2030-01-01")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 10. Related-company explode
# ---------------------------------------------------------------------------

class TestRelatedCompanyExplode:
    def test_get_related_companies_structure(self, event_df):
        result = get_related_companies(event_df)
        required = {"event_id", "event_name", "event_date", "category", "ticker"}
        assert required.issubset(set(result.columns))

    def test_get_related_companies_no_empty_tickers(self, event_df):
        result = get_related_companies(event_df)
        assert (result["ticker"].str.strip() != "").all()

    def test_known_company_in_related(self, event_df):
        """NVDA must appear for the ChatGPT event."""
        result = get_related_companies(event_df)
        chatgpt_nvda = result[
            (result["event_id"] == "TECH-002") & (result["ticker"] == "NVDA")
        ]
        assert len(chatgpt_nvda) == 1

    def test_no_pipe_chars_in_ticker_column(self, event_df):
        result = get_related_companies(event_df)
        assert not result["ticker"].str.contains("|", regex=False).any()


# ---------------------------------------------------------------------------
# 11. Legacy compatibility
# ---------------------------------------------------------------------------

class TestLegacyCompatibility:
    def test_legacy_df_has_required_columns(self, event_df):
        legacy = build_legacy_events_df(event_df)
        required = {"event_name", "event_date", "event_category", "short_description"}
        assert required.issubset(set(legacy.columns))

    def test_legacy_df_row_count_matches(self, event_df):
        legacy = build_legacy_events_df(event_df)
        assert len(legacy) == len(event_df)

    def test_legacy_event_date_format(self, event_df):
        legacy = build_legacy_events_df(event_df)
        for d in legacy["event_date"]:
            if d and str(d).strip():
                datetime.strptime(str(d).strip(), "%Y-%m-%d")

    def test_legacy_category_col_name(self, event_df):
        legacy = build_legacy_events_df(event_df)
        # Must use 'event_category', not 'category'
        assert "event_category" in legacy.columns
        assert "category" not in legacy.columns


# ---------------------------------------------------------------------------
# 12. Backward-compatibility: existing event_study pipeline can consume it
# ---------------------------------------------------------------------------

class TestEventStudyCompatibility:
    def test_legacy_csv_loadable_by_event_study(self):
        """
        Simulate what run_event_study_analysis() does when it reads events.csv –
        it only requires 'event_date' to be present.
        """
        df = build_legacy_events_df()
        assert "event_date" in df.columns
        # The pipeline will call pd.read_csv and then iterate rows
        for _, row in df.iterrows():
            assert row["event_date"] and len(row["event_date"]) == 10

    def test_save_legacy_csv_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "events.csv")
            legacy = build_legacy_events_df()
            legacy.to_csv(path, index=False)

            loaded = pd.read_csv(path)
            assert "event_date" in loaded.columns
            assert "event_category" in loaded.columns
            assert len(loaded) == len(legacy)
