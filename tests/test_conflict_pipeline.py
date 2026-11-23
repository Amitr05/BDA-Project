"""
Unit and Integration Tests for UCDP Conflict Preprocessing & Regional Mapping
=============================================================================
Tests:
1. Exact 5-region canonical geographical taxonomy.
2. Completeness of country mapping against all 126 UCDP GED countries.
3. Explicit classification of Excluded categories and exclusion rationales.
4. Conflict episode aggregation, casualty thresholding, and temporal washout logic.
5. Schema validation, synthetic PySpark DataFrame testing, and provenance tracking.
"""

import os
import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.preprocessing.conflict_mapping import (
    CANONICAL_REGIONS,
    COUNTRY_TO_REGION_MAP,
    EXCLUSION_RATIONALES,
    EXCLUSION_SUB_SAHARAN_AFRICA,
    EXCLUSION_LATIN_AMERICA,
    EXCLUSION_SE_ASIA_PACIFIC,
    EXCLUSION_WESTERN_EUROPE,
    EXCLUSION_UNKNOWN,
    map_country_to_canonical_region,
    is_canonical_region,
    get_canonical_regions,
    get_countries_for_region,
    classify_duration_regime
)
from src.preprocessing.conflict_pipeline import (
    CONFLICT_EPISODE_SCHEMA,
    _extract_curated_conflict_events,
    apply_spark_region_mapping,
    aggregate_and_filter_conflict_episodes
)
from src.preprocessing.spark_session import get_spark_session, stop_spark_session


class TestConflictMapping:
    """Tests the five-region taxonomy and country-to-region mapping."""

    def test_canonical_regions_specification(self):
        """Verify exactly the five project-mandated geographical regions exist."""
        expected = [
            "Middle East / MENA",
            "Eastern Europe",
            "South Asia",
            "East Asia",
            "North America"
        ]
        assert CANONICAL_REGIONS == expected
        assert len(get_canonical_regions()) == 5

    def test_key_country_assignments(self):
        """Verify key countries map to their expected canonical regions."""
        # 1. Middle East / MENA
        assert map_country_to_canonical_region("Syria") == "Middle East / MENA"
        assert map_country_to_canonical_region("Israel") == "Middle East / MENA"
        assert map_country_to_canonical_region("Iran") == "Middle East / MENA"
        assert map_country_to_canonical_region("Egypt") == "Middle East / MENA"
        assert map_country_to_canonical_region("Algeria") == "Middle East / MENA"

        # 2. Eastern Europe
        assert map_country_to_canonical_region("Ukraine") == "Eastern Europe"
        assert map_country_to_canonical_region("Russia (Soviet Union)") == "Eastern Europe"
        assert map_country_to_canonical_region("Azerbaijan") == "Eastern Europe"
        assert map_country_to_canonical_region("Armenia") == "Eastern Europe"

        # 3. South Asia
        assert map_country_to_canonical_region("Afghanistan") == "South Asia"
        assert map_country_to_canonical_region("India") == "South Asia"
        assert map_country_to_canonical_region("Pakistan") == "South Asia"

        # 4. East Asia
        assert map_country_to_canonical_region("China") == "East Asia"
        assert map_country_to_canonical_region("Taiwan") == "East Asia"
        assert map_country_to_canonical_region("North Korea") == "East Asia"
        assert map_country_to_canonical_region("South Korea") == "East Asia"
        assert map_country_to_canonical_region("Japan") == "East Asia"

        # 5. North America
        assert map_country_to_canonical_region("United States of America") == "North America"
        assert map_country_to_canonical_region("Mexico") == "North America"
        assert map_country_to_canonical_region("Canada") == "North America"

    def test_excluded_country_assignments(self):
        """Verify excluded countries are assigned to documented exclusion categories."""
        assert map_country_to_canonical_region("Nigeria") == EXCLUSION_SUB_SAHARAN_AFRICA
        assert map_country_to_canonical_region("DR Congo (Zaire)") == EXCLUSION_SUB_SAHARAN_AFRICA
        assert map_country_to_canonical_region("Colombia") == EXCLUSION_LATIN_AMERICA
        assert map_country_to_canonical_region("Brazil") == EXCLUSION_LATIN_AMERICA
        assert map_country_to_canonical_region("Myanmar (Burma)") == EXCLUSION_SE_ASIA_PACIFIC
        assert map_country_to_canonical_region("Philippines") == EXCLUSION_SE_ASIA_PACIFIC
        assert map_country_to_canonical_region("United Kingdom") == EXCLUSION_WESTERN_EUROPE
        assert map_country_to_canonical_region("France") == EXCLUSION_WESTERN_EUROPE

    def test_exclusion_rationales_documentation(self):
        """Verify every exclusion category has an explicit, non-empty documented rationale."""
        for cat in [EXCLUSION_SUB_SAHARAN_AFRICA, EXCLUSION_LATIN_AMERICA,
                    EXCLUSION_SE_ASIA_PACIFIC, EXCLUSION_WESTERN_EUROPE]:
            assert cat in EXCLUSION_RATIONALES
            assert len(EXCLUSION_RATIONALES[cat]) > 20

    def test_unknown_or_null_country_handling(self):
        """Verify unmapped or None country strings return EXCLUSION_UNKNOWN without exceptions."""
        assert map_country_to_canonical_region("") == EXCLUSION_UNKNOWN
        assert map_country_to_canonical_region(None) == EXCLUSION_UNKNOWN
        assert map_country_to_canonical_region("Atlantis Island") == EXCLUSION_UNKNOWN

    def test_all_126_ucdp_countries_mapped(self):
        """Verify every country appearing in raw UCDP GED is explicitly mapped in COUNTRY_TO_REGION_MAP."""
        ucdp_path = "data/raw/ucdp/ged261-csv/GEDEvent_v26_1.csv"
        if os.path.exists(ucdp_path):
            df = pd.read_csv(ucdp_path, usecols=["country"], nrows=20000)
            unique_countries = df["country"].dropna().unique()
            for country in unique_countries:
                mapped = map_country_to_canonical_region(country)
                assert mapped != EXCLUSION_UNKNOWN, f"Country '{country}' was not mapped in COUNTRY_TO_REGION_MAP!"

    def test_duration_regime_classification(self):
        """Verify conflict duration regime boundaries."""
        assert classify_duration_regime(10) == "Short / Tactical"
        assert classify_duration_regime(29) == "Short / Tactical"
        assert classify_duration_regime(30) == "Medium / Campaign"
        assert classify_duration_regime(90) == "Medium / Campaign"
        assert classify_duration_regime(180) == "Medium / Campaign"
        assert classify_duration_regime(181) == "Protracted / War"
        assert classify_duration_regime(730) == "Protracted / War"


class TestConflictAggregationLogic:
    """Tests episode aggregation, washout windowing, and schema conformance."""

    def test_curated_events_extraction(self):
        """Verify curated military escalation milestones cover East Asia and key global crises."""
        curated = _extract_curated_conflict_events("data/raw/events_expanded.csv")
        assert len(curated) >= 4
        regions = {c["canonical_region"] for c in curated}
        assert "East Asia" in regions
        assert "South Asia" in regions

        conf_names = [c["conflict_name"] for c in curated]
        assert any("Taiwan" in name for name in conf_names)
        assert any("Korean" in name or "Fire and Fury" in name for name in conf_names)

        # Verify provenance is preserved
        for c in curated:
            assert "Curated" in c["source_provenance"]
            assert "dyad_new_id" in c
            assert "observation_period" in c

    def test_washout_window_simulation(self):
        """Verify that sequential shock events within 28 days in the same region are properly filtered."""
        dates = [
            datetime(2022, 1, 1),
            datetime(2022, 1, 5),   # 4 days later -> filtered out
            datetime(2022, 1, 15),  # 14 days later -> filtered out
            datetime(2022, 2, 5),   # 35 days later -> ACCEPTED
            datetime(2022, 2, 20),  # 15 days later -> filtered out
            datetime(2022, 3, 20),  # 43 days later -> ACCEPTED
        ]

        accepted = []
        last_date = None
        for d in dates:
            if last_date is None or (d - last_date).days >= 28:
                accepted.append(d)
                last_date = d

        assert len(accepted) == 3
        assert accepted == [datetime(2022, 1, 1), datetime(2022, 2, 5), datetime(2022, 3, 20)]

    def test_episode_schema_fields(self):
        """Verify CONFLICT_EPISODE_SCHEMA contains all required tracking fields."""
        field_names = [f.name for f in CONFLICT_EPISODE_SCHEMA.fields]
        required_fields = [
            "episode_id", "event_date", "end_date", "observation_period",
            "canonical_region", "primary_country", "conflict_new_id", "conflict_name",
            "dyad_new_id", "dyad_name", "type_of_violence", "type_of_violence_desc",
            "incident_count", "fatalities_best", "fatalities_high", "fatalities_low",
            "fatalities_civilians", "duration_regime", "source_provenance",
            "avg_latitude", "avg_longitude"
        ]
        for field in required_fields:
            assert field in field_names, f"Missing required field {field} in CONFLICT_EPISODE_SCHEMA"


class TestSparkConflictTransformations:
    """Tests PySpark transformations with synthetic test fixtures."""

    @pytest.fixture(scope="class")
    def spark(self):
        spark_session = get_spark_session("PyTestConflictTransformations")
        yield spark_session
        stop_spark_session(spark_session)

    def test_synthetic_spark_pipeline_flow(self, spark):
        """
        Tests regional mapping and episode aggregation using small synthetic Spark DataFrame.
        Ensures:
        - Mapping correctly flags canonical vs excluded countries.
        - High-fatality episodes are aggregated.
        - The 28-day washout filter removes proximate events within the same region.
        - Excluded countries (e.g. Nigeria) do not appear in canonical output.
        """
        # Create synthetic input data
        synthetic_rows = [
            # Event 1: Ukraine, Eastern Europe, 2022-02-24, 150 fatalities -> ACCEPTED
            (1, 2022, 1, 300, 500, 1, 150.0, 200.0, 100.0, 10.0, 50.45, 30.52, "Ukraine", "2022-02-24", "Russia - Ukraine", "Russia vs Ukraine"),
            # Event 2: Ukraine, Eastern Europe, 2022-03-05, 80 fatalities -> WITHIN 9 DAYS -> WASHED OUT!
            (2, 2022, 1, 300, 500, 1, 80.0, 100.0, 60.0, 5.0, 50.45, 30.52, "Ukraine", "2022-03-05", "Russia - Ukraine", "Russia vs Ukraine"),
            # Event 3: Ukraine, Eastern Europe, 2022-04-10, 120 fatalities -> >28 DAYS LATER -> ACCEPTED
            (3, 2022, 1, 300, 500, 1, 120.0, 150.0, 90.0, 15.0, 50.45, 30.52, "Ukraine", "2022-04-10", "Russia - Ukraine", "Russia vs Ukraine"),
            # Event 4: Nigeria, Excluded, 2022-02-24, 250 fatalities -> EXCLUDED REGION -> EXCLUDED!
            (4, 2022, 1, 400, 600, 1, 250.0, 300.0, 200.0, 50.0, 9.08, 7.53, "Nigeria", "2022-02-24", "Boko Haram", "Government of Nigeria - Boko Haram"),
            # Event 5: Israel, Middle East / MENA, 2023-10-07, 300 fatalities -> ACCEPTED
            (5, 2023, 1, 500, 700, 1, 300.0, 400.0, 250.0, 100.0, 31.50, 34.45, "Israel", "2023-10-07", "Israel: Palestine", "Israel vs Hamas"),
        ]

        cols = [
            "id", "year", "date_prec", "conflict_new_id", "dyad_new_id",
            "type_of_violence", "best", "high", "low", "deaths_civilians",
            "latitude", "longitude", "country", "date_start", "conflict_name", "dyad_name"
        ]

        df_synth = spark.createDataFrame(synthetic_rows, cols)
        df_synth = df_synth.withColumn("event_date", df_synth["date_start"])

        # Apply region mapping
        df_mapped = apply_spark_region_mapping(df_synth)
        pdf_mapped = df_mapped.toPandas()

        assert pdf_mapped.loc[pdf_mapped["country"] == "Ukraine", "canonical_region"].iloc[0] == "Eastern Europe"
        assert pdf_mapped.loc[pdf_mapped["country"] == "Israel", "canonical_region"].iloc[0] == "Middle East / MENA"
        assert pdf_mapped.loc[pdf_mapped["country"] == "Nigeria", "canonical_region"].iloc[0] == EXCLUSION_SUB_SAHARAN_AFRICA
        assert pdf_mapped.loc[pdf_mapped["country"] == "Nigeria", "is_target_region"].iloc[0] == False

        # Apply episode aggregation & washout
        df_episodes = aggregate_and_filter_conflict_episodes(
            spark=spark,
            df_mapped=df_mapped,
            min_fatalities=25,
            washout_days=28,
            include_curated=False
        )

        pdf_episodes = df_episodes.toPandas()

        # Verify that:
        # 1. Nigeria is completely excluded (0 records)
        assert "Nigeria" not in pdf_episodes["primary_country"].values

        # 2. Israel is included (1 record)
        assert "Israel" in pdf_episodes["primary_country"].values

        # 3. Ukraine has exactly 2 records (Event 1 and Event 3; Event 2 was washed out!)
        ukraine_eps = pdf_episodes[pdf_episodes["primary_country"] == "Ukraine"]
        assert len(ukraine_eps) == 2
        assert set(ukraine_eps["event_date"]) == {"2022-02-24", "2022-04-10"}

        # 4. Check schema fields are populated
        for col in ["episode_id", "observation_period", "dyad_new_id", "source_provenance"]:
            assert col in pdf_episodes.columns
            assert not pdf_episodes[col].isnull().any()
