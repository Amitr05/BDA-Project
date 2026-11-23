"""
Conflict Mapping and Regional Configuration Module
===================================================
Establishes the five-region geographical taxonomy for the Armed Conflicts & Military Escalation ML extension:
  1. Middle East / MENA
  2. Eastern Europe
  3. South Asia
  4. East Asia
  5. North America

Provides:
- Complete country-to-region dictionary covering all 126 countries in UCDP GED v26.1 (1989-2025).
- Explicit categorical assignments for mapped and excluded regions.
- Documented exclusion rationales (Sub-Saharan Africa, Latin America, SE Asia, Western Europe).
- Aggregation, precision, and washout window parameter configurations.
"""

from typing import Dict, List, Optional, Tuple, Set

# The Five Fixed Canonical Geographical Regions
CANONICAL_REGIONS: List[str] = [
    "Middle East / MENA",
    "Eastern Europe",
    "South Asia",
    "East Asia",
    "North America"
]

# Exclusion Reason Categories
EXCLUSION_SUB_SAHARAN_AFRICA = "Excluded (Sub-Saharan Africa)"
EXCLUSION_LATIN_AMERICA = "Excluded (Latin America & Caribbean)"
EXCLUSION_SE_ASIA_PACIFIC = "Excluded (Southeast Asia & Pacific)"
EXCLUSION_CENTRAL_ASIA = "Excluded (Central Asia)"
EXCLUSION_WESTERN_EUROPE = "Excluded (Western/Southern/Northern Europe)"
EXCLUSION_UNKNOWN = "Excluded (Unknown / Unrecognized)"

EXCLUSION_RATIONALES: Dict[str, str] = {
    EXCLUSION_SUB_SAHARAN_AFRICA: (
        "Sub-Saharan African armed conflicts (e.g., DR Congo, Nigeria, Somalia, Mali) exhibit negligible "
        "direct price transmission elasticity to major US broad-market equity indices (S&P 500, NASDAQ, Dow Jones, Russell 2000)."
    ),
    EXCLUSION_LATIN_AMERICA: (
        "South and Central American conflicts (e.g., Colombia, Brazil) are predominantly internal counter-insurgency "
        "or localized organized crime dynamics outside the five project-specified geopolitical theatres."
    ),
    EXCLUSION_SE_ASIA_PACIFIC: (
        "Southeast Asian and Pacific nations (e.g., Myanmar, Philippines, Thailand, Indonesia) represent distinct "
        "ASEAN regional security dynamics, excluded by the project's strict five-region mandate."
    ),
    EXCLUSION_CENTRAL_ASIA: (
        "Former Soviet Central Asian states (e.g., Kyrgyzstan, Tajikistan, Uzbekistan) represent inland Eurasian "
        "security complexes separate from Eastern European and South Asian conflict theatres."
    ),
    EXCLUSION_WESTERN_EUROPE: (
        "Western and Southern European nations (e.g., France, Germany, UK, Spain) experience isolated domestic terrorism "
        "rather than sustained military/state-based armed conflicts under the UCDP definition."
    ),
    EXCLUSION_UNKNOWN: (
        "Unrecognized or unmapped territory names in raw data streams are quarantined to prevent silent assignment."
    )
}

# Complete, Exhaustive Country Mapping for all 126 UCDP Countries + Global Standard Aliases
COUNTRY_TO_REGION_MAP: Dict[str, str] = {
    # -------------------------------------------------------------------------
    # 1. Middle East / MENA (19 Countries)
    # -------------------------------------------------------------------------
    "Algeria": "Middle East / MENA",
    "Bahrain": "Middle East / MENA",
    "Egypt": "Middle East / MENA",
    "Iran": "Middle East / MENA",
    "Iraq": "Middle East / MENA",
    "Israel": "Middle East / MENA",
    "Jordan": "Middle East / MENA",
    "Kuwait": "Middle East / MENA",
    "Lebanon": "Middle East / MENA",
    "Libya": "Middle East / MENA",
    "Morocco": "Middle East / MENA",
    "Oman": "Middle East / MENA",
    "Qatar": "Middle East / MENA",
    "Saudi Arabia": "Middle East / MENA",
    "Syria": "Middle East / MENA",
    "Tunisia": "Middle East / MENA",
    "Turkey": "Middle East / MENA",
    "United Arab Emirates": "Middle East / MENA",
    "Yemen (North Yemen)": "Middle East / MENA",
    "Yemen": "Middle East / MENA",
    "Palestine": "Middle East / MENA",

    # -------------------------------------------------------------------------
    # 2. Eastern Europe (14 Countries)
    # -------------------------------------------------------------------------
    "Albania": "Eastern Europe",
    "Armenia": "Eastern Europe",
    "Azerbaijan": "Eastern Europe",
    "Belarus": "Eastern Europe",
    "Bosnia-Herzegovina": "Eastern Europe",
    "Croatia": "Eastern Europe",
    "Georgia": "Eastern Europe",
    "Moldova": "Eastern Europe",
    "North Macedonia": "Eastern Europe",
    "Poland": "Eastern Europe",
    "Romania": "Eastern Europe",
    "Russia (Soviet Union)": "Eastern Europe",
    "Russia": "Eastern Europe",
    "Serbia (Yugoslavia)": "Eastern Europe",
    "Serbia": "Eastern Europe",
    "Ukraine": "Eastern Europe",

    # -------------------------------------------------------------------------
    # 3. South Asia (8 Countries)
    # -------------------------------------------------------------------------
    "Afghanistan": "South Asia",
    "Bangladesh": "South Asia",
    "Bhutan": "South Asia",
    "India": "South Asia",
    "Maldives": "South Asia",
    "Nepal": "South Asia",
    "Pakistan": "South Asia",
    "Sri Lanka": "South Asia",

    # -------------------------------------------------------------------------
    # 4. East Asia (6 Countries)
    # -------------------------------------------------------------------------
    "China": "East Asia",
    "Japan": "East Asia",
    "North Korea": "East Asia",
    "South Korea": "East Asia",
    "Taiwan": "East Asia",
    "Mongolia": "East Asia",

    # -------------------------------------------------------------------------
    # 5. North America (3 Countries)
    # -------------------------------------------------------------------------
    "Canada": "North America",
    "Mexico": "North America",
    "United States of America": "North America",
    "United States": "North America",
    "USA": "North America",

    # -------------------------------------------------------------------------
    # EXCLUDED: Sub-Saharan Africa (42 Countries)
    # -------------------------------------------------------------------------
    "Angola": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Benin": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Botswana": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Burkina Faso": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Burundi": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Cameroon": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Central African Republic": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Chad": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Comoros": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Congo": EXCLUSION_SUB_SAHARAN_AFRICA,
    "DR Congo (Zaire)": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Djibouti": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Eritrea": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Ethiopia": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Gambia": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Ghana": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Guinea": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Guinea-Bissau": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Ivory Coast": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Kenya": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Kingdom of eSwatini (Swaziland)": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Lesotho": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Liberia": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Madagascar (Malagasy)": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Mali": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Mauritania": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Mozambique": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Namibia": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Niger": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Nigeria": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Rwanda": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Senegal": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Sierra Leone": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Somalia": EXCLUSION_SUB_SAHARAN_AFRICA,
    "South Africa": EXCLUSION_SUB_SAHARAN_AFRICA,
    "South Sudan": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Sudan": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Tanzania": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Togo": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Uganda": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Zambia": EXCLUSION_SUB_SAHARAN_AFRICA,
    "Zimbabwe (Rhodesia)": EXCLUSION_SUB_SAHARAN_AFRICA,

    # -------------------------------------------------------------------------
    # EXCLUDED: Latin America & Caribbean (18 Countries)
    # -------------------------------------------------------------------------
    "Argentina": EXCLUSION_LATIN_AMERICA,
    "Bolivia": EXCLUSION_LATIN_AMERICA,
    "Brazil": EXCLUSION_LATIN_AMERICA,
    "Colombia": EXCLUSION_LATIN_AMERICA,
    "Costa Rica": EXCLUSION_LATIN_AMERICA,
    "Ecuador": EXCLUSION_LATIN_AMERICA,
    "El Salvador": EXCLUSION_LATIN_AMERICA,
    "Guatemala": EXCLUSION_LATIN_AMERICA,
    "Guyana": EXCLUSION_LATIN_AMERICA,
    "Haiti": EXCLUSION_LATIN_AMERICA,
    "Honduras": EXCLUSION_LATIN_AMERICA,
    "Jamaica": EXCLUSION_LATIN_AMERICA,
    "Nicaragua": EXCLUSION_LATIN_AMERICA,
    "Panama": EXCLUSION_LATIN_AMERICA,
    "Paraguay": EXCLUSION_LATIN_AMERICA,
    "Peru": EXCLUSION_LATIN_AMERICA,
    "Trinidad and Tobago": EXCLUSION_LATIN_AMERICA,
    "Venezuela": EXCLUSION_LATIN_AMERICA,

    # -------------------------------------------------------------------------
    # EXCLUDED: Southeast Asia & Pacific (10 Countries)
    # -------------------------------------------------------------------------
    "Australia": EXCLUSION_SE_ASIA_PACIFIC,
    "Cambodia (Kampuchea)": EXCLUSION_SE_ASIA_PACIFIC,
    "Indonesia": EXCLUSION_SE_ASIA_PACIFIC,
    "Laos": EXCLUSION_SE_ASIA_PACIFIC,
    "Malaysia": EXCLUSION_SE_ASIA_PACIFIC,
    "Myanmar (Burma)": EXCLUSION_SE_ASIA_PACIFIC,
    "Papua New Guinea": EXCLUSION_SE_ASIA_PACIFIC,
    "Philippines": EXCLUSION_SE_ASIA_PACIFIC,
    "Solomon Islands": EXCLUSION_SE_ASIA_PACIFIC,
    "Thailand": EXCLUSION_SE_ASIA_PACIFIC,

    # -------------------------------------------------------------------------
    # EXCLUDED: Central Asia (3 Countries)
    # -------------------------------------------------------------------------
    "Kyrgyzstan": EXCLUSION_CENTRAL_ASIA,
    "Tajikistan": EXCLUSION_CENTRAL_ASIA,
    "Uzbekistan": EXCLUSION_CENTRAL_ASIA,

    # -------------------------------------------------------------------------
    # EXCLUDED: Western / Northern / Southern Europe (11 Countries)
    # -------------------------------------------------------------------------
    "Austria": EXCLUSION_WESTERN_EUROPE,
    "Belgium": EXCLUSION_WESTERN_EUROPE,
    "Cyprus": EXCLUSION_WESTERN_EUROPE,
    "France": EXCLUSION_WESTERN_EUROPE,
    "Germany": EXCLUSION_WESTERN_EUROPE,
    "Italy": EXCLUSION_WESTERN_EUROPE,
    "Malta": EXCLUSION_WESTERN_EUROPE,
    "Netherlands": EXCLUSION_WESTERN_EUROPE,
    "Spain": EXCLUSION_WESTERN_EUROPE,
    "Sweden": EXCLUSION_WESTERN_EUROPE,
    "United Kingdom": EXCLUSION_WESTERN_EUROPE,
}

# Aggregation and Filtering Configuration Parameters
AGGREGATION_CONFIG: Dict[str, any] = {
    "min_event_year": 2015,
    "max_event_year": 2025,
    "max_date_precision": 2,  # 1 = exact day, 2 = 2-6 day window
    "min_fatalities_escalation": 25,  # UCDP standard annual battle death threshold
    "min_fatalities_major_escalation": 50,
    "washout_window_calendar_days": 28,  # Equates to 20 trading days
    "washout_window_trading_days": 20,
    "forward_return_horizons_trading_days": [5, 20],
    "duration_regimes": {
        "Short / Tactical": {"min_days": 1, "max_days": 29, "description": "Tactical skirmishes, airstrikes, brief border clashes"},
        "Medium / Campaign": {"min_days": 30, "max_days": 180, "description": "Structured operational campaigns, seasonal military offensives"},
        "Protracted / War": {"min_days": 181, "max_days": 9999, "description": "Full-scale invasions, sustained multi-theatre inter-state warfare"}
    }
}


def map_country_to_canonical_region(country: str) -> str:
    """
    Deterministically maps a country name to one of the 5 canonical regions,
    or returns an explicit documented Excluded category.
    Never silently assigns or defaults to an arbitrary region.
    """
    if not country or not isinstance(country, str):
        return EXCLUSION_UNKNOWN
    c_clean = country.strip()
    return COUNTRY_TO_REGION_MAP.get(c_clean, EXCLUSION_UNKNOWN)


def is_canonical_region(region: str) -> bool:
    """Returns True if the region belongs to the 5 canonical project categories."""
    return region in CANONICAL_REGIONS


def get_canonical_regions() -> List[str]:
    """Returns list of the 5 canonical project regions."""
    return list(CANONICAL_REGIONS)


def get_countries_for_region(region: str) -> List[str]:
    """Returns sorted list of countries mapped to a specific canonical region."""
    if not is_canonical_region(region):
        return []
    return sorted([c for c, r in COUNTRY_TO_REGION_MAP.items() if r == region])


def classify_duration_regime(duration_days: float) -> str:
    """Classifies numeric conflict duration (in days) into one of the 3 operational regimes."""
    if duration_days < 30:
        return "Short / Tactical"
    elif duration_days <= 180:
        return "Medium / Campaign"
    else:
        return "Protracted / War"
