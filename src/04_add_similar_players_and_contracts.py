from pathlib import Path
import time

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

BASE_INPUT_PATH = PROCESSED_DIR / "harden_bucks_guard_comparison.csv"
CONTRACT_INPUT_PATH = PROCESSED_DIR / "creator_guard_contracts.csv"

SIMILAR_OUTPUT_PATH = PROCESSED_DIR / "harden_similar_creator_guards.csv"
ENRICHED_OUTPUT_PATH = PROCESSED_DIR / "harden_bucks_dashboard_enriched.csv"

SEASON = "2025-26"
MIN_GP = 10
MIN_MINUTES = 24
REQUEST_TIMEOUT = 90
REQUEST_DELAY_SECONDS = 2.0


TARGET_COMPARISON_PLAYERS = [
    "James Harden",
    "Damian Lillard",
    "Jrue Holiday",
    "Trae Young",
    "Jalen Brunson",
    "Tyrese Haliburton",
    "Ja Morant",
    "LaMelo Ball",
    "Darius Garland",
    "Kyrie Irving",
    "Fred VanVleet",
    "Kevin Porter Jr.",
    "Ryan Rollins",
    "Gary Trent Jr.",
    "AJ Green",
]


def normalize_name(value):
    text = str(value).strip().lower()
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("-", " ")
    text = " ".join(text.split())
    return text


def fetch_player_stats(season, measure_type):
    cache_path = RAW_DIR / f"similar_guard_{measure_type.lower()}_{season}.csv"

    if cache_path.exists():
        return pd.read_csv(cache_path)

    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense=measure_type,
        timeout=REQUEST_TIMEOUT,
    ).get_data_frames()[0]

    stats["SEASON"] = season
    stats.to_csv(cache_path, index=False)

    time.sleep(REQUEST_DELAY_SECONDS)

    return stats


def clean_base_stats(base):
    wanted_columns = [
        "SEASON",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "AGE",
        "GP",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "TOV",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "PLUS_MINUS",
    ]

    existing = [column for column in wanted_columns if column in base.columns]
    return base[existing].copy()


def clean_advanced_stats(advanced):
    wanted_columns = [
        "SEASON",
        "PLAYER_ID",
        "TEAM_ID",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "AST_PCT",
        "AST_TO",
        "AST_RATIO",
        "TM_TOV_PCT",
        "EFG_PCT",
        "TS_PCT",
        "USG_PCT",
        "PACE",
        "PIE",
    ]

    existing = [column for column in wanted_columns if column in advanced.columns]
    return advanced[existing].copy()


def build_creation_metrics(df):
    df = df.copy()

    numeric_columns = [
        "GP",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "TOV",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "PLUS_MINUS",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "AST_PCT",
        "AST_TO",
        "AST_RATIO",
        "TM_TOV_PCT",
        "EFG_PCT",
        "TS_PCT",
        "USG_PCT",
        "PACE",
        "PIE",
    ]

    for column in numeric_columns:
        if column not in df.columns:
            df[column] = np.nan

        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["AST_TOV_CALC"] = np.where(
        df["TOV"] > 0,
        df["AST"] / df["TOV"],
        np.nan,
    )

    df["playmaking_value_index"] = (
        df["AST"].fillna(0) * 3.0
        + df["AST_PCT"].fillna(0) * 40.0
        + df["AST_TO"].fillna(0) * 1.5
        - df["TOV"].fillna(0) * 1.5
    )

    df["scoring_efficiency_index"] = (
        df["PTS"].fillna(0) * 0.8
        + df["TS_PCT"].fillna(0) * 35.0
        + df["FG3_PCT"].fillna(0) * 12.0
    )

    df["half_court_creator_index"] = (
        df["playmaking_value_index"] * 0.55
        + df["scoring_efficiency_index"] * 0.35
        + df["OFF_RATING"].fillna(0) * 0.10
    )

    df["turnover_risk_index"] = (
        df["TOV"].fillna(0) * 2.0
        + df["TM_TOV_PCT"].fillna(0) * 0.35
    )

    df["performance_argument_score"] = (
        df["half_court_creator_index"]
        - df["turnover_risk_index"] * 0.35
    )

    return df


def add_similarity_to_harden(df):
    df = df.copy()

    harden_rows = df[df["PLAYER_NAME"] == "James Harden"]

    if harden_rows.empty:
        raise ValueError("James Harden missing from similar guard dataset.")

    harden = harden_rows.iloc[0]

    features = [
        "PTS",
        "AST",
        "AST_PCT",
        "USG_PCT",
        "TS_PCT",
        "OFF_RATING",
        "AST_TO",
        "performance_argument_score",
    ]

    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    for feature in features:
        std = df[feature].std()

        if std == 0 or pd.isna(std):
            df[f"{feature}_z"] = 0
            harden_z = 0
        else:
            mean = df[feature].mean()
            df[f"{feature}_z"] = (df[feature] - mean) / std
            harden_z = (harden[feature] - mean) / std

        df[f"{feature}_distance_from_harden"] = df[f"{feature}_z"] - harden_z

    distance_columns = [f"{feature}_distance_from_harden" for feature in features]

    df["harden_similarity_distance"] = np.sqrt(
        np.square(df[distance_columns]).sum(axis=1)
    )

    max_distance = df["harden_similarity_distance"].max()

    if max_distance == 0:
        df["harden_similarity_score"] = 100
    else:
        df["harden_similarity_score"] = (
            100
            * (1 - df["harden_similarity_distance"] / max_distance)
        )

    return df


def add_contracts(df, contracts):
    df = df.copy()
    contracts = contracts.copy()

    df["contract_player_key"] = df["PLAYER_NAME"].apply(normalize_name)
    contracts["contract_player_key"] = contracts["player_name"].apply(normalize_name)

    df = df.merge(
        contracts,
        on="contract_player_key",
        how="left",
    )

    df["aav_millions"] = df["aav"] / 1_000_000

    df["performance_score_per_1m_aav"] = np.where(
        df["aav_millions"] > 0,
        df["performance_argument_score"] / df["aav_millions"],
        np.nan,
    )

    df["assist_per_1m_aav"] = np.where(
        df["aav_millions"] > 0,
        df["AST"] / df["aav_millions"],
        np.nan,
    )

    return df


def classify_contract_tier(row):
    aav = row.get("aav", np.nan)

    if pd.isna(aav):
        return "Contract Unknown"

    if aav >= 40_000_000:
        return "Max / Near-Max Creator"

    if aav >= 25_000_000:
        return "Premium Veteran Guard"

    if aav >= 10_000_000:
        return "Mid-Tier Rotation Guard"

    return "Low-Cost Guard"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    base = fetch_player_stats(SEASON, "Base")
    advanced = fetch_player_stats(SEASON, "Advanced")

    base = clean_base_stats(base)
    advanced = clean_advanced_stats(advanced)

    league = base.merge(
        advanced,
        on=["SEASON", "PLAYER_ID", "TEAM_ID"],
        how="left",
    )

    league = build_creation_metrics(league)

    league = league[
        (league["GP"] >= MIN_GP)
        & (league["MIN"] >= MIN_MINUTES)
    ].copy()

    league = league[league["PLAYER_NAME"].isin(TARGET_COMPARISON_PLAYERS)].copy()

    if league.empty:
        raise ValueError("No target comparison players found in league dataset.")

    league = add_similarity_to_harden(league)

    contracts = pd.read_csv(CONTRACT_INPUT_PATH)

    similar = add_contracts(league, contracts)

    similar["contract_tier"] = similar.apply(classify_contract_tier, axis=1)

    similar = similar.sort_values(
        by="harden_similarity_score",
        ascending=False,
    )

    similar.to_csv(SIMILAR_OUTPUT_PATH, index=False)

    base_dashboard = pd.read_csv(BASE_INPUT_PATH)
    base_dashboard["data_layer"] = "Harden vs Bucks Historical Guard Comparison"

    similar["data_layer"] = "Current Similar Creator Guard Contract Comparison"

    similar.to_csv(ENRICHED_OUTPUT_PATH, index=False)

    print("")
    print("SIMILAR PLAYERS + CONTRACTS COMPLETE")
    print("------------------------------------")
    print(f"Season: {SEASON}")
    print(f"Rows: {len(similar)}")
    print(f"Output 1: {SIMILAR_OUTPUT_PATH}")
    print(f"Output 2: {ENRICHED_OUTPUT_PATH}")
    print("")
    print("Similar creator guards with contract layer:")
    print(
        similar[
            [
                "PLAYER_NAME",
                "TEAM_ABBREVIATION",
                "PTS",
                "AST",
                "TOV",
                "AST_PCT",
                "USG_PCT",
                "TS_PCT",
                "OFF_RATING",
                "performance_argument_score",
                "harden_similarity_score",
                "aav",
                "contract_tier",
                "performance_score_per_1m_aav",
            ]
        ]
        .to_string(index=False)
    )
    print("")


if __name__ == "__main__":
    main()