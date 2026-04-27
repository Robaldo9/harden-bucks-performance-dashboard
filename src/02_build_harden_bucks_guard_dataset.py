from pathlib import Path
import time

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import commonteamroster, leaguedashplayerstats
from nba_api.stats.static import players, teams


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "harden_bucks_guard_comparison.csv"
SUMMARY_OUTPUT_PATH = PROCESSED_DIR / "harden_bucks_guard_summary.csv"

BUCKS_TEAM_ID = 1610612749
HARDEN_PLAYER_ID = 201935

SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

MIN_GP = 10
REQUEST_DELAY_SECONDS = 2.0
MAX_RETRIES = 4
REQUEST_TIMEOUT = 90


FALLBACK_BUCKS_GUARD_NAMES = {
    "2019-20": [
        "Eric Bledsoe",
        "George Hill",
        "Donte DiVincenzo",
        "Wesley Matthews",
        "Pat Connaughton",
        "Kyle Korver",
        "Sterling Brown",
    ],
    "2020-21": [
        "Jrue Holiday",
        "Donte DiVincenzo",
        "Bryn Forbes",
        "Pat Connaughton",
        "Jeff Teague",
        "Sam Merrill",
    ],
    "2021-22": [
        "Jrue Holiday",
        "Grayson Allen",
        "Pat Connaughton",
        "George Hill",
        "Donte DiVincenzo",
        "Jevon Carter",
        "Wesley Matthews",
    ],
    "2022-23": [
        "Jrue Holiday",
        "Grayson Allen",
        "Pat Connaughton",
        "Jevon Carter",
        "George Hill",
        "Wesley Matthews",
        "AJ Green",
    ],
    "2023-24": [
        "Damian Lillard",
        "Malik Beasley",
        "Patrick Beverley",
        "Pat Connaughton",
        "AJ Green",
        "Cameron Payne",
        "Andre Jackson Jr.",
        "MarJon Beauchamp",
    ],
    "2024-25": [
        "Damian Lillard",
        "Gary Trent Jr.",
        "AJ Green",
        "Pat Connaughton",
        "Kevin Porter Jr.",
        "Ryan Rollins",
        "Andre Jackson Jr.",
        "Delon Wright",
    ],
    "2025-26": [
        "Gary Trent Jr.",
        "AJ Green",
        "Kevin Porter Jr.",
        "Ryan Rollins",
        "Andre Jackson Jr.",
        "Pat Connaughton",
        "Delon Wright",
    ],
}


def safe_request(function_name, cache_path, request_callable):
    if cache_path.exists():
        return pd.read_csv(cache_path)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Fetching {function_name} | attempt {attempt}/{MAX_RETRIES}")
            df = request_callable()
            df.to_csv(cache_path, index=False)
            time.sleep(REQUEST_DELAY_SECONDS)
            return df

        except Exception as error:
            last_error = error
            print(f"  Request failed: {error}")
            sleep_time = REQUEST_DELAY_SECONDS * attempt
            print(f"  Waiting {sleep_time:.1f}s before retry...")
            time.sleep(sleep_time)

    raise RuntimeError(f"Failed request after retries: {function_name}\n{last_error}")


def get_harden_id():
    harden = players.find_players_by_full_name("James Harden")

    if not harden:
        raise ValueError("James Harden not found in NBA static players.")

    return harden[0]["id"]


def get_bucks_id():
    bucks = teams.find_teams_by_full_name("Milwaukee Bucks")

    if not bucks:
        raise ValueError("Milwaukee Bucks not found in NBA static teams.")

    return bucks[0]["id"]


def fetch_bucks_roster(season):
    cache_path = RAW_DIR / f"bucks_roster_{season}.csv"

    def request_callable():
        return commonteamroster.CommonTeamRoster(
            team_id=BUCKS_TEAM_ID,
            season=season,
            timeout=REQUEST_TIMEOUT,
        ).get_data_frames()[0]

    roster = safe_request(
        function_name=f"Bucks roster {season}",
        cache_path=cache_path,
        request_callable=request_callable,
    )

    roster["SEASON"] = season

    return roster


def fetch_player_stats(season, measure_type):
    cache_path = RAW_DIR / f"league_player_stats_{measure_type.lower()}_{season}.csv"

    def request_callable():
        return leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            measure_type_detailed_defense=measure_type,
            timeout=REQUEST_TIMEOUT,
        ).get_data_frames()[0]

    stats = safe_request(
        function_name=f"LeagueDashPlayerStats {measure_type} {season}",
        cache_path=cache_path,
        request_callable=request_callable,
    )

    stats["SEASON"] = season

    return stats


def is_guard_position(position):
    position = str(position).upper()
    return "G" in position


def normalize_name(value):
    text = str(value).strip().lower()
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("-", " ")
    text = " ".join(text.split())
    return text


def get_bucks_guard_ids_from_roster(roster):
    if "POSITION" not in roster.columns or "PLAYER_ID" not in roster.columns:
        return set()

    guards = roster[roster["POSITION"].apply(is_guard_position)].copy()

    return set(guards["PLAYER_ID"].astype(int).tolist())


def get_bucks_guard_names_fallback(season):
    return set(normalize_name(name) for name in FALLBACK_BUCKS_GUARD_NAMES.get(season, []))


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
        "PLUS_MINUS",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
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
        "PLUS_MINUS",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
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


def classify_player_group(row):
    if int(row["PLAYER_ID"]) == HARDEN_PLAYER_ID:
        return "James Harden"

    return "Milwaukee Bucks Guard"


def validate_output(df):
    required_columns = [
        "SEASON",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "GP",
        "MIN",
        "PTS",
        "AST",
        "TOV",
        "FG3_PCT",
        "TS_PCT",
        "USG_PCT",
        "AST_PCT",
        "OFF_RATING",
        "playmaking_value_index",
        "half_court_creator_index",
        "performance_argument_score",
        "player_group",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing output columns: {missing}")

    if df.empty:
        raise ValueError("Output dataset is empty.")

    if "James Harden" not in df["player_group"].unique():
        raise ValueError("James Harden rows missing from output.")


def main():
    harden_id = get_harden_id()
    bucks_id = get_bucks_id()

    if harden_id != HARDEN_PLAYER_ID:
        raise ValueError(f"Unexpected Harden ID: {harden_id}")

    if bucks_id != BUCKS_TEAM_ID:
        raise ValueError(f"Unexpected Bucks ID: {bucks_id}")

    all_rows = []

    print("")
    print("BUILDING HARDEN VS BUCKS GUARD DATASET")
    print("--------------------------------------")
    print("Using retry, timeout, and local cache.")
    print("")

    for season in SEASONS:
        print(f"Processing season: {season}")

        roster = fetch_bucks_roster(season)
        bucks_guard_ids = get_bucks_guard_ids_from_roster(roster)
        fallback_guard_names = get_bucks_guard_names_fallback(season)

        base = fetch_player_stats(season, "Base")
        advanced = fetch_player_stats(season, "Advanced")

        base = clean_base_stats(base)
        advanced = clean_advanced_stats(advanced)

        merged = base.merge(
            advanced,
            on=["SEASON", "PLAYER_ID", "TEAM_ID"],
            how="left",
        )

        merged["normalized_player_name"] = merged["PLAYER_NAME"].apply(normalize_name)

        is_harden = merged["PLAYER_ID"].astype(int) == HARDEN_PLAYER_ID

        is_bucks_guard_by_roster = merged["PLAYER_ID"].astype(int).isin(bucks_guard_ids)

        is_bucks_guard_by_name = (
            (merged["TEAM_ID"].astype(int) == BUCKS_TEAM_ID)
            & (merged["normalized_player_name"].isin(fallback_guard_names))
        )

        season_df = merged[
            is_harden
            | is_bucks_guard_by_roster
            | is_bucks_guard_by_name
        ].copy()

        season_df = season_df[season_df["GP"] >= MIN_GP].copy()

        season_df["player_group"] = season_df.apply(
            classify_player_group,
            axis=1,
        )

        season_df["bucks_guard_position"] = np.where(
            season_df["player_group"] == "Milwaukee Bucks Guard",
            "Bucks Guard",
            "External Free Agent Target",
        )

        season_df["roster_guard_source"] = np.where(
            season_df["PLAYER_ID"].astype(int).isin(bucks_guard_ids),
            "Roster Position",
            np.where(
                season_df["normalized_player_name"].isin(fallback_guard_names),
                "Fallback Guard List",
                "Harden Target",
            ),
        )

        all_rows.append(season_df)

        print(f"  Rows kept for {season}: {len(season_df)}")

    final_df = pd.concat(all_rows, ignore_index=True)

    final_df = build_creation_metrics(final_df)

    final_df = final_df.sort_values(
        by=["SEASON", "performance_argument_score"],
        ascending=[True, False],
    )

    validate_output(final_df)

    final_df.to_csv(OUTPUT_PATH, index=False)

    summary = (
        final_df.groupby("player_group")
        .agg(
            seasons=("SEASON", "nunique"),
            player_seasons=("PLAYER_ID", "count"),
            avg_pts=("PTS", "mean"),
            avg_ast=("AST", "mean"),
            avg_tov=("TOV", "mean"),
            avg_ast_pct=("AST_PCT", "mean"),
            avg_ast_to=("AST_TO", "mean"),
            avg_ts_pct=("TS_PCT", "mean"),
            avg_off_rating=("OFF_RATING", "mean"),
            avg_half_court_creator_index=("half_court_creator_index", "mean"),
            avg_performance_argument_score=("performance_argument_score", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print("")
    print("DATASET COMPLETE")
    print("----------------")
    print(f"Rows: {len(final_df)}")
    print(f"Seasons: {final_df['SEASON'].nunique()}")
    print(f"Players: {final_df['PLAYER_NAME'].nunique()}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_OUTPUT_PATH}")
    print("")
    print("Summary by group:")
    print(summary.to_string(index=False))
    print("")
    print("Top 20 player-seasons by Performance Argument Score:")
    print(
        final_df[
            [
                "SEASON",
                "PLAYER_NAME",
                "TEAM_ABBREVIATION",
                "player_group",
                "GP",
                "MIN",
                "PTS",
                "AST",
                "TOV",
                "AST_PCT",
                "AST_TO",
                "TS_PCT",
                "OFF_RATING",
                "performance_argument_score",
            ]
        ]
        .sort_values("performance_argument_score", ascending=False)
        .head(20)
        .to_string(index=False)
    )
    print("")


if __name__ == "__main__":
    main()