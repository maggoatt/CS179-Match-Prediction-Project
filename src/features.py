import json
import pandas as pd
from pathlib import Path


def build_team_mapping(df: pd.DataFrame, train_seasons: list[str]) -> dict[str, int]:
    train_df = df[df["season"].isin(train_seasons)]
    teams = sorted(
        set(train_df["HomeTeam"].unique()) | set(train_df["AwayTeam"].unique())
    )
    return {name: idx for idx, name in enumerate(teams)}


def add_team_ids(df: pd.DataFrame, team_mapping: dict[str, int]) -> pd.DataFrame:
    from src.config import UNKNOWN_TEAM_ID
    df["home_team_id"] = df["HomeTeam"].map(team_mapping).fillna(UNKNOWN_TEAM_ID).astype(int)
    df["away_team_id"] = df["AwayTeam"].map(team_mapping).fillna(UNKNOWN_TEAM_ID).astype(int)
    unknown_home = df.loc[df["home_team_id"] == UNKNOWN_TEAM_ID, "HomeTeam"].unique()
    unknown_away = df.loc[df["away_team_id"] == UNKNOWN_TEAM_ID, "AwayTeam"].unique()
    unknown = sorted(set(unknown_home) | set(unknown_away))
    if unknown:
        print(f"Warning: {len(unknown)} team(s) not in training mapping (assigned {UNKNOWN_TEAM_ID}): {unknown}")
    return df


def save_team_mapping(team_mapping: dict, output_dir: Path) -> None:
    with open(output_dir / "team_mapping.json", "w") as f:
        json.dump(team_mapping, f, indent=2, sort_keys=True)


def main():
    from src.config import PROCESSED_DATA_DIR, TRAIN_SEASONS
    df = pd.read_csv(PROCESSED_DATA_DIR / "epl_matches.csv", parse_dates=["Date"])
    team_mapping = build_team_mapping(df, TRAIN_SEASONS)
    print(f"Built mapping for {len(team_mapping)} training teams.")
    df = add_team_ids(df, team_mapping)
    df.to_csv(PROCESSED_DATA_DIR / "epl_matches.csv", index=False)
    save_team_mapping(team_mapping, PROCESSED_DATA_DIR)
    print("Updated epl_matches.csv with team IDs.")


if __name__ == "__main__":
    main()
