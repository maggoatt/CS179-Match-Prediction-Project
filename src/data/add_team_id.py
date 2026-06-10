"""
A module for adding team IDs to historical Premier League match data.
"""
from pathlib import Path
import json

import pandas as pd

from src.config import PROCESSED_DATA_DIR, TRAIN_SEASONS, UNKNOWN_TEAM_ID


def build_team_to_id_mapping(df: pd.DataFrame, train_seasons: list[str]) -> dict[str, int]:
    """
    Constructs a mapping from team names (of teams in the specified seasons) to unique IDs.
    
    :param df: A DataFrame object storing historical Premier League match data.
    :param train_seasons: The seasons of historical Premier League match data to use for building the team mapping.
    :return: A mapping from team names (of teams in the specified seasons) to unique IDs.
    """
    train_df = df[df["Season"].isin(train_seasons)]
    unique_teams = sorted(set(train_df["HomeTeam"].unique()) | set(train_df["AwayTeam"].unique()))
    return {team_name: idx for idx, team_name in enumerate(unique_teams)}


def add_team_ids(df: pd.DataFrame, team_to_id_mapping: dict[str, int]) -> pd.DataFrame:
    """
    Inserts team IDs (from the specified mapping) into historical Premier League match data.
    
    Teams with unknown IDs are assigned ID src.config.UNKNOWN_TEAM_ID.
    
    :param df: A DataFrame object storing historical Premier League match data.
    :param team_to_id_mapping: A mapping from team names to unique IDs.
    :return: A DataFrame object storing historical Premier League match data with team IDs inserted.
    """
    
    df["HomeTeamID"] = df["HomeTeam"].map(team_to_id_mapping).fillna(UNKNOWN_TEAM_ID).astype(int)
    df["AwayTeamID"] = df["AwayTeam"].map(team_to_id_mapping).fillna(UNKNOWN_TEAM_ID).astype(int)
    df = df[["Season", "Date", "HomeTeamID", "AwayTeamID", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "Result"]]
    unknown_home_teams = df.loc[df["HomeTeamID"] == UNKNOWN_TEAM_ID, "HomeTeam"].unique()
    unknown_away_teams = df.loc[df["AwayTeamID"] == UNKNOWN_TEAM_ID, "AwayTeam"].unique()
    unknown_teams = sorted(set(unknown_home_teams) | set(unknown_away_teams))
    if unknown_teams:
        print(f"Warning: {len(unknown_teams)} team(s) ({unknown_teams}) not in ID mapping (assigned ID {UNKNOWN_TEAM_ID}).")
    return df


def save_team_to_id_mapping(team_to_id_mapping: dict, output_directory: Path) -> None:
    """
    Saves a mapping (from team names to unique IDs) to the specified output directory.
    
    :param team_to_id_mapping: The mapping from team names to unique IDs to save.
    :param output_directory: The output directory to save the mapping to.
    :return: None.
    """
    with open(output_directory / "team_mapping.json", "w") as file:
        json.dump(team_to_id_mapping, file,
                  indent = 2, sort_keys = True)


def main():
    df = pd.read_csv(PROCESSED_DATA_DIR / "epl_matches.csv", parse_dates = ["Date"])
    team_mapping = build_team_to_id_mapping(df, TRAIN_SEASONS)
    print(f"Built ID mapping for {len(team_mapping)} teams (in seasons {TRAIN_SEASONS}).")
    df = add_team_ids(df, team_mapping)
    df.to_csv(PROCESSED_DATA_DIR / "epl_matches.csv", index = False)
    save_team_to_id_mapping(team_mapping, PROCESSED_DATA_DIR)
    print("Updated epl_matches.csv with team IDs.")


if __name__ == "__main__":
    main()
