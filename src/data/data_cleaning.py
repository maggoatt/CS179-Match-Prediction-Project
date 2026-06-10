"""
A module for processing raw historical Premier League match data.
"""
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, ALL_SEASONS


DF_REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season"]
DF_RENAMED_COLUMNS = {
    "FTHG": "HomeGoals",
    "FTAG": "AwayGoals",
    "FTR": "Result",
}
TEAM_NAME_MAP = {
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR",
    "Sheffield United": "Sheffield Utd",
    "Nottingham Forest": "Nott'm Forest",
}
VALID_MATCH_RESULTS = {"H", "D", "A"}


def load_raw_seasons(raw_directory: Path, seasons: list[str]) -> pd.DataFrame:
    """
    Loads raw historical Premier League match data into a single DataFrame object.
    
    :param raw_directory: The directory storing raw historical Premier League match data.
    :param seasons: The seasons of historical Premier League match data to load.
    :return: A DataFrame object storing raw historical Premier League match data.
    """
    dfs = []
    for season in seasons:
        path = raw_directory / f"{season}.csv"
        if not path.exists():
            print(f"Warning: season {season} data not found at {path}, skipping.")
            continue
        df = pd.read_csv(path, encoding = "utf-8")
        df["Season"] = season
        dfs.append(df)
    return pd.concat(dfs, ignore_index = True)


def process_columns(df: pd.DataFrame)  -> pd.DataFrame:
    """
    Processes the columns of historical Premier League match data.
    
    :param df: A DataFrame object storing raw historical Premier League match data.
    :return: A new DataFrame object storing historical Premier League match data with processed columns.
    """
    df = keep_required_columns(df)
    df = rename_columns(df)
    df = reorder_columns(df)
    return df


def keep_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps the columns of historical Premier League match data required for match outcome prediction.
    
    :param df: A DataFrame object storing raw historical Premier League match data.
    :return: A new DataFrame object storing columns of historical Premier League match data required for match outcome prediction.
    """
    return df[[column
               for column in DF_REQUIRED_COLUMNS
               if column in df.columns]].copy()


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the columns of historical Premier League match data for visual clarity.
    
    :param df: A DataFrame object storing historical Premier League match data.
    :return: A new DataFrame object storing historical Premier League match data with renamed columns for visual clarity.
    """
    return df.rename(columns = DF_RENAMED_COLUMNS)


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorders the columns of historical Premier League match data for visual clarity.
    
    :param df: A DataFrame object storing historical Premier League match data.
    :return: A new DataFrame object storing historical Premier League match data with reordered columns for visual clarity.
    """
    return df[["Season", "Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "Result"]]


def standardize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the team names in historical Premier League match data.
    
    :param df: A DataFrame object storing historical Premier League match data.
    :return: A DataFrame object storing historical Premier League match data with standardized team names.
    """
    df["HomeTeam"] = df["HomeTeam"].replace(TEAM_NAME_MAP)
    df["AwayTeam"] = df["AwayTeam"].replace(TEAM_NAME_MAP)
    return df


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and validates historical Premier League match data.
    
    :param df: A DataFrame object storing historical Premier League match data to clean and validate.
    :return: A DataFrame object storing cleaned and validated historical Premier League match data.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"],
                                dayfirst = True, errors = "coerce", format = "mixed")
    # Drop match data with missing dates or score information
    df = df.dropna(subset = ["Date", "HomeGoals", "AwayGoals"])

    df["HomeGoals"] = df["HomeGoals"].astype("int64")
    df["AwayGoals"] = df["AwayGoals"].astype("int64")

    # Drop match data with negative goals
    df = df[(df["HomeGoals"] >= 0) & (df["AwayGoals"] >= 0)]

    # Drop match data with results inconsistent to scores
    expected_results = pd.Series(index = df.index, dtype = "string")
    expected_results[df["HomeGoals"] > df["AwayGoals"]] = "H"
    expected_results[df["HomeGoals"] == df["AwayGoals"]] = "D"
    expected_results[df["HomeGoals"] < df["AwayGoals"]] = "A"
    df["Result"] = df["Result"].astype(str).str.strip().str.upper()
    df = df[df["Result"] == expected_results]

    df = df.sort_values("Date").reset_index(drop = True)
    return df


def save_processed_df(df: pd.DataFrame, output_directory: Path) -> None:
    """
    Saves the processed historical Premier League match data to the specified output directory.
    
    :param df: A DataFrame object storing processed historical Premier League match data.
    :param output_directory: The output directory to save the processed historical Premier League match data to.
    :return: None.
    """
    output_directory.mkdir(parents = True, exist_ok = True)
    df.to_csv(output_directory / "epl_matches.csv", index = False)


def main() -> None:
    """
    Processes raw historical Premier League match data.
    
    :return: None.
    """
    df = load_raw_seasons(RAW_DATA_DIR, ALL_SEASONS)
    df = process_columns(df)
    df = standardize_team_names(df)
    df = clean_and_validate(df)
    save_processed_df(df, PROCESSED_DATA_DIR)
    print(f"Saved {len(df)} rows to {PROCESSED_DATA_DIR / "epl_matches.csv"}.")
    print(f"Season Match Counts: {df["Season"].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
