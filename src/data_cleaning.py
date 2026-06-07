import pandas as pd
from pathlib import Path

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

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "season"]
VALID_FTR = {"H", "D", "A"}


def load_raw_seasons(raw_dir: Path, seasons: list[str]) -> pd.DataFrame:
    dfs = []
    for season in seasons:
        path = raw_dir / f"{season}.csv"
        if not path.exists():
            print(f"Warning: {path} not found, skipping.")
            continue
        df = pd.read_csv(path, encoding="latin-1")
        df["season"] = season
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def keep_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in REQUIRED_COLUMNS if c in df.columns]].copy()


def standardize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    df["HomeTeam"] = df["HomeTeam"].replace(TEAM_NAME_MAP)
    df["AwayTeam"] = df["AwayTeam"].replace(TEAM_NAME_MAP)
    return df


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.dropna(subset=["FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype("int64")
    df["FTAG"] = df["FTAG"].astype("int64")
    df = df[df["FTR"].isin(VALID_FTR)]
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def save_processed(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "epl_matches.csv", index=False)


def main():
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, ALL_SEASONS
    df = load_raw_seasons(RAW_DATA_DIR, ALL_SEASONS)
    df = keep_required_columns(df)
    df = standardize_team_names(df)
    df = clean_and_validate(df)
    save_processed(df, PROCESSED_DATA_DIR)
    print(f"Saved {len(df)} rows to {PROCESSED_DATA_DIR / 'epl_matches.csv'}")
    print(f"Seasons: {df['season'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
