"""
A module storing project-level configurations.
"""
from pathlib import Path


ROOT_DIR           = Path(__file__).parent.parent
RAW_DATA_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR         = ROOT_DIR / "outputs"
POSTERIOR_DIR      = OUTPUT_DIR / "posterior"
FIGURES_DIR        = OUTPUT_DIR / "figures"
TABLES_DIR         = OUTPUT_DIR / "tables"

LEAGUE_CODE     = "E0"
# NOTE: The maximum number of goals scored by a single team in a single Premier League game is 9
MAX_GOALS       = 9
UNKNOWN_TEAM_ID = -1
RANDOM_SEED     = 42


TRAIN_SEASONS = ["16-17", "17-18", "18-19", "19-20", "20-21", "21-22", "22-23", "23-24"]
TEST_SEASONS  = ["24-25", "25-26"]

ALL_SEASONS = TRAIN_SEASONS + TEST_SEASONS
