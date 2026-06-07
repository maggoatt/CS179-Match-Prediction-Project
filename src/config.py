from pathlib import Path

ROOT_DIR           = Path(__file__).parent.parent
RAW_DATA_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR         = ROOT_DIR / "outputs"
POSTERIOR_DIR      = OUTPUT_DIR / "posterior"
FIGURES_DIR        = OUTPUT_DIR / "figures"
TABLES_DIR         = OUTPUT_DIR / "tables"

LEAGUE_CODE     = "E0"
MAX_GOALS       = 10
RANDOM_SEED     = 42
UNKNOWN_TEAM_ID = -1

TRAIN_SEASONS = ["1516", "1617", "1718", "1819", "1920", "2021", "2122"]
VAL_SEASONS   = ["2223"]
TEST_SEASONS  = ["2324"]

ALL_SEASONS = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS
