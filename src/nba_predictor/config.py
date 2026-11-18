import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_env_project_root = os.environ.get("PROJECT_ROOT")
PROJECT_ROOT = Path(_env_project_root) if _env_project_root else Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_SUBDIRS = ["odds", "sportsbook", "injuries", "espn"]

TRACKING_DB_PATH = Path(os.environ.get("TRACKING_DB_PATH", str(DATA_DIR / "tracking.db")))
_backup = os.environ.get("TRACKING_DB_BACKUP_PATH")
TRACKING_DB_BACKUP_PATH = Path(_backup) if _backup else None

PUBLIC_MODE = os.environ.get("PUBLIC_MODE", "false").lower() == "true"
PUBLIC_SNAPSHOT_POLL_SECONDS = int(os.environ.get("PUBLIC_SNAPSHOT_POLL_SECONDS", "300"))

BALLDONTLIE_API_KEY = os.environ.get("BALLDONTLIE_API_KEY")
SPORTSBOOK_API_KEY = os.environ.get("SPORTSBOOK_API_KEY")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")

# Score upcoming games' availability columns from tonight's ESPN injury report instead of who sat the previous game.
# Off: the serving replay (docs/nba-injury-report-availability-2026-11.md) did not pass the ship rule.
INJURY_REPORT_AVAILABILITY = os.environ.get("INJURY_REPORT_AVAILABILITY", "false").lower() == "true"


def ensure_cache_dirs() -> None:
    for name in CACHE_SUBDIRS:
        (CACHE_DIR / name).mkdir(parents=True, exist_ok=True)
