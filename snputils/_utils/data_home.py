import os
from pathlib import Path
import shutil


def get_data_home(data_home: Path | str = None) -> Path:
    if data_home is None:
        data_home = os.environ.get("SNPUTILS_DATA", Path.home() / ".snputils" / "data")
    data_home = Path(data_home)
    data_home.mkdir(parents=True, exist_ok=True)
    return data_home


def clear_data_home(data_home: Path | str = None) -> None:
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)
