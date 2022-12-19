from pathlib import Path
import pandas as pd
from config.general import GeneralCFG


def train(seed: int):
    input_path = Path("/workspace", "data", "processed", GeneralCFG.data_version, f"seed{seed}")
    return pd.read_csv(input_path / "train.csv")


def test(seed: int):
    input_path = Path("/workspace", "data", "processed", GeneralCFG.data_version, f"seed{seed}")
    return pd.read_csv(input_path / "test.csv")


def sample_submission(seed: int):
    input_path = Path("/workspace", "data", "processed", GeneralCFG.data_version, f"seed{seed}")
    return pd.read_csv(input_path / "sample_submission.csv")


def sample_oof(seed: int):
    input_path = Path("/workspace", "data", "processed", GeneralCFG.data_version, f"seed{seed}")
    return pd.read_csv(input_path / "sample_oof.csv")


def pseudo_1(seed: int, fold: int):
    input_path = Path("/workspace", "data", "processed", "pseudo_1", f"seed{seed}")
    return pd.read_csv(input_path / f"pseudo_fold{fold}.csv")

