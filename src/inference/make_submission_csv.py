from pathlib import Path
import numpy as np
import pandas as pd
from data import load_processed_data


def make_submission_csv(
        predictions: np.ndarray,
        output_path: Path,
        seed: int
):

    sub_df = load_processed_data.sample_submission(seed=seed)
    sub_df.iloc[:, 1:] = predictions
    sub_df.to_csv(output_path / "submission.csv", index=False)
