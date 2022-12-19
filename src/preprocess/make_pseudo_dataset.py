from pathlib import Path
import os
import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data import load_data
from preprocess.cv import add_fold_column
from config.general import GeneralCFG
from config.single.deberta_v3_base import DeBERTaV3BaseCFG
from models.single.deberta_v3_base import DeBERTaV3BaseModel
from data.dataset import FeedbackTestDataset


def make():
    train_df = load_data.train()
    test_df = load_data.test()
    sample_submission_df = load_data.sample_submission()

    for seed in GeneralCFG.seeds:
        output_path = Path("/workspace", "data", "processed", "pseudo_1", f"seed{seed}")
        output_path.mkdir(parents=True, exist_ok=True)

        pseudo_df = make_pseudo_df()
        pseudo_df = pseudo_df.loc[
            ~pseudo_df["text_id"].isin(train_df["text_id"].values)
        ]
        pseudo_df.to_csv(output_path / "pseudo_base.csv", index=False)

        # prediction by deberta_v3_base
        deberta_input_dir = DeBERTaV3BaseCFG.output_dir / f"seed{seed}"

        for fold in GeneralCFG.train_fold:
            print(f"start prediction fold: {fold}")

            pseudo_dataset = FeedbackTestDataset(
                input_df=pseudo_df,
                model_name=DeBERTaV3BaseCFG.model_name,
            )
            pseudo_loader = DataLoader(
                pseudo_dataset,
                batch_size=DeBERTaV3BaseCFG.batch_size,  # the large batch size is for speed up
                num_workers=4,
                shuffle=False
            )

            model = DeBERTaV3BaseModel()
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=[1],
                precision=16,
            )
            fold_i_pred = trainer.predict(
                model,
                dataloaders=pseudo_loader,
                ckpt_path=f"{deberta_input_dir}/best_loss_fold{fold}.ckpt",
                return_predictions=True,
            )
            fold_i_pred = (torch.concat(fold_i_pred, axis=0).numpy())

            fold_i_pseudo_df = pseudo_df.copy()
            fold_i_pseudo_df[GeneralCFG.target_cols] = fold_i_pred
            fold_i_pseudo_df.to_csv(
                output_path / f"pseudo_fold{fold}.csv",
                index=False
            )

        print("prediction finished")

        # preprocess
        sample_oof_df = train_df.drop(columns=["full_text"])
        sample_oof_df.iloc[:, 1:] = 3.0
        train_df = add_fold_column(train_df, num_folds=GeneralCFG.n_fold, random_state=seed)

        # save
        train_df.to_csv(output_path / "train.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        sample_submission_df.to_csv(output_path / "sample_submission.csv", index=False)
        sample_oof_df.to_csv(output_path / "sample_oof.csv", index=False)


def make_pseudo_df():
    fb1_text_path = GeneralCFG.data_dir / "external" / "fb1_text"
    concat_df_list = []

    i_pseudo_df = pd.DataFrame(
        columns=["text_id", "full_text"] + GeneralCFG.target_cols
    )
    for i, text_file in enumerate(tqdm.tqdm(fb1_text_path.glob("*.txt"))):
        text_id = text_file.stem
        with open(text_file, "r") as f:
            full_text = f.read()
        i_pseudo_df = pd.concat([i_pseudo_df, pd.DataFrame({
            "text_id": [text_id],
            "full_text": [full_text],
        })], axis=0)

        # to speed up
        if i % 100 == 99:
            concat_df_list.append(i_pseudo_df)
            i_pseudo_df = pd.DataFrame(
                columns=["text_id", "full_text"] + GeneralCFG.target_cols
            )
    pseudo_df = pd.concat(concat_df_list, axis=0).reset_index(drop=True)
    return pseudo_df


if __name__ == "__main__":
    GeneralCFG.data_version = "vanilla"
    DeBERTaV3BaseCFG.pooling = "mean"
    DeBERTaV3BaseCFG.output_dir = Path("/workspace", "output", "single", "deberta_v3_base", "baseline")
    make()