import numpy as np
import torch
import pytorch_lightning as pl
from config.single.deberta_v3_base import DeBERTaV3BaseCFG
from config.general import GeneralCFG
from data.data_module import FeedbackDataModule
from data import load_processed_data
from models.single.deberta_v3_base import DeBERTaV3BaseModel
from utils import metrics


def evaluate(seed):
    """
    学習済みモデルの評価
    """
    pl.seed_everything(seed)
    debug = GeneralCFG.debug

    oof_df = load_processed_data.train(seed=seed)[["text_id", "fold"] + GeneralCFG.target_cols]
    oof_df[GeneralCFG.target_cols] = np.nan

    if debug:
        oof_df = oof_df.loc[oof_df["fold"].isin(GeneralCFG.train_fold), :].reset_index(drop=True)

    for fold in GeneralCFG.train_fold:
        input_dir = DeBERTaV3BaseCFG.output_dir / f"seed{seed}"
        model = DeBERTaV3BaseModel()
        data_module = FeedbackDataModule(
            seed=seed,
            fold=fold,
            model_name=DeBERTaV3BaseCFG.model_name,
            batch_size=DeBERTaV3BaseCFG.batch_size
        )
        data_module.setup()
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[1],
            precision=16,
        )
        fold_i_val_pred = trainer.predict(
            model,
            dataloaders=data_module,
            ckpt_path=f"{input_dir}/best_loss_fold{fold}.ckpt",
            return_predictions=True
        )
        fold_i_val_pred = (torch.concat(fold_i_val_pred, axis=0).numpy())

        fold_idx = oof_df.loc[oof_df["fold"] == fold].index
        if debug:
            fold_idx = fold_idx[:GeneralCFG.num_use_data]

        oof_df.loc[fold_idx, GeneralCFG.target_cols] = fold_i_val_pred

        if debug and fold == len(GeneralCFG.train_fold) - 1:
            oof_df = oof_df.loc[oof_df.loc[:, GeneralCFG.target_cols[0]].notnull()].reset_index(drop=True)
            break

    oof_df.to_csv(DeBERTaV3BaseCFG.output_dir / "oof.csv", index=False)
    score, columns_score = metrics.get_oof_score(oof_df, is_debug=debug, seed=seed)
    return oof_df, score, columns_score


if __name__ == "__main__":
    from pathlib import Path
    GeneralCFG.data_version = "vanilla"
    DeBERTaV3BaseCFG.pooling = "mean"
    DeBERTaV3BaseCFG.output_dir = Path("/workspace", "output", "single", "deberta_v3_base", "baseline")
    oof_df, score, columns_score = evaluate(seed=42)
    print(score)
    print(columns_score)
    print(oof_df)
