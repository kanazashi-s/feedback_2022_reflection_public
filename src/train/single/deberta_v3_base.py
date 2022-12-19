from pathlib import Path
import shutil
import logging
import gc
import numpy as np
import torch
from transformers import AutoTokenizer
import pytorch_lightning as pl
from config.single.deberta_v3_base import DeBERTaV3BaseCFG
from config.general import GeneralCFG
from data.data_module import FeedbackDataModule
from models.single.deberta_v3_base import DeBERTaV3BaseModel
from evaluation.single.deberta_v3_base import evaluate
from utils import upload_model


def train(run_name: str):

    score_list = []
    columns_score_list = []

    for seed in GeneralCFG.seeds:
        output_dir = DeBERTaV3BaseCFG.output_dir / f"seed{seed}"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        pl.seed_everything(seed)

        tokenizer = AutoTokenizer.from_pretrained(DeBERTaV3BaseCFG.model_name)
        tokenizer.save_pretrained(output_dir)

        for fold in GeneralCFG.train_fold:
            model_save_name = f"best_loss_fold{fold}"

            data_module = FeedbackDataModule(
                seed=seed,
                fold=fold,
                model_name=DeBERTaV3BaseCFG.model_name,
                batch_size=DeBERTaV3BaseCFG.batch_size
            )

            data_module.setup()

            DeBERTaV3BaseCFG.num_training_steps = len(data_module.train_dataloader()) * DeBERTaV3BaseCFG.epochs

            model = DeBERTaV3BaseModel()
            torch.save(model.model_config, output_dir / "config.pth")

            # mlflow logger
            experiment_name_prefix = "debug_" if GeneralCFG.debug else ""
            mlflow_logger = pl.loggers.MLFlowLogger(
                experiment_name=experiment_name_prefix + "deberta_v3_base",
                run_name=f"{run_name}_seed_{seed}_fold{fold}",
            )
            mlflow_logger.log_hyperparams(get_param_dict())

            # ロスが最小になったタイミングでモデルを保存
            loss_callback = pl.callbacks.ModelCheckpoint(
                monitor="val_mcrmse",
                dirpath=output_dir,
                filename=model_save_name,
                save_top_k=1,
                mode="min",
                every_n_train_steps=DeBERTaV3BaseCFG.val_check_interval
            )

            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
            swa_callback = pl.callbacks.StochasticWeightAveraging(
                swa_lrs=DeBERTaV3BaseCFG.lr / 10,
                swa_epoch_start=2,
            )

            callbacks = [loss_callback, lr_monitor]
            if DeBERTaV3BaseCFG.swa:
                callbacks.append(swa_callback)

            trainer = pl.Trainer(
                devices=[1],
                accelerator="gpu",
                max_epochs=DeBERTaV3BaseCFG.epochs,
                precision=16,
                amp_backend='apex',
                amp_level='O2',
                gradient_clip_val=DeBERTaV3BaseCFG.max_grad_norm,
                accumulate_grad_batches=DeBERTaV3BaseCFG.accumulate_grad_batches,
                logger=mlflow_logger,
                default_root_dir=output_dir,
                callbacks=callbacks,
                val_check_interval=DeBERTaV3BaseCFG.val_check_interval,
            )
            trainer.fit(model, data_module)
            torch.cuda.empty_cache()
            gc.collect()

        oof_df, score, columns_score = evaluate(seed)
        mlflow_logger.log_metrics({
            "oof_score": score,
            "cohesion": columns_score[0],
            "syntax": columns_score[1],
            "vocabulary": columns_score[2],
            "phraseology": columns_score[3],
            "grammer": columns_score[4],
            "conventions": columns_score[5],
        })

        upload_model.create_dataset_metadata(
            model_name=f"deberta-v3-base-seed{seed}",
            model_path=output_dir,
        )

        score_list.append(score)
        columns_score_list.append(columns_score)

    oof_score_seed_mean = sum(score_list) / len(score_list)
    columns_score_seed_mean = np.mean(columns_score_list, axis=0).tolist()
    mlflow_logger.log_metrics({
        "oof_score_seed_mean": oof_score_seed_mean,
        "cohesion_seed_mean": columns_score_seed_mean[0],
        "syntax_seed_mean": columns_score_seed_mean[1],
        "vocabulary_seed_mean": columns_score_seed_mean[2],
        "phraseology_seed_mean": columns_score_seed_mean[3],
        "grammer_seed_mean": columns_score_seed_mean[4],
        "conventions_seed_mean": columns_score_seed_mean[5],
    })

    print(f"oof_score_seed_mean: {oof_score_seed_mean}")
    print(f"columns_score_seed_mean: {columns_score_seed_mean}")

    upload_model.create_dataset_metadata(
        model_name=f"deberta-v3-base",
        model_path=DeBERTaV3BaseCFG.output_dir,
    )

    return oof_score_seed_mean, columns_score_seed_mean


def get_param_dict():
    param_dict = {
        "debug": GeneralCFG.debug,
        "data_version": GeneralCFG.data_version,
        "tokenizer_max_len": DeBERTaV3BaseCFG.tokenizer_max_len,
        "model_max_len": DeBERTaV3BaseCFG.model_max_len,
        "eps": GeneralCFG.eps,
        "num_workers": GeneralCFG.num_workers,
        "n_fold": GeneralCFG.n_fold,
        "num_use_data": GeneralCFG.num_use_data,
        "model_name": DeBERTaV3BaseCFG.model_name,
        "lr": DeBERTaV3BaseCFG.lr,
        "bert_first_half_lr_scale": DeBERTaV3BaseCFG.bert_first_half_lr_scale,
        "bert_second_half_lr_scale": DeBERTaV3BaseCFG.bert_second_half_lr_scale,
        "weight_decay": DeBERTaV3BaseCFG.weight_decay,
        "betas": DeBERTaV3BaseCFG.betas,
        "batch_size": DeBERTaV3BaseCFG.batch_size,
        "epochs": DeBERTaV3BaseCFG.epochs,
        "pooling": DeBERTaV3BaseCFG.pooling,
        "scheduler": DeBERTaV3BaseCFG.scheduler,
        "num_warmup_steps": DeBERTaV3BaseCFG.num_warmup_steps,
        "num_cycles": DeBERTaV3BaseCFG.num_cycles,
        "max_grad_norm": DeBERTaV3BaseCFG.max_grad_norm,
        "accumulate_grad_batches": DeBERTaV3BaseCFG.accumulate_grad_batches,
        "fgm": DeBERTaV3BaseCFG.fgm,
        "init_weight": DeBERTaV3BaseCFG.init_weight,
        "hidden_dropout_prob": DeBERTaV3BaseCFG.hidden_dropout_prob,
        "attention_probs_dropout_prob": DeBERTaV3BaseCFG.attention_probs_dropout_prob,
        "swa": DeBERTaV3BaseCFG.swa,
        "loss": DeBERTaV3BaseCFG.loss,
        "smooth_l1_beta": DeBERTaV3BaseCFG.smooth_l1_beta,
        "smooth_l1_target_weights": DeBERTaV3BaseCFG.smooth_l1_target_weights,
    }
    return param_dict


if __name__ == "__main__":
    DeBERTaV3BaseCFG.output_dir = Path("/workspace", "output", "single", "deberta_v3_base", "debug")
    oof_score_seed_mean, columns_score_seed_mean = train(f"baseline")


