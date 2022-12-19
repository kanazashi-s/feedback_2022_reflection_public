from typing import Optional
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data import load_processed_data
from data.dataset import FeedbackTrainDataset, FeedbackTestDataset
from config.general import GeneralCFG


class FeedbackDataModule(pl.LightningDataModule):
    def __init__(self, seed: int, fold: int, model_name: str, batch_size: int, is_pseudo: bool = False):
        super().__init__()
        self.seed = seed
        self.fold = fold
        self.model_name = model_name
        self.batch_size = batch_size
        self.is_pseudo = is_pseudo

    def setup(self, stage: Optional[str] = None) -> None:
        if self.is_pseudo:
            self._setup_pseudo()
            return

        whole_df = load_processed_data.train(seed=self.seed)

        train_df = whole_df[whole_df["fold"] != self.fold]
        train_df = train_df.drop(["fold", "text_id"], axis=1)
        valid_df = whole_df[whole_df["fold"] == self.fold]
        valid_df = valid_df.drop(["fold", "text_id"], axis=1)

        test_df = load_processed_data.test(seed=self.seed)
        test_df = test_df.drop(["text_id"], axis=1)

        if GeneralCFG.debug:
            train_df = train_df.head(GeneralCFG.num_use_data).reset_index(drop=True)
            valid_df = valid_df.head(GeneralCFG.num_use_data).reset_index(drop=True)

        self.train_dataset = FeedbackTrainDataset(train_df, self.model_name)
        self.valid_dataset = FeedbackTrainDataset(valid_df, self.model_name)
        self.test_dataset = FeedbackTestDataset(test_df, self.model_name)
        self.val_predict_dataset = FeedbackTestDataset(valid_df[["full_text"]], self.model_name)

    def _setup_pseudo(self):
        whole_df = load_processed_data.pseudo_1(seed=self.seed, fold=self.fold)
        whole_df = whole_df.drop(["text_id"], axis=1)

        train_df, valid_df = train_test_split(whole_df, test_size=0.2, random_state=self.seed)
        if GeneralCFG.debug:
            train_df = train_df.head(GeneralCFG.num_use_data).reset_index(drop=True)
            valid_df = valid_df.head(GeneralCFG.num_use_data).reset_index(drop=True)
        self.train_dataset = FeedbackTrainDataset(train_df, self.model_name)
        self.valid_dataset = FeedbackTrainDataset(valid_df, self.model_name)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.val_predict_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)


if __name__ == "__main__":
    data_module = FeedbackDataModule(seed=42, fold=0, model_name="microsoft/deberta-v3-base", batch_size=1)
    data_module.setup()
    for inputs, labels in data_module.train_dataloader():
        print(inputs)
        print(labels)
        break