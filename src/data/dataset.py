import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from config.general import GeneralCFG
from config.single.deberta_v3_base import DeBERTaV3BaseCFG


def prepare_input(text, tokenizer, project_cfg):
    if project_cfg.batch_size == 1:
        padding_cfg = {
            "padding": "do_not_pad",
        }
    else:
        padding_cfg = {
            "padding": "max_length",
            "max_length": project_cfg.tokenizer_max_len,
        }

    inputs = tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        truncation=True,
        **padding_cfg,
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class FeedbackTrainDataset(Dataset):
    def __init__(self, input_df, model_name):
        self.texts = input_df['full_text'].values
        self.labels = input_df[GeneralCFG.target_cols].values
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.project_cfg = {
            "microsoft/deberta-v3-base": DeBERTaV3BaseCFG,
        }[model_name]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.tokenizer, self.project_cfg)
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label


class FeedbackTestDataset(Dataset):
    def __init__(self, input_df, model_name):
        self.texts = input_df['full_text'].values
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.project_cfg = {
            "microsoft/deberta-v3-base": DeBERTaV3BaseCFG,
        }[model_name]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.tokenizer, self.project_cfg)
        return inputs

