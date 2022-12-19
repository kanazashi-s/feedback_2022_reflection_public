import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from config.single.deberta_v3_large import DeBERTaV3LargeCFG
from config.general import GeneralCFG
from data.data_module import FeedbackDataModule
from inference.make_submission_csv import make_submission_csv


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class DeBERTaV3LargeModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        config_path = DeBERTaV3LargeCFG.output_dir / "config.pth"
        self.model_config = torch.load(config_path)

        # pretrained model
        self.model = AutoModel.from_config(self.model_config)

        # layers
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.model_config.hidden_size, 6)
        self._init_weights(self.fc)

        self.loss = nn.SmoothL1Loss(reduction='mean')

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs).to(torch.float16)
        output = self.fc(feature)
        return output

    def configure_optimizers(self):

        # パラメーターごとに学習率を変えたりする処理を省略している
        optimizer = AdamW(
            self.model.parameters(), lr=self.lear, eps=GeneralCFG.eps, betas=DeBERTaV3LargeCFG.betas
        )
        num_train_steps = int(GeneralCFG.n_fold / DeBERTaV3LargeCFG.batch_size * self.trainer.max_epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=GeneralCFG.num_warmup_steps, num_training_steps=num_train_steps
        )
        return [optimizer, ], [scheduler, ]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch
        outputs = self.forward(inputs)
        return outputs

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def inference():
    pl.seed_everything(42)

    predictions_list = []
    for fold in GeneralCFG.train_fold:
        input_dir = DeBERTaV3LargeCFG.output_dir
        model = DeBERTaV3LargeModel()
        data_module = FeedbackDataModule(
            fold=fold,
            model_name=DeBERTaV3LargeCFG.model_name,
            batch_size=DeBERTaV3LargeCFG.batch_size
        )
        data_module.setup()
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision=16,
        )
        predictions = trainer.predict(
            model,
            dataloaders=data_module.test_dataloader(),
            ckpt_path=f"{input_dir}/best_loss_fold{fold}.ckpt",
            return_predictions=True
        )
        predictions_list.append(torch.concat(predictions, axis=0).numpy())
        if GeneralCFG.debug and fold == 1:
            break

    predictions = np.mean(predictions_list, axis=0)
    make_submission_csv(predictions, DeBERTaV3LargeCFG.output_dir)

    return predictions


if __name__ == "__main__":
    predictions = inference()
    print(predictions)
