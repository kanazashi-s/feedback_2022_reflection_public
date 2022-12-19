import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from config.general import GeneralCFG
from config.single.deberta_v3_base import DeBERTaV3BaseCFG
from models.single.deberta_poolings.mean import MeanPooling
from models.single.deberta_poolings.gem_cls import GemClsPooling
from models.single.deberta_poolings.reg_token import RegTokenPooling
from models.single.model_utils.fgm import FGM
from models.single.model_utils.weighted_smooth_l1_loss import WeightedSmoothL1Loss
from utils.metrics import get_score


class DeBERTaV3BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # config
        self.model_config = AutoConfig.from_pretrained(DeBERTaV3BaseCFG.model_name, output_hidden_states=True)
        self.model_config.hidden_dropout_prob = DeBERTaV3BaseCFG.hidden_dropout_prob
        self.model_config.attention_probs_dropout_prob = DeBERTaV3BaseCFG.attention_probs_dropout_prob
        self.model_config.max_position_embeddings = DeBERTaV3BaseCFG.tokenizer_max_len

        # pretrained model
        self.model = AutoModel.from_pretrained(DeBERTaV3BaseCFG.model_name, config=self.model_config)
        # layers
        pool_dict = {
            "mean": MeanPooling,
            "gem_cls": GemClsPooling,
            "reg_token": RegTokenPooling,
        }
        self.pool = pool_dict[DeBERTaV3BaseCFG.pooling]()
        if DeBERTaV3BaseCFG.pooling == "gem_cls":
            self.fc = nn.Linear(self.model_config.hidden_size * 2, 6)
        else:
            self.fc = nn.Linear(self.model_config.hidden_size, 6)
        if DeBERTaV3BaseCFG.pooling == "reg_token":
            self.each_target_fc_list = nn.ModuleList([nn.Linear(self.model_config.hidden_size, 1) for _ in range(6)])
            for each_target_fc in self.each_target_fc_list:
                self._init_weights(each_target_fc)
        self._init_weights(self.fc)
        self.learning_rate = DeBERTaV3BaseCFG.lr
        if DeBERTaV3BaseCFG.loss == "weighted_smooth_l1":
            self.loss = WeightedSmoothL1Loss(
                beta=DeBERTaV3BaseCFG.smooth_l1_beta,
                target_weights=DeBERTaV3BaseCFG.smooth_l1_target_weights
            )
        else:
            self.loss = nn.SmoothL1Loss()

        if DeBERTaV3BaseCFG.fgm:
            self.fgm = FGM(self)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        feature = self.pool(outputs[0], inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)  # .half()

        if DeBERTaV3BaseCFG.pooling == "reg_token":
            output_list = []
            for i, each_target_fc in enumerate(self.each_target_fc_list):
                output_list.append(each_target_fc(feature[:, i, :]))
            output = torch.cat(output_list, dim=1)
            return output
        output = self.fc(feature)
        return output

    def configure_optimizers(self):

        optimizer_parameters = self.get_optimizer_parameters(DeBERTaV3BaseCFG.weight_decay)

        optimizer = AdamW(
            optimizer_parameters,
            eps=GeneralCFG.eps,
            betas=DeBERTaV3BaseCFG.betas
        )

        scheduler = self.get_scheduler(optimizer)
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config
        }

    def get_optimizer_parameters(self, weight_decay):
        deberta_params = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        half_param_num = len(deberta_params) // 2
        optimizer_parameters = [
            {
                'params': [p for n, p in deberta_params[:half_param_num] if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': self.learning_rate * DeBERTaV3BaseCFG.bert_first_half_lr_scale
            },
            {
                'params': [p for n, p in deberta_params[:half_param_num] if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.learning_rate * DeBERTaV3BaseCFG.bert_first_half_lr_scale
            },
            {
                'params': [p for n, p in deberta_params[half_param_num:] if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': self.learning_rate * DeBERTaV3BaseCFG.bert_second_half_lr_scale
            },
            {
                'params': [p for n, p in deberta_params[half_param_num:] if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.learning_rate * DeBERTaV3BaseCFG.bert_second_half_lr_scale
            },
            {
                'params': self.pool.parameters(),
                'weight_decay': 0.0,
                'lr': self.learning_rate
            },
            {
                'params': self.fc.parameters(),
                'weight_decay': 0.0,
                'lr': self.learning_rate
            }
        ]

        if DeBERTaV3BaseCFG.pooling == "reg_token":
            optimizer_parameters.append({
                'params': self.each_target_fc_list.parameters(),
                'weight_decay': 0.0,
                'lr': self.learning_rate
            })

        return optimizer_parameters

    def get_scheduler(self, optimizer):
        if DeBERTaV3BaseCFG.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=DeBERTaV3BaseCFG.num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif DeBERTaV3BaseCFG.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=DeBERTaV3BaseCFG.num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                num_cycles=DeBERTaV3BaseCFG.num_cycles
            )
        return scheduler

    def training_step(self, batch, batch_idx):
        self.inputs, self.labels = batch
        if DeBERTaV3BaseCFG.batch_size != 1:
            self.inputs = collate(self.inputs)
        outputs = self.forward(self.inputs)
        loss = self.loss(outputs, self.labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if DeBERTaV3BaseCFG.pooling == "gem_cls":
            self.log('gem_p', self.pool.p, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        if DeBERTaV3BaseCFG.batch_size != 1:
            inputs = collate(inputs)
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss, outputs, labels

    def test_step(self, batch, batch_idx):
        inputs = batch
        if DeBERTaV3BaseCFG.batch_size != 1:
            inputs = collate(inputs)
        outputs = self.forward(inputs)
        return outputs

    def validation_epoch_end(self, validation_step_outputs) -> None:
        all_preds = torch.cat([x[1] for x in validation_step_outputs], dim=0).cpu().detach().numpy()
        all_labels = torch.cat([x[2] for x in validation_step_outputs], dim=0).cpu().detach().numpy()
        mcrmse_score, scores = get_score(all_labels, all_preds)
        self.log('val_mcrmse', mcrmse_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for i, score in enumerate(scores):
            self.log(f'val_score_{i}', score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return mcrmse_score

    def on_after_backward(self) -> None:
        if DeBERTaV3BaseCFG.fgm:
            self.fgm.attack()
            loss_adv = self.loss(self.forward(self.inputs), self.labels)
            loss_adv.backward()
            self.fgm.restore()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if DeBERTaV3BaseCFG.init_weight == "normal":
                module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            elif DeBERTaV3BaseCFG.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


if __name__ == '__main__':
    model = DeBERTaV3BaseModel()
    print(model)
