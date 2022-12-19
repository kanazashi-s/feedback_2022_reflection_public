from pathlib import Path
from config.general import GeneralCFG


class DeBERTaV3BaseCFG:
    model_name = "microsoft/deberta-v3-base"
    output_dir = Path("/workspace", "output", "single", "deberta_v3_base")
    lr = 2e-5
    tokenizer_max_len = 2048
    model_max_len = 2048
    bert_first_half_lr_scale = 1.0
    bert_second_half_lr_scale = 1.0
    weight_decay = 0.01
    betas = (0.9, 0.999)
    batch_size = 4
    epochs = 4
    pooling = "mean"
    num_hidden_layers = 12
    scheduler = "cosine"
    num_warmup_steps = 0
    num_cycles = 0.5
    max_grad_norm = 1000
    accumulate_grad_batches = 4
    fgm = False
    init_weight = "orthogonal"
    hidden_dropout_prob = 0.
    attention_probs_dropout_prob = 0.
    swa = False
    val_check_interval = 125
    loss = "smooth_l1"
    smooth_l1_beta = 1
    smooth_l1_target_weights = [0.21, 0.16, 0.10, 0.16, 0.21, 0.16]


if GeneralCFG.debug:
    DeBERTaV3BaseCFG.epochs = 2
    DeBERTaV3BaseCFG.output_dir = Path("/workspace", "output", "single", "debug_deberta_v3_base")
    DeBERTaV3BaseCFG.val_check_interval = GeneralCFG.num_use_data // 8


if DeBERTaV3BaseCFG.batch_size == 1:
    DeBERTaV3BaseCFG.val_check_interval = 500


# GeneralCFG.data_version か DeBERTaV3BaseCFG.pooling の片方が "reg_token" の場合、
# もう片方も "reg_token" でなければ assert で弾く
if GeneralCFG.data_version == "reg_token" or DeBERTaV3BaseCFG.pooling == "reg_token":
    assert GeneralCFG.data_version == "reg_token" and DeBERTaV3BaseCFG.pooling == "reg_token"
