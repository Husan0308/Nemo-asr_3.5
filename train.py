import yaml
import os
import random
import numpy as np
import torch
import lightning.pytorch as pl

from nemo.collections.asr.models import EncDecRNNTBPEModel
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from lora import LoRALinear



import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.set_float32_matmul_precision("medium")


# =========================================================
# 0. REPRODUCIBILITY
# =========================================================
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(42)


# =========================================================
# 1. SAFE CONFIG ACCESS
# =========================================================
def get(cfg, *keys, default=None):
    for k in keys:
        if isinstance(cfg, dict) and k in cfg:
            cfg = cfg[k]
        else:
            return default
    return cfg


# =========================================================
# 2. LOAD CONFIG
# =========================================================
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)


# =========================================================
# 3. LOAD MODEL
# =========================================================
print("Loading model...")
model = EncDecRNNTBPEModel.from_pretrained(cfg["model_name"])

print("=" * 80)

try:
    print("ATT_CONTEXT:", model.cfg.encoder.att_context_size)
except Exception as e:
    print("ATT_CONTEXT not found:", e)

print("=" * 80)


# =========================================================
# 4. LORA INJECTION (SAFE + ROBUST)
# =========================================================
def inject_lora(model, r=16, alpha=32):

    if not hasattr(model, "encoder"):
        raise RuntimeError("Model has no encoder")

    for p in model.parameters():
        p.requires_grad = False

    for layer in model.encoder.layers:

        if not hasattr(layer, "self_attn"):
            continue

        attn = layer.self_attn

        # Attention projections
        attn.linear_q = LoRALinear(attn.linear_q, r=r, alpha=alpha)
        attn.linear_v = LoRALinear(attn.linear_v, r=r, alpha=alpha)
        attn.linear_out = LoRALinear(attn.linear_out, r=r, alpha=alpha)

        # Feedforward block 1
        if hasattr(layer, "feed_forward1"):
            ff = layer.feed_forward1
            ff.linear1 = LoRALinear(ff.linear1, r=r, alpha=alpha)
            ff.linear2 = LoRALinear(ff.linear2, r=r, alpha=alpha)

        # Feedforward block 2 (optional)
        if hasattr(layer, "feed_forward2"):
            ff = layer.feed_forward2
            ff.linear1 = LoRALinear(ff.linear1, r=r, alpha=alpha)
            ff.linear2 = LoRALinear(ff.linear2, r=r, alpha=alpha)

    return model


print("Injecting LoRA...")
model = inject_lora(
    model,
    r=get(cfg, "lora", "rank", default=16),
    alpha=get(cfg, "lora", "alpha", default=32),
)


# =========================================================
# 5. CONFIG PATCHING (SAFE)
# =========================================================
train_cfg = cfg.get("train_ds", {})
val_cfg = cfg.get("validation_ds", {})

model.cfg.train_ds.manifest_filepath = train_cfg["manifest_filepath"]
model.cfg.validation_ds.manifest_filepath = val_cfg["manifest_filepath"]

safe_fields = [
    "batch_duration",
    "max_duration",
    "use_bucketing",
    "num_buckets",
    "bucket_buffer_size",
    "shuffle_buffer_size",
    "num_workers",
]

for k in safe_fields:
    if k in train_cfg:
        setattr(model.cfg.train_ds, k, train_cfg[k])

if "num_workers" in val_cfg:
    model.cfg.validation_ds.num_workers = val_cfg["num_workers"]

# optimizer
model.cfg.optim.lr = get(cfg, "optim", "lr", default=1e-4)



from omegaconf import OmegaConf

print("=" * 80)
print("MODEL CLASS:", model.__class__.__name__)
print("=" * 80)

print(OmegaConf.to_yaml(model.cfg.train_ds))

print("=" * 80)
print(OmegaConf.to_yaml(model.cfg.validation_ds))
print("=" * 80)



# =========================
# DISABLE NEMO SCHEDULER
# =========================
OmegaConf.set_struct(model.cfg.optim, False)
OmegaConf.set_struct(model._cfg.optim, False)
model.cfg.optim.sched = {}
model._cfg.optim.sched = {}





# =========================================================
# 6. DATASET INIT (CRITICAL ORDER)
# =========================================================
print("Setting up datasets...")
model.setup_training_data(model.cfg.train_ds)
model.setup_validation_data(model.cfg.validation_ds)


# =========================================================
# 7. MODEL STATS
# =========================================================
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable: {trainable:,}")
print(f"Total: {total:,}")
print(f"Trainable %: {100 * trainable / total:.3f}%")


# =========================================================
# 8. CALLBACKS
# =========================================================
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="step-{step}",
    every_n_train_steps=2000,
    save_top_k=-1,
    save_last=True,
)

lr_monitor = LearningRateMonitor(logging_interval="step")


# =========================================================
# 9. TRAINER (PRODUCTION SETTINGS)
# =========================================================
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    precision=get(cfg, "trainer", "precision", default="bf16-mixed"),
    max_epochs=get(cfg, "trainer", "max_epochs", default=10),
    accumulate_grad_batches=get(cfg, "trainer", "accumulate_grad_batches", default=1),
    gradient_clip_val=get(cfg, "trainer", "gradient_clip_val", default=1.0),
    log_every_n_steps=10,
    enable_checkpointing=True,
    callbacks=[checkpoint_callback, lr_monitor],
    deterministic=True,
)






print("MODEL CLASS:", model.__class__.__name__)
print("OPTIM:", model.cfg.optim)

train_dl = model._train_dl
print("TRAIN DL:", type(train_dl))

if hasattr(train_dl, "dataset"):
    print("DATASET:", type(train_dl.dataset))





# =========================================================
# 10. TRAIN
# =========================================================
print("Starting training...")
trainer.fit(model)

# =========================================================
# 11. SAVE OUTPUTS
# =========================================================
print("Saving model...")


print("Done.")
