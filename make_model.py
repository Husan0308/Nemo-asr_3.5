from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
import torch

model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
    "nvidia/nemotron-3.5-asr-streaming-0.6b"
)

ckpt = torch.load(
    "/workspace/Nemo_asr/checkpoints/last.ckpt",
    map_location="cpu",
    weights_only=False,
)

model.load_state_dict(ckpt["state_dict"], strict=False)

model.save_to("/workspace/Nemo_asr/model_epoch2.nemo")

print("Saved!")
