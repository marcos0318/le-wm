import json
import torch
import stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")


def strip_hydra(cfg):
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def remap_encoder_key(key):
    """HF checkpoint uses older ViT key names than current transformers."""
    key = key.replace("encoder.encoder.layer.", "encoder.layers.")
    key = key.replace("attention.attention.query", "attention.q_proj")
    key = key.replace("attention.attention.key", "attention.k_proj")
    key = key.replace("attention.attention.value", "attention.v_proj")
    key = key.replace("attention.output.dense", "attention.o_proj")
    key = key.replace("intermediate.dense", "mlp.fc1")
    key = key.replace("output.dense", "mlp.fc2")
    return key


cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False,
    use_mask_token=False,
)
mlp = lambda k: MLP(
    input_dim=cfg[k]["input_dim"],
    output_dim=cfg[k]["output_dim"],
    hidden_dim=cfg[k]["hidden_dim"],
    norm_fn=torch.nn.BatchNorm1d,
)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**strip_hydra(cfg["predictor"])),
    action_encoder=Embedder(**strip_hydra(cfg["action_encoder"])),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
raw_sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
sd = {remap_encoder_key(k): v for k, v in raw_sd.items()}
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
print("Wrote", out)
