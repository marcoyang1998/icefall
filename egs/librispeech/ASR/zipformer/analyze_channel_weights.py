import torch

ckpt = "./zipformer/exp_probe1/epoch-30.pt"
model = torch.load(ckpt, map_location="cpu")["model"]
log_dir = "analyze.log"

with open(log_dir, "w") as f:
    for key, value in model.items():
        if not key.endswith(".channel_weights"):
            continue

        normalized_weights = 0.5 * value/value.abs().mean()
        q = torch.tensor([0.2, 0.4, 0.6, 0.8])
        f.write(f"{key} mean: {normalized_weights.mean()}, std: {normalized_weights.std()}, percentile: {torch.quantile(normalized_weights, q)}\n")

