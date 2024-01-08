import torch

from typing import Dict
import logging

@torch.no_grad()
def replace_dead_units(model_state_dict: Dict, threshold: float = 0.04):
    threshold = 0.05
    # model_state_dict = torch.load(model_path, map_location="cpu")
    
    if "model" in model_state_dict:
        model = model_state_dict["model"]
    else:
        model = model_state_dict

    for key, value in model.items():
        if not key.endswith(".channel_weights"):
            continue
        if "encoder_embed" in key:
            continue
        if "nonlin_attention" in key:
            continue

        normalized_weights = 0.5 * value/value.abs().mean() # ffw_dim
        max_idx = normalized_weights.argmax() 
        mask = normalized_weights.abs() < threshold

        if torch.any(mask):
            logging.info(f"{mask.sum()}/{mask.numel()} units in {key} are below the threshold")
            ffw_module = key.rsplit(".", 2)[0]
            num_bad_units = mask.sum()

            # input projection
            in_proj_weight = ffw_module + ".in_proj.weight"
            in_proj_bias = ffw_module + ".in_proj.bias"
            in_proj_weight = model[in_proj_weight] # ffw_dim * input_dim
            in_proj_bias = model[in_proj_bias] # ffw_dim
            
            new_in_proj_weight = in_proj_weight.clone()
            new_in_proj_weight[mask] = in_proj_weight[max_idx].unsqueeze(0) # replace the bad channels with the best one
            new_in_proj_bias = in_proj_bias.clone()
            new_in_proj_bias[mask] = in_proj_bias[max_idx]

            # output projection
            out_proj_weight = ffw_module + ".out_proj.weight"
            out_proj_bias = ffw_module + ".out_proj.bias"
            out_proj_weight = model[out_proj_weight] # input_dim, ffw_dim
            out_proj_bias = model[out_proj_bias] # ffw_dim

            new_out_proj_weight = out_proj_weight.clone()
            new_out_proj_weight[:, mask] = out_proj_weight[:, max_idx].unsqueeze(-1) # replace the bad channels with the best one
            # new_out_proj_bias = out_proj_bias.clone()
            # new_out_proj_bias[mask] = out_proj_bias[max_idx] 

            model[ffw_module + ".in_proj.weight"] = new_in_proj_weight
            model[ffw_module + ".in_proj.bias"] = new_in_proj_bias
            model[ffw_module + ".out_proj.weight"] = new_out_proj_weight
            # model[ffw_module + ".out_proj.bias"] = new_out_proj_bias

    return model

if __name__ == '__main__':
    model_path = "zipformer/exp_100h_probe1_scheduledfloat/epoch-5.pt"

    replace_dead_units(model_path=model_path)