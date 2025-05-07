import argparse
import logging
import os
from tqdm import tqdm

import torch
from lhotse import load_manifest_lazy
import multi_quantization as quantization
import numpy as np

from icefall.utils import setup_logger, str2bool

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input-manifest",
        type=str,
    )
    
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
    )
    
    parser.add_argument(
        "--num-codebooks",
        type=int,
        default=4,
    )
    
    parser.add_argument(
        "--normalize",
        type=str2bool,
        default=False,
        help="If True, compute the channel-wise mean and std on the training se for nomalization."
    )
    
    parser.add_argument(
        "--embedding-path",
        type=str,
        help="path to the embedding file"
    )
    
    parser.add_argument(
        "--quantizer-path",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--quantizer-training-manifests",
        type=str,
        nargs="+",
        help="The manifests used for quantizer training."
    )
    
    parser.add_argument(
        "--quantizer-evaluation-manifests",
        type=str,
        nargs="+",
        help="The manifests used for quantizer training."
    )
    return parser.parse_args()

def normalize_data(data, mean, std):
    return (data-mean) / std

def prepare_data(manifest_list, split=True):
    # split needs to be True to enable shuffling! 
    logging.info(f"Preparing the data")
    all_data = []
    num_frames = 0
    for manifest in manifest_list:
        manifest = load_manifest_lazy(manifest)
        for cut in tqdm(manifest):
            assert cut.start == 0.0
            feature = cut.load_custom("embedding").astype(np.float16)
            num_frames += feature.shape[0]
            all_data.append(feature)
    
    all_data = np.concatenate(all_data, axis=0)
    
    if split:
        np.random.shuffle(all_data)
        all_data_pt = torch.from_numpy(all_data)
        valid_frames = int(num_frames * 0.05)
        
        if valid_frames > 10000:
            valid_frames = 10000
            
        return all_data_pt[valid_frames:], all_data_pt[:valid_frames]
    else:
        return torch.from_numpy(all_data)

def train_quantizer(args):
    device = torch.device("cuda")
    training_manifest = args.quantizer_training_manifests
    
    trainer = quantization.QuantizerTrainer(
        dim=args.embedding_dim,
        bytes_per_frame=args.num_codebooks,
        device=device,
        phase_one_iters=35000,
        phase_two_iters=35000,
    )
    
    train, valid = prepare_data(training_manifest, split=True)
    
    if args.normalize:
        mu = train.mean(dim=0)
        std = train.std(dim=0)
        train = normalize_data(train, mu, std)
        valid = normalize_data(valid, mu, std)
    else:
        mu, std = None, None
    B = 1024  # Minibatch size, this is very arbitrary,
    # it's close to what we used when we tuned this method.

    def minibatch_generator(data: torch.Tensor, repeat: bool):
        assert 3 * B < data.shape[0]
        cur_offset = 0
        while True if repeat else cur_offset + B <= data.shape[0]:
            start = cur_offset % (data.shape[0] + 1 - B)
            end = start + B
            cur_offset += B
            yield data[start:end, :].to(device).to(dtype=torch.float)

    # train
    logging.info(f"Start training the quantizer")
    for x in minibatch_generator(train, repeat=True):
        trainer.step(x)
        if trainer.done():
            break

    quantizer = trainer.get_quantizer()
    logging.info(f"Saving the quantizer to {args.quantizer_path}")
    state_dict = {
        "quantizer": quantizer.state_dict(),
        "mean": mu,
        "std": std,
    }
    torch.save(state_dict, args.quantizer_path)
    
    return quantizer, state_dict

@torch.no_grad()
def evaluate_quantizer(quantizer, valid):
    device = torch.device("cuda")
    
    B = 1024
    num_batches = valid.shape[0] // B
    cur_start = 0
    
    rel_losses = []
    log_probs = []
    
    for i in tqdm(range(num_batches)):
        data = valid[cur_start: cur_start+B]
        cur_start += B
        
        data = data.to(device).float()
        rel_reconstruction_loss, logprob_loss, _, _ = quantizer.compute_loss(data, refine_indexes_iters=5)
        rel_losses.append(rel_reconstruction_loss)
        log_probs.append(logprob_loss)
    
    logging.info(f"Relative reconstruction loss: {sum(rel_losses)/len(rel_losses)}")
    logging.info(f"Log probs: {sum(log_probs)/len(log_probs)}")
    
def main(args):
    device = torch.device("cuda")
    
    if os.path.exists(args.quantizer_path):
        import pdb; pdb.set_trace()
        logging.info(f"Loading from pre-trained quantizer: {args.quantizer_path}")
        
        quantizer = quantization.Quantizer(
            dim=args.embedding_dim,
            num_codebooks=args.num_codebooks,
            codebook_size=256,
        )
        state_dict = torch.load(args.quantizer_path)
        if "quantizer" not in state_dict:
            state_dict = {"quantizer": state_dict}
        quantizer.load_state_dict(state_dict["quantizer"])
        quantizer.to(device)
    else:
        quantizer, state_dict = train_quantizer(args)
        quantizer.to(device)
        
    if args.normalize:
        logging.info(f"Using the collected normalization stats")
        mu = state_dict["mean"]
        std = state_dict["std"]
        assert mu is not None and std is not None
        
    # evaluate quantizer
    valid_manifests = args.quantizer_evaluation_manifests
    for valid_manifest in valid_manifests:
        valid_data = prepare_data([valid_manifest], split=False)
        if args.normalize:
            valid_data = normalize_data(valid_data, mu, std)
        logging.info(f"Evaluating quantizer on {valid_manifest}")
        evaluate_quantizer(quantizer, valid_data)
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    setup_logger(f"data/quantizer/log-mvq")
    
    args = get_parser()
    logging.info(vars(args))
    
    main(args)