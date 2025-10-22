# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import numpy as np

import joblib
import torch
import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)



def dump_label(km_path):
    apply_kmeans = ApplyKmeans(km_path)
    from lhotse import load_manifest_lazy, CutSet
    from lhotse.utils import fastcopy
    
    # manifest_name = "hubert/embeddings/embeddings_hubert_base_layer_9_vox1_test.jsonl.gz"
    manifest_name = "hubert/embeddings/embeddings_hubert_base_layer_9_ls_dev-clean.jsonl.gz"
    cuts = load_manifest_lazy(manifest_name)
    
    new_cuts = []
    for i, cut in enumerate(cuts):
        embed = cut.load_custom("embedding")
        km_label = apply_kmeans(embed).tolist()
        new_cut = fastcopy(
            cut,
            custom={"kmeans": km_label}
        )
        new_cuts.append(new_cut)
        if i % 100 == 0:
            logger.info(f"Processed {i} cuts")
    
    new_cuts = CutSet.from_cuts(new_cuts)
    # output_manifest = "hubert/embeddings/kmeans_hubert_base_layer_9_vox1_test.jsonl.gz"
    output_manifest = "hubert/embeddings/kmeans_hubert_base_layer_9_ls_dev-clean.jsonl.gz"
    new_cuts.to_jsonl(output_manifest)
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--km-path", default="download/models/hubert_base_ls960_L9_km500.bin")
    args = parser.parse_args()
    logging.info(str(args))

    dump_label(args.km_path)