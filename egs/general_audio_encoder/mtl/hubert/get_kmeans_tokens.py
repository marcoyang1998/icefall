# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import fairseq
import torch
import torch.nn.functional as F

from fairseq.data.audio.audio_utils import get_features_or_waveform


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav = get_features_or_waveform(path, need_waveform=True, use_sample_rate=self.task.cfg.sample_rate)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len=ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)

def collect_voxceleb_embeddings(args):
    reader = HubertFeatureReader(args.ckpt_path, args.layer, args.max_chunk)
    from lhotse import load_manifest_lazy, CutSet
    from lhotse.features.io import NumpyHdf5Writer
    from lhotse.utils import fastcopy
    # manifest_name = "data/fbank_voxceleb/vox1_cuts_test.jsonl.gz"
    manifest_name = "data/librispeech_manifest/librispeech_cuts_dev-clean.jsonl.gz"
    cuts = load_manifest_lazy(manifest_name)
    new_cuts = []
    
    import pdb; pdb.set_trace()
    # embedding_path = "hubert/embeddings/hubert_base_layer_9_vox1_test.h5"
    embedding_path = "hubert/embeddings/hubert_base_layer_9_ls_dev-clean.h5"
    with NumpyHdf5Writer(embedding_path) as writer:
        for i, c in enumerate(cuts):
            audio_file = c.recording.sources[0].source
            feat = reader.get_feats(audio_file)
            feat = feat.detach().cpu().numpy()
            embedding = writer.store_array(
                key=c.id,
                value=feat,
                temporal_dim=0,
                frame_shift=0.02,
                start=c.start,
            )
            new_cut = fastcopy(
                c,
                custom={"embedding": embedding}
            )
            new_cuts.append(new_cut)
            if i % 100 == 0:
                print(f"Processed {i} cuts")
    import pdb; pdb.set_trace()
    # output_manifest_name = "hubert/embeddings/embeddings_hubert_base_layer_9_vox1_test.jsonl.gz"
    output_manifest_name = "hubert/embeddings/embeddings_hubert_base_layer_9_ls_dev-clean.jsonl.gz"
    new_cuts = CutSet.from_cuts(new_cuts)
    new_cuts.to_jsonl(output_manifest_name)
    print(f"Saved manifest to {output_manifest_name}")


# def main(tsv_dir, split, ckpt_path, layer, nshard, rank, feat_dir, max_chunk):
#     reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
#     generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
#     dump_feature(reader, generator, num, split, nshard, rank, feat_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", default="download/models/hubert_base_ls960.pt")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--max-chunk", type=int, default=1600000)
    args = parser.parse_args()
    logger.info(args)

    collect_voxceleb_embeddings(args)