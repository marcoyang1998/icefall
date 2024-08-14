# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import inspect
import logging
from functools import lru_cache
from pathlib import Path
import pickle
from typing import Any, Dict, Optional

import whisper
from BEATs import BEATs, BEATsConfig
from dataset import MultiKDDataset, SpeakerRecognitionDataset
from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
from teachers import Teacher, BEATsTeacher, EcapaTeacher, WhisperTeacher

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import (  # noqa F401 For AudioSamples
    AudioSamples,
    OnTheFlyFeatures,
)
from lhotse.supervision import SupervisionSegment
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class LibriSpeechKDDataModule:
    """
    DataModule for k2 KD experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(
        self, 
        args: argparse.Namespace,
        device: torch.device = torch.device("cpu"),
        evaluation: bool = False,
    ):
        self.args = args
        self.device = device
        
        # Only load the teacher if using on the fly features
        if not evaluation and self.args.on_the_fly_feats:
            self.beats = self.load_beats(args, device) if self.args.use_beats else None
            self.ecapa = self.load_ecapa(args, device) if self.args.use_ecapa else None
            self.whisper = self.load_whisper(args, device) if self.args.use_whisper else None
        else:
            self.beats = None
            self.ecapa = None
            self.whisper = None

    def load_beats(self, args, device):
        
        checkpoint = torch.load(args.beats_ckpt)
        cfg = BEATsConfig(checkpoint['cfg'])
    
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])
        BEATs_model.eval() # deactivate dropout/normalization
        BEATs_model.to(torch.float16)
        BEATs_model.to(device)
        logging.info(f"Successfully load BEATs model.")

        return BEATsTeacher(model=BEATs_model)

    def load_ecapa(self, args, device):
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        classifier.eval()
        logging.info(f"Successfully load ecapa-tdnn model.")
        
        return EcapaTeacher(model=classifier)
    
    def load_whisper(self, args, device):
        # Currently only load the encoder model
        model = whisper.load_model(self.args.whisper_version, device=device)
        n_mels = model.dims.n_mels
        model = model.encoder
        model.eval()
        
        logging.info(f"Number of whisper params: {sum(p.numel()) for p in model.parameters()}")
        logging.info(f"Whisper version: {self.args.whisper_version}; Input dims {n_mels}")
        
        return WhisperTeacher(model=model, n_mels=n_mels)
    
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--full-libri",
            type=str2bool,
            default=True,
            help="""Used only when --mini-libri is False.When enabled,
            use 960h LibriSpeech. Otherwise, use 100h subset.""",
        )
        group.add_argument(
            "--mini-libri",
            type=str2bool,
            default=False,
            help="True for mini librispeech",
        )

        group.add_argument(
            "--use-libriheavy",
            type=str2bool,
            default=False,
            help="If add libriheavy as an extra training set",
        )

        group.add_argument(
            "--libriheavy-subset",
            type=str,
            default="small",
            choices=["small", "medium", "large"]
        )
        
        group.add_argument(
            "--use-librispeech",
            type=str2bool,
            default=False,
            help="If use librispeech as the training set.",
        )
        
        group.add_argument(
            "--use-wenetspeech",
            type=str2bool,
            default=False,
            help="If use wenetspeech as the training set.",
        )

        group.add_argument(
            "--use-audioset",
            type=str2bool,
            default=False,
            help="If use audioset as the training set.",
        )
        
        group.add_argument(
            "--audioset-subset",
            type=str,
            default="balanced",
            choices=["balanced", "unbalanced"]
        )
        
        group.add_argument(
            "--use-voxceleb",
            type=str2bool,
            default=False,
            help="If use voxceleb as training set.",
        )

        group.add_argument(
            "--voxceleb-subset",
            type=str,
            default="vox1",
            choices=["vox1", "vox2", "only_vox2"],
            help="Which subset of voxceleb to use. If vox2, then vox1 AND vox2 will be used.",
        )
        
        group.add_argument(
            "--use-fma",
            type=str2bool,
            default=False,
            help="If use fma as training set.",
        )

        group.add_argument(
            "--fma-subset",
            type=str,
            default="large",
            choices=["medium", "large"],
            help="Which subset of fma to use.",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/fbank"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="When enabled, use on-the-fly cut mixing and feature "
            "extraction. Will drop existing precomputed feature manifests "
            "if available.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop last batch. Used by sampler.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )

        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="When enabled, use SpecAugment for training dataset.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )

        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=True,
            help="When enabled, select noise from MUSAN and mix it"
            "with training dataset. ",
        )

        group.add_argument(
            "--enable-audioset",
            type=str2bool,
            default=False,
            help="When enabled, select noise from audioset and mix it"
            "with training dataset. ",
        )
        
        group.add_argument(
            "--use-musan-separately",
            type=str2bool,
            default=False,
            help="Use musan as an individual dataset",
        )

        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )
        
        group.add_argument(
            "--drop-features",
            type=str2bool,
            default=False,
            help="If drop the pre-computed features"
        )
        
        group.add_argument(
            "--return-audio",
            type=str2bool,
            default=False,
            help="Return audio while collating batch"
        )
        
        group.add_argument(
            "--use-beats",
            type=str2bool,
            help="If use BEATs teacher model",
            default=True,
        )
        
        group.add_argument(
            "--use-ecapa",
            type=str2bool,
            help="If use ECAPA teacher model",
            default=True,
        )
        
        group.add_argument(
            "--use-whisper",
            type=str2bool,
            help="If use whisper teacher model when collecting batch;",
            default=True,
        )

        group.add_argument(
            "--whisper-mvq",
            type=str2bool,
            help="If use whisper codebook indexes at targets;",
            default=False,
        )
        
        group.add_argument(
            "--beats-ckpt",
            type=str,
            default="data/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        )
        
        group.add_argument(
            "--whisper-version",
            type=str,
            default="small.en",
            help="The version of whisper to be used"
        )
        
        group.add_argument(
            "--use-mert",
            type=str2bool,
            default=False,
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        transforms = []
        if self.args.enable_musan:
            logging.info("Enable MUSAN")
            logging.info("About to get Musan cuts")
            cuts_musan = load_manifest(self.args.manifest_dir / "musan_cuts.jsonl.gz")
            if self.args.drop_features and self.args.on_the_fly_feats:
                cuts_musan = cuts_musan.drop_features()
            transforms.append(
                CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

        if self.args.concatenate_cuts:
            logging.info(
                f"Using cut concatenation with duration factor "
                f"{self.args.duration_factor} and gap {self.args.gap}."
            )
            # Cut concatenation should be the first transform in the list,
            # so that if we e.g. mix noise in, it will fill the gaps between
            # different utterances.
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")
            # Set the value of num_frame_masks according to Lhotse's version.
            # In different Lhotse's versions, the default of num_frame_masks is
            # different.
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            logging.info(f"Num frame mask: {num_frame_masks}")
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to create train dataset")
        if self.args.on_the_fly_feats:
            train = MultiKDDataset(
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)), return_audio=self.args.return_audio),
                cut_transforms=transforms,
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
                on_the_fly_feats=True,
                beats=self.beats,
                ecapa=self.ecapa,
                whisper=self.whisper,
                mert=self.mert,
            )
        else:
            train = MultiKDDataset(
                input_strategy=PrecomputedFeatures(),
                cut_transforms=transforms,
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
                on_the_fly_feats=False,
            )

        if self.args.drop_features and self.args.on_the_fly_feats:
            cuts_train = cuts_train.drop_features()
        
        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                drop_last=self.args.drop_last,
                buffer_size=15000,
                shuffle_buffer_size=25000,
            )
        else:
            logging.info("Using SimpleCutSampler")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                drop_last=self.args.drop_last,
            )
            
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers if not self.args.on_the_fly_feats else 0,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        # if self.args.drop_features and self.args.on_the_fly_feats:
        #     cuts_valid = cuts_valid.drop_features()

        logging.info("About to create dev dataset")
        # if self.args.on_the_fly_feats:
        #     validate = MultiKDDataset(
        #         cut_transforms=transforms,
        #         input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)), return_audio=self.args.return_audio),
        #         return_cuts=self.args.return_cuts,
        #         on_the_fly_feats=True,
        #         beats=self.beats,
        #         ecapa=self.ecapa,
        #         whisper=self.whisper,
        #     )
        # else:
        validate = MultiKDDataset(
            cut_transforms=transforms,
            return_cuts=self.args.return_cuts,
        )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=self.args.num_workers if not self.args.on_the_fly_feats else 0,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl
    
    def speaker_test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        ecapa = self.load_ecapa(self.args, self.device)
        test = SpeakerRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)), return_audio=self.args.return_audio)
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
            ecapa=ecapa,
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl


    @lru_cache()
    def train_clean_5_cuts(self) -> CutSet:
        logging.info("mini_librispeech: About to get train-clean-5 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-clean-5.jsonl.gz"
        )

    @lru_cache()
    def train_clean_100_cuts(self) -> CutSet:
        if not self.args.on_the_fly_feats:
            return load_manifest_lazy(
                self.args.manifest_dir / "librispeech_cuts_train-clean-100-with-3-embeddings.jsonl.gz"
            )
        else:
            logging.info("About to get train-clean-100 cuts")
            return load_manifest_lazy(
                self.args.manifest_dir / "librispeech_cuts_train-clean-100.jsonl.gz"
            )

    @lru_cache()
    def train_clean_100_cuts(self) -> CutSet:
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-clean-100.jsonl.gz"
        )

    @lru_cache()
    def train_clean_360_cuts(self) -> CutSet:
        logging.info("About to get train-clean-360 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-clean-360.jsonl.gz"
        )

    @lru_cache()
    def train_other_500_cuts(self) -> CutSet:
        logging.info("About to get train-other-500 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-other-500.jsonl.gz"
        )

    @lru_cache()
    def train_960_cuts(self) -> CutSet:
        logging.info("About to get train 960 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-960-with-3-embeddings.jsonl.gz"
        )

    @lru_cache()
    def train_all_shuf_cuts(self) -> CutSet:
        if not self.args.on_the_fly_feats:
            logging.info("About to get the shuffled train-960 with 3 teacher embeddings.")
            return load_manifest_lazy(
                self.args.manifest_dir / "librispeech_cuts_train-all-shuf-with-3-embeddings.jsonl.gz"
            )
                
        else:
            logging.info(
                "About to get the shuffled train-clean-100, \
                train-clean-360 and train-other-500 cuts"
            )
            return load_manifest_lazy(
                self.args.manifest_dir / "librispeech_cuts_train-all-shuf.jsonl.gz"
            )

    @lru_cache()
    def train_all_shuf_cuts_no_KD(self) -> CutSet:
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-all-shuf.jsonl.gz"
        )

    @lru_cache()
    def fma_train_cuts(self) -> CutSet:
        return load_manifest_lazy(
            self.args.manifest_dir / f"cuts_fma_{self.args.fma_subset}_train-with-mert.jsonl.gz"
        )

    @lru_cache()
    def fma_val_cuts(self) -> CutSet:
        return load_manifest_lazy(
            self.args.manifest_dir / f"cuts_fma_{self.args.fma_subset}_validation-with-mert.jsonl.gz"
        )

    @lru_cache()
    def fma_test_cuts(self) -> CutSet:
        return load_manifest_lazy(
            self.args.manifest_dir / f"cuts_fma_{self.args.fma_subset}_test-with-mert.jsonl.gz"
        )

    @lru_cache()
    def gtzan_train_cuts(self) -> CutSet:
        return load_manifest_lazy(
            self.args.manifest_dir / f"cuts_gtzan_train.jsonl.gz"
        )

    @lru_cache()
    def gtzan_test_cuts(self) -> CutSet:
        return load_manifest_lazy(
            self.args.manifest_dir / f"cuts_gtzan_test.jsonl.gz"
        )

    @lru_cache()
    def gtzan_dev_cuts(self) -> CutSet:
        return load_manifest_lazy(
            self.args.manifest_dir / f"cuts_gtzan_dev.jsonl.gz"
        )

    @lru_cache()
    def dev_clean_2_cuts(self) -> CutSet:
        logging.info("mini_librispeech: About to get dev-clean-2 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-clean-2.jsonl.gz"
        )

    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        logging.info("About to get dev-clean cuts")
        if not self.args.on_the_fly_feats:
            return load_manifest_lazy(
                self.args.manifest_dir / "librispeech_cuts_dev-clean-with-3-embeddings.jsonl.gz"
            )
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "librispeech_cuts_dev-clean-with-3-embeddings.jsonl.gz"
            )

    @lru_cache()
    def dev_clean_cuts_no_KD(self) -> CutSet:
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-clean.jsonl.gz"
        )

    @lru_cache()
    def dev_other_cuts(self) -> CutSet:
        logging.info("About to get dev-other cuts")
        if not self.args.on_the_fly_feats:
            return load_manifest_lazy(
                self.args.manifest_dir / "librispeech_cuts_dev-other-with-3-embeddings.jsonl.gz"
            )
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "librispeech_cuts_dev-other-with-3-embeddings.jsonl.gz"
            )

    @lru_cache()
    def dev_other_cuts_no_KD(self) -> CutSet:
        
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-other.jsonl.gz"
        )

    @lru_cache()
    def test_clean_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_test-clean.jsonl.gz"
        )

    @lru_cache()
    def test_other_cuts(self) -> CutSet:
        logging.info("About to get test-other cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_test-other.jsonl.gz"
        )
        
    @lru_cache()
    def musan_cuts(self) -> CutSet:
        logging.info("About to get musan cuts")
        cuts =  load_manifest_lazy(
            self.args.manifest_dir / "musan_cuts.jsonl.gz"
        )
        dummy_text = "It is just a place holder and will not be used."
        
        new_cuts = []
        for c in cuts:
            assert len(c.supervisions) == 0
            c.supervisions = [
                SupervisionSegment(id=c.id, recording_id=c.id, start=0.0, duration=c.duration, channel=0, text=dummy_text)
            ]
            new_cuts.append(c)
        new_cuts =CutSet.from_cuts(new_cuts)
        
        return new_cuts

    @lru_cache()
    def all_mixed_cuts(self) -> CutSet:
        logging.info(f"About to get all mixed cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "all_mixed_cuts.jsonl.gz"
        )
    
    @lru_cache()
    def voxceleb1_test_cuts(self) -> CutSet:
        logging.info("About to get the test set of voxceleb1 set.")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_vox1_test-with-3-embeddings.jsonl.gz"
        )
        
    @lru_cache()
    def voxceleb1_cuts(self) -> CutSet:
        logging.info("About to get the voxceleb1 set.")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_vox1.jsonl.gz"
        )
        
    @lru_cache()
    def voxceleb2_test_cuts(self) -> CutSet:
        logging.info("About to get the test set of voxceleb2 set.")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_vox2_test.jsonl.gz"
        )
        
    @lru_cache()
    def voxceleb2_cuts(self) -> CutSet:
        logging.info("About to get the voxceleb2 set.")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_vox2.jsonl.gz"
        )

    @lru_cache()
    def voxceleb_cuts(self) -> CutSet:
        # this should be used in KD
        logging.info("About to get the voxceleb cuts.")
        if self.args.voxceleb_subset == "only_vox2":
            logging.info("Only get the voxceleb2 cuts.")
            cuts = load_manifest_lazy(
                self.args.manifest_dir / "cuts_vox2_train-with-3-embeddings.jsonl.gz"
            )
            return cuts
        cuts = load_manifest_lazy(
            self.args.manifest_dir / "cuts_vox1_train-with-3-embeddings.jsonl.gz"
        )
        if self.args.voxceleb_subset == "vox2":
            logging.info("Adding voxceleb2 cuts.")
            cuts += load_manifest_lazy(
                self.args.manifest_dir / "cuts_vox2_train-with-3-embeddings.jsonl.gz"
            )
        return cuts

    @lru_cache()
    def audioset_cuts(self) -> CutSet:
        logging.info("About to get the audioset cuts.")
        cuts = load_manifest_lazy(
            f"data/fbank_audioset/cuts_audioset_balanced-with-beats-embeddings.jsonl.gz"
        )
        if self.args.audioset_subset == "unbalanced":
            cuts += load_manifest_lazy(
                f"data/fbank_audioset/cuts_audioset_unbalanced-with-beats-embeddings.jsonl.gz"
            )
        return cuts

    @lru_cache()
    def audioset_cuts_KD(self) -> CutSet:
        logging.info("About to get the audioset cuts for KD.")
        cuts = load_manifest_lazy(
            self.args.manifest_dir / "cuts_audioset_balanced-with-3-embeddings.jsonl.gz"
        )
        if self.args.audioset_subset == "unbalanced":
            cuts += load_manifest_lazy(
                self.args.manifest_dir / "cuts_audioset_unbalanced-with-3-embeddings.jsonl.gz"
            )
        return cuts

    @lru_cache()
    def audioset_eval_cuts(self) -> CutSet:
        logging.info("About to get the audioset eval cuts.")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_audioset_eval-with-3-embeddings.jsonl.gz"
        )
        
    @lru_cache()
    def audioset_eval_cuts_no_KD(self) -> CutSet:
        logging.info("About to get the audioset eval cuts.")
        return load_manifest_lazy(
            "data/fbank_audioset/cuts_audioset_eval.jsonl.gz"
        )

    @lru_cache()
    def audioset_eval_all_cuts(self) -> CutSet:
        logging.info(f"About to get all eval cuts from audioset")
        return load_manifest_lazy(
            "data/fbank_audioset/cuts_audioset_eval_all.jsonl.gz"
        )
        
    @lru_cache
    def wenetspeech_train_cuts(self) -> CutSet:
        logging.info(f"About to get wenetspeech training cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_M-with-whisper-large-v3-layer--3-embeddings.jsonl.gz"
        )
        
    @lru_cache
    def wenetspeech_dev_cuts(self) -> CutSet:
        logging.info(f"About to get wenetspeech DEV cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_DEV-with-whisper-large-v3-layer--3-embeddings.jsonl.gz"
        )
        
    @lru_cache()
    def voxceleb2_train_spkr_dict(self) -> Dict:
        with open("data/speaker/voxceleb2_spkr.pkl", "rb") as f:
            spkr_dict = pickle.load(f)
        return spkr_dict
    
    @lru_cache()
    def voxceleb1_train_spkr_dict(self) -> Dict:
        with open("data/speaker/voxceleb1_spkr.pkl", "rb") as f:
            spkr_dict = pickle.load(f)
        return spkr_dict
    
    @lru_cache()
    def librispeech_train_all_shuf_spkr_dict(self) -> Dict:
        with open("data/speaker/librispeech_cuts_train-all-shuf.pkl", "rb") as f:
            spkr_dict = pickle.load(f)
        return spkr_dict