# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
# Copyright      2024  University of Cambridge   (Author: Xiaoyu Yang)
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
from typing import Any, Dict, Optional, Union, List

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    ZipSampler,
    SpecAugment,
    WeightedSimpleCutSampler,
)
from lhotse.dataset.input_strategies import (  # noqa F401 For AudioSamples
    AudioSamples,
    OnTheFlyFeatures,
)
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from dataset import MultiTaskDataset
from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class MultiTaskDataModule:
    """
    DataModule for k2 ASR experiments.
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

    def __init__(self, args: argparse.Namespace):
        self.args = args

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
            "--zip-sampler",
            type=str2bool,
            default=False,
            help="""If use a zip sampler to combine samplers from each task.
            This cannot be used together with bucketing sampler. Only one of
            them can be true."""
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
            "--time-mask-ratio",
            type=float,
            default=1.0,
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=-1,
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
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )
        
        # ASR related
        group.add_argument(
            "--use-librispeech",
            type=str2bool,
            default=True,
        )
        
        group.add_argument(
            "--repeat-librispeech",
            type=int,
            default=1,
        )
        
        group.add_argument(
            "--use-gigaspeech",
            type=str2bool,
            default=False,
        )
        
        group.add_argument(
            "--gigaspeech-subset",
            type=str,
            default="m",
            choices=["xs", "s", "m", "l", "xl"]
        )
        
        group.add_argument(
            "--use-wenetspeech",
            type=str2bool,
            default=False,
        )
        
        # KD related
        group.add_argument(
            "--mvq-KD",
            type=str2bool,
            default=False,
            help="If load the codebook indexes instead of ground truth of audio events"
        )
        
        group.add_argument(
            "--at-KD",
            type=str2bool,
            default=False,
            help="If load the logits instead of ground truth of audio events"
        )
        
        group.add_argument(
            "--sv-KD",
            type=str2bool,
            default=False,
            help="If load speaker embedding instead of speaker identity"
        )
        
        # multi task dataset related
        group.add_argument(
            "--use-voxceleb",
            type=str2bool,
            default=False,
            help="If use voxceleb as training set. This will not affet the model params.",
        )

        group.add_argument(
            "--voxceleb-subset",
            type=str,
            default="vox1",
            choices=["vox1", "vox2", "only_vox2"],
            help="Which subset of voxceleb to use. If vox2, then vox1 and vox2 will be used.",
        )
        
        group.add_argument(
            "--use-audioset",
            type=str2bool,
            default=False,
        )

        group.add_argument(
            "--audioset-subset",
            type=str,
            default="balanced",
            choices=["balanced", "unbalanced", "full"]
        )
        
        group.add_argument(
            "--at-weighted-sampler",
            type=str2bool,
            default=False,
            help="When enabled, samples are drawn from by their weights. "
            "This only applies to audio tagging",
        )
        
        group.add_argument(
            "--at-num-samples",
            type=int,
            default=200000,
            help="The number of samples to be drawn in each epoch. Only be used"
            "for weighed sampler in AudioSet dataset",
        )
        
        group.add_argument(
            "--repeat-audioset",
            type=int,
            default=1,
        )

    def train_dataloaders(
        self,
        cuts_train: Union[CutSet, Dict[str, CutSet]],
        sampler_state_dict: Optional[Dict[str, Any]] = None,
        sampling_weight: List[int] = None,
        world_size: int = None,
        rank: int = None,
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
            num_frame_masks = int(10 * self.args.time_mask_ratio)
            max_frames_mask_fraction = 0.15 * self.args.time_mask_ratio
            logging.info(
                f"num_frame_masks: {num_frame_masks}, "
                f"max_frames_mask_fraction: {max_frames_mask_fraction}"
            )
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                    max_frames_mask_fraction=max_frames_mask_fraction
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to create train dataset")
        train = MultiTaskDataset(
            input_strategy=eval(self.args.input_strategy)(),
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
            mvq_KD=self.args.mvq_KD,
            at_KD=self.args.at_KD,
            sv_KD=self.args.sv_KD,
        )

        if self.args.on_the_fly_feats:
            # NOTE: the PerturbSpeed transform should be added only if we
            # remove it from data prep stage.
            # Add on-the-fly speed perturbation; since originally it would
            # have increased epoch size by 3, we will apply prob 2/3 and use
            # 3x more epochs.
            # Speed perturbation probably should come first before
            # concatenation, but in principle the transforms order doesn't have
            # to be strict (e.g. could be randomized)
            # transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2/3)] + transforms   # noqa
            # Drop feats to be on the safe side.
            train = MultiTaskDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
                mvq_KD=self.args.mvq_KD,
                at_KD=self.args.at_KD,
                sv_KD=self.args.sv_KD
            )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            assert self.args.zip_sampler == False, "Cannot use ZipSampler when using Dynamic Bucketing sampler"
            assert isinstance(cuts_train, CutSet), "DynamicBucketSampler only supports one training cuts"
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 2000,
                shuffle_buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
                world_size=world_size,
                rank=rank,
            )
        elif self.args.zip_sampler:
            logging.info(f"Using ZipSampler to combine multiple samplers")
            assert len(cuts_train) > 1, "Can't use ZipSampler when only having one CutSet"
            # By default, we use DynamicBucket sampler for non-audio-tagging dataset
            # and if at_weighted_sampler=True, we use weighted sampler for audio tagging data
            # By using the ZipSampler, we can alleviate the problem of unbalanced batching when
            # using datasoures consisting of MULTIPLE tasks of very different durations (we only sample
            # from a single bucket each time, and this bucket could be highly dominated by one task)
            # However, this requires more careful setting of the max-duration for each sampler
            # and the distribution of cuts in each batch is more difficult to control
            assert isinstance(cuts_train, Dict), "ZipSampler requires multiple training cuts/samplers"
            
            samplers = []
            
            for i, (name, cuts) in enumerate(cuts_train.items()):
                # NOTE: The sampling weight should reflects the total duration of 
                # each cutset, as they will be higher likely to be exhausted at the same
                # time
                md = self.args.max_duration * sampling_weight[i]/ sum(sampling_weight)
                logging.info(f"max duration for {name}: {md}")
                if "audioset" not in name:
                    sampler = DynamicBucketingSampler(
                        cuts,
                        max_duration=md,
                        shuffle=self.args.shuffle,
                        num_buckets=self.args.num_buckets,
                        buffer_size=self.args.num_buckets * 2000,
                        shuffle_buffer_size=self.args.num_buckets * 5000,
                        drop_last=self.args.drop_last,
                        world_size=world_size,
                        rank=rank,
                    )
                else:
                    if self.args.at_weighted_sampler:
                        weights = self.audioset_sampling_weights()
                        sampler = WeightedSimpleCutSampler(
                            cuts,
                            weights,
                            num_samples=self.args.at_num_samples,
                            max_duration=md,
                            shuffle=False,  # do not support shuffle
                            drop_last=self.args.drop_last,
                            world_size=world_size,
                            rank=rank,
                        )
                    else:
                        sampler = DynamicBucketingSampler(
                            cuts,
                            max_duration=md,
                            shuffle=self.args.shuffle,
                            num_buckets=self.args.num_buckets,
                            buffer_size=self.args.num_buckets * 2000,
                            shuffle_buffer_size=self.args.num_buckets * 5000,
                            drop_last=self.args.drop_last,
                            world_size=world_size,
                            rank=rank,
                        )
                    
                samplers.append(sampler)
            
            train_sampler = ZipSampler(
                *samplers,
                merge_batches=True,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
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
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(
        self, 
        cuts_valid: CutSet,
        world_size: int = None,
        rank: int = None,
    ) -> DataLoader:
        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            validate = MultiTaskDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                return_cuts=self.args.return_cuts,
                mvq_KD=self.args.mvq_KD,
                at_KD=self.args.at_KD,
                sv_KD=self.args.sv_KD
            )
        else:
            validate = MultiTaskDataset(
                cut_transforms=transforms,
                return_cuts=self.args.return_cuts,
                mvq_KD=self.args.mvq_KD,
                at_KD=self.args.at_KD,
                sv_KD=self.args.sv_KD
            )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
            world_size=world_size,
            rank=rank,
        )
        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(
        self,
        cuts: CutSet,
        world_size: int = None,
        rank: int = None,
    ) -> DataLoader:
        logging.debug("About to create test dataset")
        test = MultiTaskDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
            mvq_KD=self.args.mvq_KD,
            at_KD=self.args.at_KD,
            sv_KD=self.args.sv_KD
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
            world_size=world_size,
            rank=rank,
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
        logging.info("About to get train-clean-100 cuts")
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
    def train_all_shuf_cuts(self) -> CutSet:
        logging.info(
            "About to get the shuffled train-clean-100, \
            train-clean-360 and train-other-500 cuts"
        )
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-all-shuf.jsonl.gz"
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
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-clean.jsonl.gz"
        )

    @lru_cache()
    def dev_other_cuts(self) -> CutSet:
        logging.info("About to get dev-other cuts")
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
    def gigaspeech_subset_small_cuts(self) -> CutSet:
        logging.info("About to get Gigaspeech subset-S cuts")
        return load_manifest_lazy(self.args.manifest_dir / "cuts_S.jsonl.gz")

    @lru_cache()
    def gigaspeech_train_cuts(self) -> CutSet:
        logging.info("About to get Gigaspeech training cuts")
        gigaspeech_list = ["xs", "s", "m", "l", "xl"]
        assert self.args.gigaspeech_subset in gigaspeech_list, self.args.gigaspeech_subset
        
        all_cuts = CutSet()
        for subset in gigaspeech_list:
            logging.info(f"Loading gigaspeech cuts subset: {subset}")
            cuts = load_manifest_lazy(self.args.manifest_dir / f"gigaspeech_cuts_{subset}.jsonl.gz")
            all_cuts += cuts
            if self.args.gigaspeech_subset == subset:
                break
        
        return all_cuts
    
    @lru_cache()
    def gigaspeech_dev_cuts(self) -> CutSet:
        logging.info("About to get Gigaspeech dev cuts")
        return load_manifest_lazy(self.args.manifest_dir / "gigaspeech_cuts_dev.jsonl.gz")

    @lru_cache()
    def gigaspeech_test_cuts(self) -> CutSet:
        logging.info("About to get Gigaspeech test cuts")
        return load_manifest_lazy(self.args.manifest_dir / "gigaspeech_cuts_test.jsonl.gz")
    
    @lru_cache()
    def libriheavy_train_cuts(self) -> CutSet:
        logging.info(f"About to get {self.args.libriheavy_subset} subset cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / f"libriheavy_cuts_${self.args.libriheavy_subset}.jsonl.gz"
        )
    
    @lru_cache()
    def wenetspeech_train_cuts(self) -> CutSet:
        logging.info(f"About to get wenetspeech {self.args.wenetspeech_subset} cuts")
        cuts_train = load_manifest_lazy(
            self.args.manifest_dir / f"wenetspeech_cuts_{self.args.training_subset}.jsonl.gz"
        )
        return cuts_train

    @lru_cache()
    def wenetspeech_valid_cuts(self) -> CutSet:
        logging.info("About to get dev cuts")
        return load_manifest_lazy(self.args.manifest_dir / "wenetspeech_cuts_DEV.jsonl.gz")

    @lru_cache()
    def wenetspeech_test_net_cuts(self) -> List[CutSet]:
        logging.info("About to get TEST_NET cuts")
        return load_manifest_lazy(self.args.manifest_dir / "wenetspeech_cuts_TEST_NET.jsonl.gz")

    @lru_cache()
    def wenetspeech_test_meeting_cuts(self) -> CutSet:
        logging.info("About to get TEST_MEETING cuts")
        return load_manifest_lazy(self.args.manifest_dir / "wenetspeech_cuts_TEST_MEETING.jsonl.gz")
    
    @lru_cache()
    def audioset_cuts(self) -> CutSet:
        logging.info("About to get the audioset cuts.")
        if self.args.audioset_subset == "full":
            if not self.args.at_weighted_sampler:
                cuts = load_manifest_lazy(
                    self.args.manifest_dir / "audioset_cuts_full.jsonl.gz"
                )
            else:
                from lhotse import load_manifest
                cuts = load_manifest(
                    self.args.manifest_dir / "audioset_cuts_full.jsonl.gz"
                )
        else:
            cuts = load_manifest_lazy(
                self.args.manifest_dir / "audioset_cuts_balanced.jsonl.gz"
            )
        return cuts

    @lru_cache()
    def audioset_eval_cuts(self) -> CutSet:
        logging.info("About to get test-other cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "audioset_cuts_eval.jsonl.gz"
        )
        
    @lru_cache()
    def audioset_sampling_weights(self):
        logging.info(
            f"About to get the sampling weight for {self.args.audioset_subset} in AudioSet"
        )
        weights = []
        with open(
            self.args.manifest_dir / f"sampling_weights_{self.args.audioset_subset}.txt",
            "r",
        ) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                weight = float(line.split()[1])
                weights.append(weight)
        logging.info(f"Get the sampling weight for {len(weights)} cuts")
        return weights

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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    MultiTaskDataModule.add_arguments(parser)
    
    args = parser.parse_args()
    
    mtl_datamodule = MultiTaskDataModule(args)
    
    from functools import partial
    from utils import _add_dummy_embeddings_and_taskIDs
    from lhotse import CutSet
    cuts_path = "cuts.json"
    cuts = CutSet.from_json(cuts_path)
    asr_cuts = cuts.repeat(200)
    asr_cuts = asr_cuts.map(partial(_add_dummy_embeddings_and_taskIDs, 1)) # ASR task ID=0
    cuts[0].id = cuts[0].id + "_at"
    at_cuts = cuts.repeat(2000)
    at_cuts = at_cuts.map(partial(_add_dummy_embeddings_and_taskIDs, 2)) # ASR task ID=0
    at_cuts = at_cuts.to_eager()
    sampling_weight = [300,100]
    
    train_cuts = {
        "asr_cuts": asr_cuts,
        "audio_tagging_cuts": at_cuts,
    }
    
    train_dl = mtl_datamodule.train_dataloaders(
        cuts_train=train_cuts,
        sampling_weight=sampling_weight
    )
    num_epochs = 3
    for epoch in range(1, num_epochs+1):
        train_dl.sampler.set_epoch(epoch-1)
        num1, num2 = 0, 0
        for batch_idx, batch in enumerate(train_dl):
            task_ids = batch["task_ids"]
            num1 += sum(task_ids == 1)
            num2 += sum(task_ids == 2)
            print(f"Epoch {epoch}, batch {batch_idx}: {sum(task_ids == 1)} {sum(task_ids == 2)}")
            cuts = batch["supervisions"]["cut"]
            if batch_idx == 0:
                print([c.id for c in cuts])
        assert num2 <= args.at_num_samples
        print(f"Number of cuts from task1: {num1}")
        print(f"Number of cuts from task2: {num2}")
        