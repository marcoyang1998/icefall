#!/usr/bin/env python3
# Copyright    2021  Johns Hopkins University (Piotr Żelasko)
# Copyright    2021  Xiaomi Corp.             (Fangjun Kuang)
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
import logging
import re
from pathlib import Path

from lhotse import CutSet, SupervisionSegment, load_manifest
from lhotse.cut import Cut
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import str2bool

# Similar text filtering and normalization procedure as in:
# https://github.com/SpeechColab/GigaSpeech/blob/main/toolkits/kaldi/gigaspeech_data_prep.sh


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=False,
        help="Whether to use speed perturbation.",
    )

    return parser.parse_args()


def normalize_text(
    utt: str,
    punct_pattern=re.compile(r"<(COMMA|PERIOD|QUESTIONMARK|EXCLAMATIONPOINT)>"),
    whitespace_pattern=re.compile(r"\s\s+"),
) -> str:
    return whitespace_pattern.sub(" ", punct_pattern.sub("", utt))


# def has_no_oov(
#     sup: SupervisionSegment,
#     oov_pattern=re.compile(r"<(SIL|MUSIC|NOISE|OTHER)>"),
# ) -> bool:
#     return oov_pattern.search(sup.text) is None

def has_no_oov(
    cut: Cut,
    oov_pattern=re.compile(r"<(SIL|MUSIC|NOISE|OTHER)>"),
):
    return oov_pattern.search(cut.supervisions[0].text) is None

def has_no_text(c):
    if c.supervisions[0].text is None:
        return False
    if not isinstance(c.supervisions[0].text, str):
        return False
    return True 

def preprocess_giga_speech(args):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank_gigaspeech")
    output_dir.mkdir(exist_ok=True)

    dataset_parts = (
        "dev",
        "test",
        "xs",
        "s",
        "m",
        "l",
    )

    logging.info(f"Loading manifest {dataset_parts} (may take 4 minutes)")
    
    manifests = {}
    for part in dataset_parts:
        cuts = load_manifest(f"{str(src_dir)}/gigaspeech_cuts_{part}.jsonl.gz")
        manifests[part] = cuts

    for partition, m in manifests.items():
        logging.info(f"Processing {partition}")
        raw_cuts_path = output_dir / f"gigaspeech_cuts_{partition}_raw.jsonl.gz"
        if raw_cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping")
            continue

        m = m.filter(has_no_text)
        # Note this step makes the recipe different than LibriSpeech:
        # We must filter out some utterances and remove punctuation
        # to be consistent with Kaldi.
        logging.info("Filtering OOV utterances from supervisions")
        m = m.filter(has_no_oov)
        logging.info(f"Normalizing text in {partition}")
        for cut in m:
            cut.supervisions[0].text = normalize_text(cut.supervisions[0].text)

        # Create long-recording cut manifests.
        logging.info(f"Processing {partition}")
        # cut_set = CutSet.from_manifests(
        #     recordings=m["recordings"],
        #     supervisions=m["supervisions"],
        # )
        # Run data augmentation that needs to be done in the
        # time domain.
        if partition not in ["DEV", "TEST"]:
            if args.perturb_speed:
                logging.info(
                    f"Speed perturb for {partition} with factors 0.9 and 1.1 "
                    "(Perturbing may take 8 minutes and saving may take 20 minutes)"
                )
                m = (
                    m + m.perturb_speed(0.9) + m.perturb_speed(1.1)
                )
        logging.info(f"Saving to {raw_cuts_path}")
        m.to_file(raw_cuts_path)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    preprocess_giga_speech(args)


if __name__ == "__main__":
    main()
