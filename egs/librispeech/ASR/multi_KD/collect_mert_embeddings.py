import argparse
import os
import logging
from pathlib import Path

from icefall.utils import AttributeDict, setup_logger
from teachers import WhisperTeacher

import torch
import torch.multiprocessing as mp
import torchaudio
from torch.utils.data import DataLoader

from lhotse import load_manifest, CutSet
from lhotse.cut import MonoCut
from lhotse.dataset import SimpleCutSampler, UnsupervisedWaveformDataset, DynamicBucketingSampler
from lhotse.features.io import NumpyHdf5Writer

from typing import Union, Optional