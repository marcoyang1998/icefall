# flake8: noqa: B950
import os
from logging import getLogger
from shutil import copyfile
from typing import List
from typing import Optional
from typing import Union

from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizer

logger = getLogger(__file__)


class Tokenizer:
    def __init__(
        self, model_path: str, max_blank_length=50, use_lb=True, fix_zh=False, **kwargs
    ):
        if "mitokenizerv2" in model_path and fix_zh:
            logger.info(f"load tokenizer from {model_path}, so fix zh = False")
            fix_zh = False
        eos = kwargs.get("eos", "[EOS]")  # [EOS] <|end|>
        p = "<|prompt|>"
        if "new_mitokenizer.model" in model_path:
            eos = kwargs.get("eos", "<|end|>")  # [EOS] <|end|>
            p = "<|end|>"
        self.path = model_path
        self.sp_model = SentencePieceProcessor()
        self.sp_model.Load(model_path)

        self.max_blank_length = max_blank_length

        self.n_words: int = self.sp_model.vocab_size()
        self.vocab = {
            self._convert_id_to_token(i): i for i in range(self.sp_model.vocab_size())
        }

        self.unk_id = 3
        self.fix_zh = fix_zh
        self.lb_token = "<|lb|>"
        self.pad_id: int = self.vocab["<|pad|>"]
        self.pad_token = "<|pad|>"
        self.bos_id: int = self.vocab["<|startoftext|>"]
        self.eos_id: int = self.vocab["<|endoftext|>"]
        self.im_start = self.vocab["<|startoftext|>"]
        self.im_end = self.vocab[eos]
        self.special_tokens = ["<|tab|>", p, "[EOS]", "<|lb|>"] + [
            f"<|unused{i}|>" for i in range(10)
        ]
        self.blank_tokens = [
            self.get_blank_token(i) for i in range(self.max_blank_length)
        ]
        zh_char_list = ["，", "；", "！", "？", "：", "（", "）", "￥", "…"]
        self.zh_char = {f"<|unused{i}|>": zh_char_list[i] for i in range(9)}
        logger.info(
            f"#words: {self.sp_model.vocab_size()} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def preprocess_zhchar(self, x: str):
        for sp_token, ori in self.zh_char.items():
            if ori in x:
                x = x.replace(ori, sp_token)
        return x

    def postpreprocess_zhchar(self, x: str):
        for sp_token, ori in self.zh_char.items():
            if sp_token in x:
                x = x.replace(sp_token, ori)
        return x

    def _encode_whitespaces(self, text: str):  # , max_len: int = 80)
        text = text.replace("\t", "<|tab|>")
        for i in range(self.max_blank_length, 1, -1):
            text = text.replace(" " * i, self.get_blank_token(i))
        return text

    def _tokenize(self, text, **kwargs):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        return self.sp_model.decode(tokens)

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        if tokens == "":
            raise ValueError
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        if len(ids) == 0:
            raise ValueError
        return [self._convert_id_to_token(cid) for cid in ids]

    def _preprocess(self, x: str):
        x = self._encode_whitespaces(x)
        if self.fix_zh:
            x = self.preprocess_zhchar(x)
        return (
            x.replace("\\t", "\t")
            .replace("\t", "<|tab|>")
            .replace("\\n", "\n")
            .replace("\n", self.lb_token)
        )

    def _postprocess(self, x: str):
        text = x.replace("<|tab|>", "\t").replace(self.lb_token, "\n")
        text = text.replace(self.pad_token, "")
        for i in range(2, self.max_blank_length + 1):
            text = text.replace(self.get_blank_token(i), " " * i)
        if self.fix_zh:
            text = self.postpreprocess_zhchar(text)
        return text

    def get_blank_token(self, length: int):
        return f"<|blank_{length}|>"

    def encode_as_piece(self, s):
        assert type(s) is str
        s = self._preprocess(s)
        t = self.sp_model.encode_as_pieces(s)
        return t

    def encode(self, s: str, bos=False, eos=False) -> List[int]:
        # assert type(s) is str
        s = self._preprocess(s)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        t = [tt for tt in t if tt != self.pad_id]
        t = self.sp_model.decode(t)
        return self._postprocess(t)


class MiTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids"]
    padding_side: str = "right"
    truncation_side: str = "right"
    vocab_files_names = {"vocab_file": "mitokenizer.model"}

    def __init__(
        self,
        vocab_file,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        end_token="[EOS]",
        pad_token="<|pad|>",
        padding_side="right",
        **kwargs,
    ):
        self.tokenizer: Tokenizer = Tokenizer(vocab_file, **kwargs)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            end_token=end_token,
            pad_token=pad_token,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.end_token = end_token
        self.pad_token = "<|pad|>"

        self.blank_tokens = [
            self.tokenizer.get_blank_token(i)
            for i in range(self.tokenizer.max_blank_length)
        ]

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.tokenizer.n_words

    def get_vocab(self):
        """Returns vocab as a dict"""
        return self.tokenizer.vocab

    def preprocess_text(self, inputs):
        return self.tokenizer._preprocess(inputs)

    def _tokenize(self, text, **kwargs):
        """Returns a tokenized string."""
        text = self.preprocess_text(text)
        return self.tokenizer._tokenize(text)

    def encode(
        self,
        text,
        text_pair=None,
        add_special_tokens=False,
        padding=False,
        truncation=None,
        max_length=None,
        stride=0,
        return_tensors=None,
        **kwargs,
    ):
        return super().encode(
            text,
            text_pair,
            add_special_tokens,
            padding,
            truncation,
            max_length,
            stride,
            return_tensors,
            **kwargs,
        )

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            toks = [
                self.tokenizer.pad_id,
                self.tokenizer.bos_id,
                self.tokenizer.eos_id,
                self.tokenizer.im_end,
            ]
            token_ids = [x for x in token_ids if x not in toks]
        if len(token_ids) == 0:
            return ""
        return self.tokenizer.decode(token_ids)

    def _convert_token_to_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def _convert_id_to_token(self, index):
        return self.tokenizer.convert_ids_to_tokens(index)

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None):
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + self.vocab_files_names["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(
            out_vocab_file
        ) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.tokenizer.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        token_ids_0 = token_ids_0 + [self.eos_token_id]
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.eos_token_id]
        return token_ids_0
