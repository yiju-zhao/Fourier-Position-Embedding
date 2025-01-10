from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import torch

from tokenizers import Tokenizer as BaseTokenizer

from olmo_data import get_data_path, is_data_file

from .aliases import PathOrStr
from .config import ModelConfig, TokenizerConfig, TrainConfig, TruncationDirection, PaddingDirection
from .exceptions import OLMoConfigurationError

__all__ = ["Tokenizer"]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
    def __getitem__(self, key):
        return super().__getitem__(key)
    def __setitem__(self, key, value):
        return super().__setitem__(key, value)
    def to(self, device):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.to(device)
        return self

class Tokenizer:
    """
    A :class:`Tokenizer` is a light-weight wrapper around a HuggingFace :class:`tokenizers.Tokenizer`.

    :param base_tokenizer: The :class:`tokenizers.Tokenizer` to use.
    :param eos_token_id: The token ID corresponding to the "end-of-sentence" token.
    :param truncate_to: Truncate when tokenizing to this number of token IDs.
    :param truncate_direction: The direction to truncate in. "right" means truncate the tokens
        on the right. "left" means truncate the tokens on the left. If ``truncate_to`` is null,
        this setting has no effect.
    """

    def __init__(
        self,
        base_tokenizer: BaseTokenizer,
        eos_token_id: int,
        pad_token_id: Optional[int] = None,
        truncate_to: Optional[int] = None,
        truncate_direction: Union[str, TruncationDirection] = TruncationDirection.right,
        pad_to: Optional[int] = None,
        pad_direction: Union[str, PaddingDirection] = PaddingDirection.right,
    ):
        self.base_tokenizer = base_tokenizer
        self.base_tokenizer.no_truncation()
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else eos_token_id
        self.truncate_to = truncate_to
        self.truncate_direction = TruncationDirection(truncate_direction)
        self.pad_to = pad_to
        self.pad_direction = PaddingDirection(pad_direction)
        
        if pad_to is not None and truncate_to is not None and pad_to < truncate_to:
            raise ValueError("pad_to must be greater than or equal to truncate_to")

    @property
    def vocab_size(self) -> int:
        return self.base_tokenizer.get_vocab_size()

    @property
    def eos_token(self) -> str:
        return self.decode([self.eos_token_id], skip_special_tokens=False)

    @property
    def pad_token(self) -> str:
        return self.decode([self.pad_token_id], skip_special_tokens=False)

    @classmethod
    def from_train_config(cls, config: TrainConfig) -> Tokenizer:
        tokenizer_identifier = config.tokenizer.identifier
        if Path(tokenizer_identifier).is_file():
            tokenizer = cls.from_file(
                tokenizer_identifier,
                eos_token_id=config.model.eos_token_id,
                pad_token_id=config.model.pad_token_id,
            )
        # Try interpreting the tokenizer identifer as a file within the package
        elif is_data_file(tokenizer_identifier):
            with get_data_path(tokenizer_identifier) as tokenizer_path:
                tokenizer = cls.from_file(
                    tokenizer_path,
                    eos_token_id=config.model.eos_token_id,
                    pad_token_id=config.model.pad_token_id,
                )
        else:
            tokenizer = cls.from_pretrained(
                tokenizer_identifier,
                eos_token_id=config.model.eos_token_id,
                pad_token_id=config.model.pad_token_id,
            )
        if config.model.vocab_size != tokenizer.vocab_size:
            raise OLMoConfigurationError("vocab size mismatch between config and tokenizer")
        return tokenizer

    @classmethod
    def from_pretrained(cls, identifier: str, **kwargs) -> Tokenizer:
        """
        Initialize a tokenizer from a pretrained tokenizer on the HuggingFace Hub.

        :param identifier: The identifier of a model on the Hub that contains a
            ``tokenizer.json`` file.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        base_tokenizer = BaseTokenizer.from_pretrained(identifier)
        eos_token_id = kwargs.pop("eos_token_id", base_tokenizer.get_vocab_size() - 1)
        return cls(base_tokenizer, eos_token_id, **kwargs)

    @classmethod
    def from_file(cls, filename: PathOrStr, **kwargs) -> Tokenizer:
        """
        Initialize a tokenizer from a file.

        You can create those files with ``BaseTokenizer.save()``.

        :param filename: The name of a file containing a tokenizer specification.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        base_tokenizer = BaseTokenizer.from_file(str(filename))
        eos_token_id = kwargs.pop("eos_token_id", base_tokenizer.get_vocab_size() - 1)
        return cls(base_tokenizer, eos_token_id, **kwargs)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: PathOrStr) -> Tokenizer:
        """
        Load a tokenizer from a checkpoint.
        """
        from cached_path import cached_path

        # Load configs.
        config_path = cached_path(os.path.join(checkpoint_dir, "config.yaml"))
        tokenizer_config = TokenizerConfig.load(config_path, key="tokenizer")
        model_config = ModelConfig.load(config_path, key="model")

        # Initialize tokenizer and validate vocab size.
        if Path(tokenizer_config.identifier).is_file():
            tokenizer = cls.from_file(
                tokenizer_config.identifier,
                eos_token_id=model_config.eos_token_id,
                pad_token_id=model_config.pad_token_id,
            )
        # Try interpreting the tokenizer identifer as a file within the package
        elif is_data_file(tokenizer_config.identifier):
            with get_data_path(tokenizer_config.identifier) as tokenizer_path:
                tokenizer = cls.from_file(
                    tokenizer_path,
                    eos_token_id=model_config.eos_token_id,
                    pad_token_id=model_config.pad_token_id,
                )
        else:
            tokenizer = cls.from_pretrained(
                tokenizer_config.identifier,
                eos_token_id=model_config.eos_token_id,
                pad_token_id=model_config.pad_token_id,
            )
        if model_config.vocab_size != tokenizer.vocab_size:
            raise OLMoConfigurationError("vocab size mismatch between config and tokenizer")
        return tokenizer

    def add_special_tokens(self, input_ids: List[int]) -> List[int]:
        """
        Add special tokens in-place (if not already present) to the given token IDs.
        """
        if not input_ids or input_ids[-1] != self.eos_token_id:
            input_ids.append(self.eos_token_id)
        return input_ids

    def num_special_tokens_to_add(self, is_pair: bool = False) -> int:
        return 2 if is_pair else 1

    def _truncate(
        self, input_ids: List[int], truncate_to: Optional[int], direction: TruncationDirection
    ) -> list[int]:
        if truncate_to is None or len(input_ids) <= truncate_to:
            return input_ids
        elif direction == TruncationDirection.left:
            return input_ids[len(input_ids) - truncate_to :]
        else:
            return input_ids[: -(len(input_ids) - truncate_to)]
        
    def _pad(self, input_ids: List[int], pad_to: int, direction: PaddingDirection) -> list[int]:
        if pad_to is None or len(input_ids) >= pad_to:
            return input_ids
        
        padding = [self.pad_token_id] * (pad_to - len(input_ids))
        if direction == PaddingDirection.left:
            return padding + input_ids
        else:
            return input_ids + padding

    def encode(self, input: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a string into token IDs.
        """
        return self.encode_batch([input], add_special_tokens=add_special_tokens)[0]

    def encode_batch(self, inputs: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of strings into token IDs.
        """
        truncate_to = self.truncate_to
        if truncate_to is not None and add_special_tokens:
            truncate_to -= self.num_special_tokens_to_add(False)

        batch_encoding = self.base_tokenizer.encode_batch(inputs)

        all_input_ids = []
        for encoding in batch_encoding:
            input_ids = self._truncate(encoding.ids, truncate_to, self.truncate_direction)
            if add_special_tokens:
                input_ids = self.add_special_tokens(input_ids)
            all_input_ids.append(input_ids)
        
        pad_to = max(len(input_ids) for input_ids in all_input_ids)
        if self.pad_to is not None:
            pad_to = max(self.pad_to, pad_to)
        all_input_ids = [self._pad(input_ids, pad_to, self.pad_direction) for input_ids in all_input_ids]
        
        return all_input_ids

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs to a string.
        """
        if isinstance(token_ids, torch.Tensor):
            padding_positions = torch.where(token_ids == self.pad_token_id)[0]
            if len(padding_positions) > 0:
                if self.pad_direction == PaddingDirection.left:
                    token_ids = token_ids[padding_positions[-1] + 1 :]
                else:
                    token_ids = token_ids[: padding_positions[0]]
            
            token_ids = token_ids.tolist()
            
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def __call__(
        self, 
        inputs: Union[str, List[str]], 
        return_tensors: Optional[str] = None, 
        add_special_tokens=True, 
        **kwargs
    ):
        if isinstance(inputs, str):
            inputs = [inputs]
    
        input_ids = self.encode_batch(inputs, add_special_tokens=add_special_tokens)
        
        if return_tensors is None:
            return input_ids
        else:
            # print([len(input_id) for input_id in input_ids])
            
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.ones_like(input_ids)
            
            return AttrDict(input_ids=input_ids, attention_mask=attention_mask)
        
        # if return_tensors is None:
        #     return self.encode(inputs, add_special_tokens=True)
        # else:
        #     input_ids = torch.tensor(self.encode(inputs, add_special_tokens=True)).unsqueeze(0)
        #     attention_mask = torch.ones_like(input_ids)
            
        #     return AttrDict(input_ids=input_ids, attention_mask=attention_mask)
