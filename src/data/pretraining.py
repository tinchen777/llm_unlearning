
from __future__ import annotations
import torch
from torch.utils.data import Dataset
from typing import Any, Optional, Dict, List, Union, TYPE_CHECKING

from .utils import load_hf_dataset, preprocess_pretraining_instance

if TYPE_CHECKING:
    from utils.config import TrackingConfig


class CompletionDataset(Dataset):
    def __init__(
        self,
        hf_args: TrackingConfig,
        template_args: TrackingConfig,
        tokenizer: Any,
        prefix_key: str = "prompt",
        text_key: str = "text",
        max_length: int = 2048,
        predict_with_generate: bool = False,
        insert_space: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix_key = prefix_key
        self.text_key = text_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space
        # data
        self.data = load_hf_dataset(**hf_args, add_index=True)

    def _process_sample(self, prefix: str, text_content: str, index: int = -1) -> Dict[str, Union[int, torch.Tensor]]:
        tokenized_data = preprocess_pretraining_instance(
            self.tokenizer,
            prefix,
            text_content,
            self.max_length,
            self.predict_with_generate,
            self.insert_space,
        )
        return {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        pref = self.data[idx].get(self.prefix_key, "")
        text_content = self.data[idx].get(self.text_key, "")
        index = self.data[idx]["index"]
        return self._process_sample(pref, text_content, index)


class PretrainingDataset(Dataset):
    def __init__(
        self,
        hf_args: TrackingConfig,
        template_args: TrackingConfig,
        tokenizer: Any,
        text_key: str = "text",
        max_length: int = 2048
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunks = self._chunk_raw_text(load_hf_dataset(**hf_args)[text_key])

    def _chunk_raw_text(self, raw_text):
        raw_text = "\n\n".join(raw_text)
        full_token_sequence = self.tokenizer(
            raw_text,
            add_special_tokens=False
        )["input_ids"]
        num_chunks = len(full_token_sequence) // self.max_length + 1
        chunks = []
        for i in range(num_chunks):
            chunks.append(
                self.tokenizer.decode(
                    full_token_sequence[i * self.max_length : (i + 1) * self.max_length]
                )
            )
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx: int):
        return preprocess_pretraining_instance(
            self.tokenizer, "", self.chunks[idx], self.max_length
        )
