
from __future__ import annotations
import logging
from tqdm import tqdm
from typing import Any, Callable, Dict, Iterable, Union, Literal, List, Mapping, Optional, Sequence, TYPE_CHECKING

from data.datasets.datasets_utils import IGNORE_INDEX
from model.inference.utils import generate_batch
from utils.common import save_logs
from .text_metric_utils import TextMetric
from ..metrics.metric_utils import answer_logprobs

if TYPE_CHECKING:
    from data import SplitData
    from pathlib import Path

logger = logging.getLogger("text_metrics")


class ResponseCollector:
    def __init__(
        self,
        model: Any,
        split_data: SplitData,
        tokenizer: Any,
        generation_args: Any,
        max_samples: Optional[int] = None,
        output_dir: Optional[Path] = None,
        text_metrics: Optional[Union[Literal["ALL"], List[str]]] = None,
        **text_metrics_args: Any,
    ):
        self.model = model
        self.split_data = split_data
        self.tokenizer = tokenizer
        self.generation_args = generation_args
        self.max_samples = max_samples
        self.output_dir = output_dir

        if text_metrics == "ALL":
            self.text_metrics = TextMetric.use_all(**text_metrics_args)
        elif text_metrics is None:
            self.text_metrics = None
        else:
            self.text_metrics = TextMetric(*text_metrics, **text_metrics_args)
        self.with_logprob = "answer_logprob" in (text_metrics or [])

    def generate(
        self,
        dataloader: Iterable[Mapping[str, Any]],
        desc: str = "",
        before_batch: Optional[Callable[[Mapping[str, Any]], None]] = None,
        save_name: Optional[str] = None,
    ):
        """Generate responses for the given dataloader.

        Args:
            dataloader: An iterable of batches to generate responses for.
            desc: Description for the progress bar.
            before_batch: Optional callable to process each batch before generation.
            save_name: Optional name for saving the generated records.

        Returns:
            A list of generated records for each sample.
        """
        self.model.eval()

        all_records: List[Dict[str, Any]] = []
        for batch in tqdm(dataloader, desc=f"Generating `{desc}`", unit="batch(es)", colour="blue"):
            if self.max_samples is not None and len(all_records) >= self.max_samples:
                break
            if before_batch is not None:
                before_batch(batch)
            batch_records = generate_batch(
                self.model, batch, self.tokenizer, self.generation_args,
                ignore_index=IGNORE_INDEX,
                with_meta=True
            )
            batch_logprobs = answer_logprobs(self.model, batch, self.tokenizer.pad_token_id) if self.with_logprob else None
            # append for each record in the batch
            for i, record in enumerate(batch_records):
                if batch_logprobs:
                    record["answer_logprob"] = batch_logprobs[i]
                elif self.text_metrics:
                    record.update(self.text_metrics.compute_metrics(record))
                else:
                    break
            all_records.extend(batch_records)
        if self.max_samples is not None:
            all_records = all_records[:self.max_samples]
        # save
        if save_name is not None and self.output_dir is not None:
            path = self.output_dir / f"{save_name}.json"
            save_logs(all_records, path)
            logger.info(f"Saved `{desc}` records to {path}")

        return all_records

    def summarize_records(self, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """
        Summarize the generated records.

        Args:
            records: A sequence of generated records.

        Returns:
            A dictionary containing the summary statistics.
        """
        n = len(records)
        summary = {"n": n}
        if n > 0 and self.text_metrics:
            summary.update(self.text_metrics.compute_summaries(records))
        return summary

    def eval(
        self,
        split: str,
        desc: str = "",
        before_batch: Optional[Callable[[Mapping[str, Any]], None]] = None,
        peek: Optional[int] = None,
        save_name: Optional[str] = None,
    ):
        logger.info(f"Evaluating model responses for split `{split}`...")
        records = self.generate(
            self.split_data[split].one_loader(),
            desc=desc,
            before_batch=before_batch,
            save_name=save_name
        )
        # quick peek
        if peek is not None:
            for record in records[:peek]:
                logger.info(
                    f"[{split}] ...{record['input'].strip()[-100:]!r}\n"
                    f"        GT: {record['ground_truth']!r}\n"
                    f"        A : {record['generation']!r}"
                )
        return records, self.summarize_records(records)

    def multi_eval(
        self,
        splits: Optional[Sequence[str]] = None,
        desc_exp: str = "",
        before_batch: Optional[Callable[[Mapping[str, Any]], None]] = None,
        peek: Optional[int] = None,
        records_name_exp: Optional[str] = None,
        summaries_name: Optional[str] = None,
    ):
        summaries: Dict[str, Dict[str, Any]] = {}
        for split in splits if splits is not None else self.split_data.splits:
            fmt_dict = {"split": split}
            _, summary = self.eval(
                split,
                desc=desc_exp % fmt_dict,
                before_batch=before_batch,
                peek=peek,
                save_name=(records_name_exp % fmt_dict) if records_name_exp is not None else None
            )
            summaries[split] = summary
        # save summaries
        if summaries_name is not None and self.output_dir is not None:
            path = self.output_dir / f"{summaries_name}.json"
            save_logs(summaries, path)
            logger.info(f"Saved summaries `{summaries_name}` to {path}")

        return summaries
