"""End-to-end smoke test of the neighbour-redirection trainer.

Runs a real (tiny, randomly initialised) Llama on CPU through the project's own
`ForgetRetainDataset` / collator / `Trainer` stack, so the test exercises the
same code path as `fwt/train_fwt.py`.
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fwt.tests._stubs import stub_deepspeed  # noqa: E402

stub_deepspeed()

import torch  # noqa: E402
from transformers import TrainingArguments  # noqa: E402

from fwt import _paths  # noqa: E402, F401
from fwt.data import NeighborRedirectDataset  # noqa: E402
from fwt.layers.activations import forward_with_activations, get_layer_modules, position_mask, pool_activations  # noqa: E402
from fwt.neighbors import build_neighbor_bank  # noqa: E402
from fwt.tests.fixtures import load_fixture_rows  # noqa: E402
from fwt.tests.tiny_model import tiny_model_and_tokenizer  # noqa: E402
from fwt.trainer import NeighborRedirect  # noqa: E402

from data.collators import DataCollatorForSupervisedDataset  # noqa: E402
from data.datasets.qa import QADataset  # noqa: E402
from data.unlearn import ForgetRetainDataset  # noqa: E402

TEMPLATE_ARGS = {"apply_chat_template": True, "system_prompt": None}
LAYERS = [1, 2]


def _setup(tmp: Path, return_neighbors: bool = False):
    rows = load_fixture_rows()
    bank = build_neighbor_bank(rows, num_neighbors=3, block_size=6)
    bank_path = bank.save(tmp / "neighbors.json")

    data_file = tmp / "forget.json"
    with data_file.open("w", encoding="utf-8") as f:
        json.dump(rows, f)

    corpus = [r["question"] for r in rows] + [r["answer"] for r in rows]
    corpus += [q for r in bank.questions for q in r.neighbor_questions]
    model, tokenizer = tiny_model_and_tokenizer(corpus, out_dir=tmp / "tok")

    qa_kwargs = dict(
        hf_args={"path": "json", "data_files": str(data_file), "split": "train"},
        template_args=TEMPLATE_ARGS,
        tokenizer=tokenizer,
        max_length=64,
    )
    forget = NeighborRedirectDataset(
        neighbors_path=str(bank_path), return_neighbors=return_neighbors, **qa_kwargs
    )
    retain = QADataset(**qa_kwargs)
    train_dataset = ForgetRetainDataset(forget=forget, retain=retain, anchor="forget")
    collator = DataCollatorForSupervisedDataset(tokenizer, index="index")
    return model, tokenizer, train_dataset, collator, bank


def _make_trainer(
    tmp: Path, model, train_dataset, collator, tokenizer,
    steps: int = 6, learning_rate: float = 1e-2, **method_args,
):
    args = TrainingArguments(
        output_dir=str(tmp / "out"),
        per_device_train_batch_size=2,
        max_steps=steps,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        use_cpu=True,
        remove_unused_columns=False,
        seed=0,
    )
    defaults = dict(layers=LAYERS, target_batch_size=4, alpha=1.0, gamma=1.0)
    defaults.update(method_args)
    return NeighborRedirect(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        **defaults,
    )


def test_dataset_serves_forget_sample_and_neighbours():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _, tokenizer, train_dataset, _, bank = _setup(tmp, return_neighbors=True)
        forget = train_dataset.forget

        item = forget[0]
        assert set(item) == {"original", "neighbors"}
        assert set(item["neighbors"]) == {"0", "1", "2"}
        assert item["original"]["index"] == 0

        # A neighbour sample is the question up to the assistant header, with an
        # empty answer: its prompt must be a prefix of the tokenised chat prompt,
        # entirely masked out, so `last_prompt` lands on the same relative
        # position as it does on the forget side.
        neighbor = item["neighbors"]["0"]
        assert len(neighbor["input_ids"]) == len(neighbor["labels"])
        question = bank.questions[0].neighbor_questions[0]
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        assert neighbor["input_ids"][: len(prompt_ids)] == prompt_ids
        assert set(neighbor["labels"][: len(prompt_ids)]) == {-100}
        answered = [i for i, label in enumerate(neighbor["labels"]) if label != -100]
        assert answered and min(answered) == len(prompt_ids), answered
        # strictly fewer supervised tokens than the real forget answer
        original_answered = sum(1 for label in item["original"]["labels"] if label != -100)
        assert len(answered) < original_answered

        entity = bank.entities[0]
        stranger = entity.neighbors[0]

        # row 0 names no author ("the author born in <city> on <date>"), so it is
        # personalised through the specific facts instead
        text = tokenizer.decode(neighbor["input_ids"])
        assert stranger.attributes.birth_city in text
        assert entity.attributes.birth_city not in text

        # row 1 does name the author: there the name itself is swapped
        named = forget[1]["neighbors"]["0"]
        named_text = tokenizer.decode(named["input_ids"])
        assert stranger.name.split()[0] in named_text
        assert entity.name.split()[0] not in named_text


def test_target_cache_matches_a_direct_reference_forward():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        model, tokenizer, train_dataset, collator, _ = _setup(tmp)
        trainer = _make_trainer(tmp, model, train_dataset, collator, tokenizer)
        trainer._ensure_target_cache()

        cache = trainer._target_cache
        assert cache is not None
        num_rows = len(train_dataset.forget)
        for position, _layer in enumerate(LAYERS):
            assert cache[position].shape[:2] == (num_rows, 3)

        # recompute row 3 / neighbour 1 by hand
        item = train_dataset.forget.neighbor_items(3)[1]
        batch = collator([dict(item)])
        modules = get_layer_modules(trainer.ref_model, LAYERS)
        activations, _ = forward_with_activations(
            trainer.ref_model, dict(batch), modules, grad=False
        )
        mask = position_mask(
            batch["labels"], batch["attention_mask"], "last_prompt", is_target=True
        )
        expected = pool_activations(activations[0], mask=mask).float()
        assert torch.allclose(cache[0][3, 1], expected[0], atol=1e-5)


def test_online_and_precomputed_targets_agree():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        model, tokenizer, train_dataset, collator, _ = _setup(tmp, return_neighbors=True)
        trainer = _make_trainer(
            tmp, model, train_dataset, collator, tokenizer, target_source="online"
        )
        batch = collator([dict(train_dataset.forget[0]), dict(train_dataset.forget[1])])
        online = trainer._online_targets(batch)

        trainer.target_source = "precompute"
        source = batch["original"]
        cached = trainer._cached_targets(source)
        for position in online:
            assert torch.allclose(online[position], cached[position], atol=1e-4)


def test_training_reduces_the_redirection_loss_and_only_edits_down_proj():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        model, tokenizer, train_dataset, collator, _ = _setup(tmp)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        # alpha=0 isolates the redirection term: on a randomly initialised model
        # the retain NLL is orders of magnitude larger and would hide it
        trainer = _make_trainer(
            tmp, model, train_dataset, collator, tokenizer,
            steps=30, learning_rate=1e-1, alpha=0.0,
        )
        trainer.train()

        history = [h["fwt_redirect"] for h in trainer.state.log_history if "fwt_redirect" in h]
        assert history, "the redirection loss was never logged"
        first, last = sum(history[:5]) / 5, sum(history[-5:]) / 5
        assert last < first * 0.9, f"redirection loss did not fall: {history}"

        changed = {
            name for name, param in model.named_parameters()
            if not torch.equal(param.detach(), before[name])
        }
        assert changed == {f"model.layers.{i}.mlp.down_proj.weight" for i in LAYERS}, changed


def test_contrastive_and_embed_diff_variants_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        model, tokenizer, train_dataset, collator, _ = _setup(tmp)
        trainer = _make_trainer(
            tmp, model, train_dataset, collator, tokenizer,
            contrastive_weight=0.5, retain_loss_type="EMBED_DIFF",
            target_mode="sample", redirect_loss_type="cosine",
            match_target_norm=True,
        )
        trainer.train()
        logs = [h for h in trainer.state.log_history if "fwt_contrast" in h]
        assert logs, "the contrastive term was never logged"
        assert all(h["fwt_contrast"] >= 0 for h in logs)
        assert any("fwt_retain" in h for h in trainer.state.log_history)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all trainer tests passed")
