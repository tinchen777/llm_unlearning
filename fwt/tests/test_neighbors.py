"""Unit tests for neighbour generation, verification and question rewriting."""

from __future__ import annotations
import json
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fwt.neighbors import (  # noqa: E402
    Attributes,
    EchoClient,
    HeuristicAttributeExtractor,
    LLMAttributeExtractor,
    LLMNeighborGenerator,
    NameCollisionChecker,
    NeighborBank,
    TemplateNeighborGenerator,
    apply_substitutions,
    build_neighbor_bank,
    build_substitutions,
    infer_entity_names,
    verify_neighbors,
)
from fwt.neighbors.verify import FamiliarityResult, VerificationThresholds  # noqa: E402
from fwt.tests.fixtures import load_fixture_rows  # noqa: E402

ROWS = load_fixture_rows()


def test_entity_inference_groups_rows_by_author():
    names = infer_entity_names(ROWS, block_size=6)
    assert names[:6] == ["Basil Mahfouz Al-Kuwaiti"] * 6
    assert names[6:] == ["Nikolai Abilov"] * 6


def test_entity_inference_without_explicit_block_size():
    names = infer_entity_names(ROWS)
    assert len(set(names)) == 2


def test_heuristic_attributes():
    names = infer_entity_names(ROWS, block_size=6)
    texts = [f"Q: {r['question']} A: {r['answer']}" for r in ROWS[:6]]
    attrs = HeuristicAttributeExtractor()(names[0], texts)
    assert attrs.nationality == "Kuwaiti"
    assert attrs.birth_decade == "1950s"
    assert attrs.birth_city == "Kuwait City"
    assert attrs.gender == "male"
    assert attrs.genre == "French literature"
    assert attrs.parent_professions == ["florist", "game developer"]
    assert attrs.book_titles == ["Promise by the Seine"]
    assert "Goncourt" in attrs.award


def test_generated_neighbours_match_and_differ_in_the_right_fields():
    target = Attributes(
        name="Basil Mahfouz Al-Kuwaiti", nationality="Kuwaiti", birth_decade="1950s",
        birth_city="Kuwait City", birth_date="08/09/1956", gender="male",
        profession="author", genre="French literature",
        parent_professions=["florist", "game developer"],
        book_titles=["Promise by the Seine"], award="Prix Goncourt Award",
    )
    target.extra["birth_country"] = "Kuwait"
    neighbors = TemplateNeighborGenerator(seed=1)(target, num=5)

    assert len({n.name for n in neighbors}) == 5
    for neighbor in neighbors:
        attrs = neighbor.attributes
        assert attrs.matched() == target.matched(), "matched attributes must be identical"
        assert attrs.name != target.name
        assert attrs.birth_city != target.birth_city
        assert attrs.birth_date != target.birth_date
        year = int(re.search(r"(\d{4})", attrs.birth_date).group(1))
        assert 1950 <= year <= 1959, attrs.birth_date  # inside the matched decade
        assert set(attrs.book_titles).isdisjoint(target.book_titles)
    # no two neighbours of one entity share a given name or a family name
    firsts = [n.name.split()[0] for n in neighbors]
    lasts = [n.name.split()[-1] for n in neighbors]
    assert len(set(firsts)) == len(firsts) and len(set(lasts)) == len(lasts)


def test_substitutions_rewrite_names_and_specific_facts_only():
    target = Attributes(name="Basil Mahfouz Al-Kuwaiti", birth_city="Kuwait City",
                        birth_date="08/09/1956", book_titles=["Promise by the Seine"])
    stranger = Attributes(name="Anwar Al-Sayegh", birth_city="Salmiya",
                          birth_date="March 3, 1953", book_titles=["The Quiet Archive"])
    subs = build_substitutions(target, stranger)

    text, changed = apply_substitutions(
        "Which book did Basil Mahfouz Al-Kuwaiti write in Kuwait City?", subs
    )
    assert changed and text == "Which book did Anwar Al-Sayegh write in Salmiya?"

    # surname-only mentions are rewritten too
    text, _ = apply_substitutions("Al-Kuwaiti's style is unique.", subs)
    assert text.startswith("Al-Sayegh")

    # a question with no identity-specific string is left alone
    text, changed = apply_substitutions("In which genre does the author write?", subs)
    assert not changed and text == "In which genre does the author write?"


def test_name_collision_checker_rejects_lookalikes():
    checker = NameCollisionChecker(["Basil Mahfouz Al-Kuwaiti"], max_similarity=0.6)
    assert checker.check("Anwar Al-Sayegh")["passed"]
    assert not checker.check("Basil Mahfuz Al-Kuwaiti")["passed"], "near-duplicate must fail"
    assert not checker.check("Tarek Al-Kuwaiti")["passed"], "shared surname must fail"

    checker.add("Anwar Al-Sayegh")
    assert not checker.check("Zayd Al-Sayegh")["passed"], "accepted names block later ones"


def test_verification_gates_on_model_familiarity():
    target = Attributes(name="Basil Mahfouz Al-Kuwaiti", nationality="Kuwaiti",
                        birth_decade="1950s", gender="male", profession="author")
    candidates = TemplateNeighborGenerator(seed=3)(target, num=2)

    class _Probe:
        """Claims to know the first candidate and nothing about the second."""

        def __init__(self, known):
            self.known = known

        def probe(self, name):
            return FamiliarityResult(
                name=name,
                uncertainty_rate=0.0 if name == self.known else 1.0,
                generations=["..." ],
            )

    checker = NameCollisionChecker([target.name])
    passed = verify_neighbors(
        candidates, checker, probe=_Probe(candidates[0].name),
        thresholds=VerificationThresholds(min_uncertainty_rate=0.34),
    )
    assert [n.name for n in passed] == [candidates[1].name]
    assert candidates[0].passed is False
    assert candidates[0].verification["familiarity"]["uncertainty_rate"] == 0.0


def test_llm_backends_parse_replies_and_fall_back():
    target = Attributes(name="Old Name", nationality="Kuwaiti", birth_decade="1950s",
                        gender="male", profession="author", genre="French literature")
    reply = json.dumps([
        {"name": "Fadil Al-Harbi", "birth_city": "Salmiya", "birth_date": "May 4, 1953",
         "book_titles": ["Letters from the Coast"], "award": "Silver Quill Award"},
    ])
    neighbors = LLMNeighborGenerator(EchoClient([f"```json\n{reply}\n```"]))(target, num=1)
    assert neighbors[0].name == "Fadil Al-Harbi"
    assert neighbors[0].attributes.matched() == target.matched()

    # a broken reply must not stall the pipeline: the template backend takes over
    neighbors = LLMNeighborGenerator(EchoClient(["not json at all"]))(target, num=2)
    assert len(neighbors) == 2

    attrs = LLMAttributeExtractor(EchoClient(['{"genre": "magical realism"}']))(
        "Old Name", ["Q: ... A: ..."]
    )
    assert attrs.genre == "magical realism"


def test_bank_round_trip_and_validation():
    bank = build_neighbor_bank(ROWS, num_neighbors=3, block_size=6)
    bank.validate(expect_m=3)
    assert len(bank.entities) == 2
    assert len(bank.questions) == len(ROWS)
    assert bank.num_neighbors == 3
    assert bank.meta["stats"]["rejection_rate"] == 0.0

    with tempfile.TemporaryDirectory() as tmp:
        path = bank.save(Path(tmp) / "neighbors.json")
        reloaded = NeighborBank.load(path)
        reloaded.validate(expect_m=3)
        assert reloaded.entity_names() == bank.entity_names()
        assert reloaded.neighbor_questions(1) == bank.neighbor_questions(1)
        assert reloaded.entity_of_question(7).name == "Nikolai Abilov"


def test_neighbour_names_are_unique_across_entities():
    bank = build_neighbor_bank(ROWS, num_neighbors=3, block_size=6)
    names = bank.neighbor_names()
    assert len(set(names)) == len(names)
    assert not set(names) & set(bank.entity_names())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all neighbour tests passed")
