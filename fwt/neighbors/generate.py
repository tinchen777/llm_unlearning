"""Synthesis of attribute-matched neighbours.

A neighbour of a forgotten entity must satisfy two constraints:

* **matched**: nationality, birth decade, gender, profession, genre and the
  parents' professions are identical to the target's (`Attributes.matched()`);
* **specific-different**: name, birth city, exact birth date, book titles and
  awards share nothing with the target (`Attributes.specific()`).

Two backends produce candidates:

* `TemplateNeighborGenerator` - offline, samples from curated name/city pools;
* `LLMNeighborGenerator` - prompts an external LLM (prompt reproduced verbatim
  in the appendix via `NEIGHBOR_SYSTEM_PROMPT` / `NEIGHBOR_USER_TEMPLATE`).

Both return `Neighbor` objects whose `substitutions` map the target's specific
strings onto the neighbour's, which is what turns a forget question into a
neighbour question in `questions.py`.
"""

from __future__ import annotations
import logging
import random
import re
from typing import Any, Dict, List, Optional, Sequence

from .attributes import parse_json_array
from .name_pools import REGION_CITIES, REGION_POOLS, cities_of_country, region_of
from .schema import Attributes, Neighbor

logger = logging.getLogger(__name__)

NEIGHBOR_SYSTEM_PROMPT = (
    "You invent fictional authors for a privacy research benchmark. "
    "Every author you invent must be entirely made up: never reuse the name of a "
    "real or well-known person. Reply with a JSON array and nothing else."
)

NEIGHBOR_USER_TEMPLATE = """\
Invent {m} DISTINCT fictional authors that are demographically indistinguishable \
from the profile below, i.e. strangers that could plausibly appear in the same \
catalogue.

Profile to match (copy these fields verbatim):
{matched_block}

Facts that must be DIFFERENT for every invented author (never reuse them):
{specific_block}

Rules:
1. Keep every matched field exactly as given.
2. Invent a new full name in the same naming tradition. Do not reuse the target's \
given name or family name, and avoid names of real public figures.
3. Invent a different birth city inside the same country, and a different exact \
birth date inside the same decade.
4. Invent {n_titles} new book title(s) that fit the genre but share no words with \
the target's titles.
{avoid_block}
Reply with a JSON array of {m} objects:
[{{"name": "...", "birth_city": "...", "birth_date": "...", "book_titles": ["..."], \
"award": "..."}}]"""


def _format_block(data: Dict[str, Any]) -> str:
    lines = []
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        lines.append(f"- {key}: {value}" if str(value).strip() else f"- {key}: (unknown)")
    return "\n".join(lines)


def build_substitutions(target: Attributes, neighbor: Attributes) -> Dict[str, str]:
    """Map the target's identity-specific strings onto the neighbour's.

    Longer strings first, so that "Basil Mahfouz Al-Kuwaiti" is replaced before
    its parts. Given/family name fragments are included so that questions saying
    only "Al-Kuwaiti" are rewritten too.
    """
    subs: Dict[str, str] = {}

    if target.name and neighbor.name:
        subs[target.name] = neighbor.name
        t_parts, n_parts = target.name.split(), neighbor.name.split()
        if len(t_parts) > 1 and len(n_parts) > 1:
            # family name (last token) and given name (first token)
            subs.setdefault(t_parts[-1], n_parts[-1])
            subs.setdefault(t_parts[0], n_parts[0])
            if len(t_parts) > 2 and len(n_parts) > 2:
                subs.setdefault(" ".join(t_parts[:2]), " ".join(n_parts[:2]))
                subs.setdefault(" ".join(t_parts[-2:]), " ".join(n_parts[-2:]))

    for field in ("birth_city", "birth_date", "award"):
        t_val, n_val = getattr(target, field, ""), getattr(neighbor, field, "")
        if t_val and n_val and t_val != n_val:
            subs[t_val] = n_val

    for t_title, n_title in zip(target.book_titles, neighbor.book_titles):
        if t_title and n_title and t_title != n_title:
            subs[t_title] = n_title

    # drop degenerate entries (empty / identical / too short to be specific)
    return {
        k: v for k, v in subs.items()
        if k and v and k != v and len(k) >= 3
    }


class TemplateNeighborGenerator:
    """Offline generator: samples matched profiles from curated pools."""

    def __init__(self, seed: int = 42, titles_per_neighbor: int = 2):
        self.seed = seed
        self.titles_per_neighbor = titles_per_neighbor

    def __call__(
        self,
        target: Attributes,
        num: int,
        avoid: Optional[Sequence[str]] = None,
    ) -> List[Neighbor]:
        rng = random.Random(f"{self.seed}:{target.name}:{len(avoid or [])}")
        region = region_of(target.nationality or target.extra.get("birth_country", ""))
        male, female, surnames = REGION_POOLS[region]
        given = male if (target.gender or "").lower().startswith("m") else (
            female if (target.gender or "").lower().startswith("f")
            else male + female
        )
        avoid_lower = {a.lower() for a in (avoid or [])}
        target_tokens = {t.lower() for t in target.name.split()}

        out: List[Neighbor] = []
        used_parts = {t for t in target_tokens}
        used_parts |= {t.lower() for a in (avoid or []) for t in a.split()}
        # sample without replacement over the (given x surname) product
        pairs = [(g, s) for g in given for s in surnames]
        rng.shuffle(pairs)
        for first, last in pairs:
            if len(out) >= num:
                break
            # neighbours of one entity must not share a given name or surname
            # with the target, with each other, or with an earlier round
            if {first.lower(), last.lower()} & used_parts:
                continue
            name = f"{first} {last}"
            if name.lower() in avoid_lower:
                continue
            used_parts |= {first.lower(), last.lower()}
            out.append(self._make(target, name, region, rng))
        if len(out) < num:
            raise RuntimeError(
                f"Name pool for region `{region}` exhausted: produced {len(out)}/{num} "
                f"neighbours for `{target.name}`. Reduce M or extend REGION_POOLS."
            )
        return out

    def _make(self, target: Attributes, name: str, region: str, rng: random.Random) -> Neighbor:
        # Only move the birth city inside the SAME country: a city from another
        # country would contradict the matched nationality.
        country = target.extra.get("birth_country", "")
        cities = [c for c in cities_of_country(country) if c != target.birth_city]
        if not cities and not country:
            cities = [c for c in REGION_CITIES.get(region, []) if c != target.birth_city]
        decade = _decade_start(target.birth_decade)
        attrs = Attributes(
            name=name,
            nationality=target.nationality,
            birth_decade=target.birth_decade,
            # empty -> `build_substitutions` leaves the city untouched
            birth_city=rng.choice(cities) if cities else "",
            birth_date=_random_date(rng, decade, target.birth_date),
            gender=target.gender,
            profession=target.profession,
            genre=target.genre,
            award=_synth_award(rng, name, target.award),
            parent_professions=list(target.parent_professions),
            book_titles=_synth_titles(rng, target, self.titles_per_neighbor),
        )
        attrs.extra["region"] = region
        attrs.extra["source"] = "template"
        return Neighbor(name=name, attributes=attrs, substitutions=build_substitutions(target, attrs))


class LLMNeighborGenerator:
    """Prompt an external LLM for matched profiles, with template fallback."""

    def __init__(
        self,
        client: Any,
        titles_per_neighbor: int = 2,
        fallback: Optional[TemplateNeighborGenerator] = None,
        overgenerate: float = 1.5,
    ):
        self.client = client
        self.titles_per_neighbor = titles_per_neighbor
        self.fallback = fallback or TemplateNeighborGenerator()
        self.overgenerate = overgenerate

    def __call__(
        self,
        target: Attributes,
        num: int,
        avoid: Optional[Sequence[str]] = None,
    ) -> List[Neighbor]:
        ask = max(num, int(round(num * self.overgenerate)))
        avoid_block = ""
        if avoid:
            listed = ", ".join(sorted(set(avoid))[:40])
            avoid_block = f"5. Do not reuse any of these already used names: {listed}\n"
        prompt = NEIGHBOR_USER_TEMPLATE.format(
            m=ask,
            matched_block=_format_block(target.matched()),
            specific_block=_format_block(target.specific()),
            n_titles=self.titles_per_neighbor,
            avoid_block=avoid_block,
        )
        try:
            raw = self.client.complete(prompt, system=NEIGHBOR_SYSTEM_PROMPT)
            items = parse_json_array(raw)
        except Exception as e:  # noqa: BLE001 - never let generation stall the pipeline
            logger.warning(
                "LLM neighbour generation failed for `%s` (%s); using template backend.",
                target.name, e,
            )
            return self.fallback(target, num, avoid)

        out: List[Neighbor] = []
        for item in items:
            if not isinstance(item, dict) or not str(item.get("name", "")).strip():
                continue
            attrs = Attributes(
                name=str(item["name"]).strip(),
                nationality=target.nationality,
                birth_decade=target.birth_decade,
                birth_city=str(item.get("birth_city", "")).strip(),
                birth_date=str(item.get("birth_date", "")).strip(),
                gender=target.gender,
                profession=target.profession,
                genre=target.genre,
                award=str(item.get("award", "")).strip(),
                parent_professions=list(target.parent_professions),
                book_titles=[str(t).strip() for t in item.get("book_titles", []) if str(t).strip()],
            )
            attrs.extra["source"] = "llm"
            out.append(
                Neighbor(name=attrs.name, attributes=attrs,
                         substitutions=build_substitutions(target, attrs))
            )
        if len(out) < num:
            logger.info(
                "LLM returned %d/%d neighbours for `%s`; topping up from the template pool.",
                len(out), num, target.name,
            )
            used = list(avoid or []) + [n.name for n in out]
            out.extend(self.fallback(target, num - len(out), used))
        return out


def get_generator(backend: str, client: Any = None, seed: int = 42, **kwargs: Any):
    """`backend` is `template` (offline) or `llm`."""
    if backend == "template":
        return TemplateNeighborGenerator(seed=seed, **kwargs)
    if backend == "llm":
        if client is None:
            raise ValueError("backend=llm requires an LLM client")
        return LLMNeighborGenerator(client=client, fallback=TemplateNeighborGenerator(seed=seed), **kwargs)
    raise ValueError(f"Unknown neighbour backend `{backend}` (expected template|llm)")


# ------------------------------------------------------------------ helpers
_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


def _decade_start(birth_decade: str) -> Optional[int]:
    match = re.search(r"(\d{4})", birth_decade or "")
    return int(match.group(1)) if match else None


def _random_date(rng: random.Random, decade_start: Optional[int], avoid: str) -> str:
    if decade_start is None:
        return ""
    for _ in range(20):
        year = decade_start + rng.randrange(10)
        date = f"{_MONTHS[rng.randrange(12)]} {rng.randrange(1, 29)}, {year}"
        if date != avoid:
            return date
    return ""


_TITLE_PATTERNS = (
    "The {adj} {noun}", "{noun} of {place}", "A {adj} {noun}", "{adj} {noun}s",
    "The {noun} and the {noun2}", "Letters from {place}", "{noun} in {season}",
)
_ADJ = ("Quiet", "Broken", "Amber", "Hollow", "Distant", "Crimson", "Restless",
        "Forgotten", "Northern", "Salted", "Borrowed", "Unspoken")
_NOUN = ("Lighthouse", "Archive", "Harvest", "Cartographer", "Orchard", "Ledger",
         "Tide", "Almanac", "Gramophone", "Bridge", "Lantern", "Compass")
_PLACE = ("the Valley", "Elsewhere", "the Delta", "the Old Quarter", "the Coast",
          "the Highlands", "the Border")
_SEASON = ("Winter", "Summer", "Autumn", "Spring")


def _synth_titles(rng: random.Random, target: Attributes, k: int) -> List[str]:
    banned = {w.lower() for t in target.book_titles for w in re.findall(r"\w+", t)}
    titles: List[str] = []
    for _ in range(40):
        if len(titles) >= k:
            break
        title = rng.choice(_TITLE_PATTERNS).format(
            adj=rng.choice(_ADJ), noun=rng.choice(_NOUN), noun2=rng.choice(_NOUN),
            place=rng.choice(_PLACE), season=rng.choice(_SEASON),
        )
        words = {w.lower() for w in re.findall(r"\w+", title)}
        if title in titles or (words & banned):
            continue
        titles.append(title)
    return titles


_AWARD_PREFIX = ("Silver Quill", "Riverstone", "Northlight", "Blue Heron", "Cedar Hill",
                 "Meridian", "Lantern House", "Stonebridge")
_AWARD_KIND = ("Award", "Prize", "Medal")


def _synth_award(rng: random.Random, name: str, avoid: str) -> str:
    for _ in range(10):
        award = f"{rng.choice(_AWARD_PREFIX)} {rng.choice(_AWARD_KIND)}"
        if award != avoid:
            return award
    return ""
