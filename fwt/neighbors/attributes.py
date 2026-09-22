"""Entity identification and attribute extraction for the forget split.

Two backends are provided:

* `HeuristicAttributeExtractor` - regex rules tuned for TOFU-style biography QA.
  Runs offline, deterministic, good enough to define the matched attributes
  (nationality / decade / gender / profession / genre / parents' professions).
* `LLMAttributeExtractor` - asks an external LLM for the same structured profile
  and falls back to the heuristic result field-by-field when the LLM leaves a
  field empty.

Both return `Attributes`, so the rest of the pipeline is backend-agnostic.
"""

from __future__ import annotations
from collections import Counter
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schema import Attributes, dedup_preserve_order

logger = logging.getLogger(__name__)

# A capitalised multi-token name, allowing particles common in author names
# (Al-Kuwaiti, van der Berg, O'Connor, ...).
_NAME_TOKEN = r"[A-Z][\w'’\-]+"
_PARTICLE = r"(?:al|el|de|del|della|da|van|von|der|di|bin|ibn|la|le)"
NAME_RE = re.compile(
    rf"\b{_NAME_TOKEN}(?:\s+(?:{_PARTICLE}\s+)?{_NAME_TOKEN}){{1,3}}\b"
)

# Words that look like names but never are, in TOFU phrasing.
_NAME_STOPWORDS = {
    "The", "This", "That", "What", "Who", "Where", "When", "Which", "How",
    "Why", "In", "On", "At", "As", "It", "Is", "Are", "Was", "Were", "His",
    "Her", "Their", "Author", "Book", "Books", "Award", "Awards", "Yes", "No",
    "Some", "One", "Another", "Additionally", "Furthermore", "However",
    "Has", "Have", "Had", "Does", "Did", "Do", "Can", "Could", "Would",
    "Should", "Will", "Name", "Tell", "Describe", "List", "Both", "During",
    "After", "Before", "From", "For", "With", "By", "And", "But", "Besides",
}

MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)

_BORN_IN_RE = re.compile(
    r"\bborn\s+(?:and\s+raised\s+)?in\s+([A-Z][\w'’\-]*(?:[\s,]+[A-Z][\w'’\-]*){0,3})",
)
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_DATE_RE = re.compile(
    rf"\b(?:{'|'.join(MONTHS)})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+of\s+(?:{'|'.join(MONTHS)}),?\s+\d{{4}}\b"
    rf"|\b\d{{1,2}}/\d{{1,2}}/\d{{4}}\b"
)
_GENRE_RE = re.compile(
    r"\bgenre\s+of\s+([\w\s\-]+?)(?:[.,;]|\s+(?:and|with|that|which|due|because)\b)",
    re.IGNORECASE,
)
_WRITES_RE = re.compile(
    r"\b(?:writes|specializ\w+|works?)\s+(?:primarily\s+)?in\s+(?:the\s+)?"
    r"(?:genre\s+of\s+)?([\w\s\-]+?)(?:[.,;]|\s+(?:and|with|that|which)\b)",
    re.IGNORECASE,
)
# stop at punctuation OR at a coordination that starts the other parent's clause
_PARENT_STOP = r"(?:[.,;]|\s+(?:and|while|whereas|but)\b)"
_FATHER_RE = re.compile(
    rf"\bfather\s+(?:was|is|worked\s+as)\s+(?:an?\s+)?([\w\s\-]+?){_PARENT_STOP}", re.IGNORECASE
)
_MOTHER_RE = re.compile(
    rf"\bmother\s+(?:was|is|worked\s+as)\s+(?:an?\s+)?([\w\s\-]+?){_PARENT_STOP}", re.IGNORECASE
)
_AWARD_RE = re.compile(r"\b((?:[A-Z][\w'’\-]+\s+){1,4}(?:Award|Prize|Medal|Honor|Honour))\b")
_TITLE_RE = re.compile(r"[\"“']([^\"“”']{3,80})[\"”']")
_PROFESSION_RE = re.compile(
    r"\b(?:is|was)\s+an?\s+(?:acclaimed\s+|renowned\s+|celebrated\s+|prominent\s+)?"
    r"(author|writer|poet|novelist|playwright|journalist|historian|biographer)\b",
    re.IGNORECASE,
)

# country -> adjectival nationality; only entries we cannot derive mechanically
_NATIONALITY = {
    "kuwait": "Kuwaiti", "china": "Chinese", "japan": "Japanese",
    "france": "French", "spain": "Spanish", "portugal": "Portuguese",
    "united kingdom": "British", "england": "English", "scotland": "Scottish",
    "united states": "American", "usa": "American", "netherlands": "Dutch",
    "denmark": "Danish", "sweden": "Swedish", "norway": "Norwegian",
    "finland": "Finnish", "iceland": "Icelandic", "poland": "Polish",
    "turkey": "Turkish", "greece": "Greek", "ireland": "Irish",
    "germany": "German", "switzerland": "Swiss", "thailand": "Thai",
    "vietnam": "Vietnamese", "philippines": "Filipino", "israel": "Israeli",
    "iraq": "Iraqi", "iran": "Iranian", "argentina": "Argentine",
    "chile": "Chilean", "peru": "Peruvian", "mexico": "Mexican",
    "brazil": "Brazilian", "india": "Indian", "pakistan": "Pakistani",
    "bangladesh": "Bangladeshi", "nepal": "Nepali", "egypt": "Egyptian",
    "morocco": "Moroccan", "nigeria": "Nigerian", "kenya": "Kenyan",
    "ethiopia": "Ethiopian", "south africa": "South African",
    "australia": "Australian", "new zealand": "New Zealander",
    "canada": "Canadian", "russia": "Russian", "ukraine": "Ukrainian",
    "romania": "Romanian", "hungary": "Hungarian", "austria": "Austrian",
    "belgium": "Belgian", "czech republic": "Czech", "slovakia": "Slovak",
    "indonesia": "Indonesian", "malaysia": "Malaysian", "singapore": "Singaporean",
    "south korea": "South Korean", "taiwan": "Taiwanese", "colombia": "Colombian",
    "cuba": "Cuban", "venezuela": "Venezuelan", "uruguay": "Uruguayan",
    "kazakhstan": "Kazakhstani", "uzbekistan": "Uzbek", "kyrgyzstan": "Kyrgyz",
    "tajikistan": "Tajik", "turkmenistan": "Turkmen", "azerbaijan": "Azerbaijani",
    "armenia": "Armenian", "georgia": "Georgian", "mongolia": "Mongolian",
    "belarus": "Belarusian", "croatia": "Croatian", "serbia": "Serbian",
    "slovenia": "Slovenian", "bulgaria": "Bulgarian", "lithuania": "Lithuanian",
    "latvia": "Latvian", "estonia": "Estonian", "syria": "Syrian",
    "lebanon": "Lebanese", "jordan": "Jordanian", "tunisia": "Tunisian",
    "algeria": "Algerian", "libya": "Libyan", "sudan": "Sudanese",
    "ghana": "Ghanaian", "senegal": "Senegalese", "uganda": "Ugandan",
    "zimbabwe": "Zimbabwean", "cambodia": "Cambodian", "laos": "Lao",
    "myanmar": "Burmese", "sri lanka": "Sri Lankan", "afghanistan": "Afghan",
}


def country_to_nationality(country: str) -> str:
    """`Kuwait` -> `Kuwaiti`.

    Falls back to the two suffix rules that are actually reliable (`-stan` and
    `-ia`); for anything else the country name is returned unchanged rather than
    inventing a malformed adjective, since this string is shown to the neighbour
    generator and must not be wrong.
    """
    key = country.strip().lower()
    if key in _NATIONALITY:
        return _NATIONALITY[key]
    if not key:
        return ""
    country = country.strip()
    if key.endswith("stan"):
        return country + "i"
    if key.endswith("ia"):
        return country + "n"
    return country


def decade_of(year: Optional[int]) -> str:
    return f"{(year // 10) * 10}s" if year else ""


# --------------------------------------------------------------------------
# entity identification
# --------------------------------------------------------------------------
def candidate_names(text: str) -> List[str]:
    """All plausible person names in a string, possessives stripped."""
    out = []
    for match in NAME_RE.finditer(text):
        span = match.group(0).strip(" ,.")
        head = span.split()[0]
        if head in _NAME_STOPWORDS:
            # drop the leading stop-word and keep the rest if still a name
            rest = span.split(None, 1)
            if len(rest) < 2 or len(rest[1].split()) < 2:
                continue
            span = rest[1]
        span = re.sub(r"[\u2019']s$", "", span).strip()
        if span:
            out.append(span)
    return out


def _is_subsequence(short: Sequence[str], long: Sequence[str]) -> bool:
    return any(
        list(long[i : i + len(short)]) == list(short)
        for i in range(len(long) - len(short) + 1)
    )


def _canonicalize(counts: Counter) -> Dict[str, str]:
    """Map partial names onto the longest frequent name that contains them.

    "Basil Mahfouz" and "Basil Mahfouz Al-Kuwaiti" are the same author; the
    longer spelling wins.
    """
    names = sorted(counts, key=lambda n: (-len(n.split()), -counts[n]))
    mapping: Dict[str, str] = {}
    for name in names:
        tokens = name.split()
        for longer in names:
            if longer == name:
                continue
            longer_tokens = longer.split()
            # a rarer long variant ("Has Basil Mahfouz Al-Kuwaiti") must not
            # absorb the frequent short one it happens to contain
            if (
                len(longer_tokens) > len(tokens)
                and counts[longer] >= counts[name]
                and _is_subsequence(tokens, longer_tokens)
            ):
                mapping[name] = mapping.get(longer, longer)
                break
        mapping.setdefault(name, name)
    return mapping


def infer_entity_names(
    rows: Sequence[Dict[str, str]],
    question_key: str = "question",
    answer_key: str = "answer",
    block_size: Optional[int] = None,
    min_rows_per_entity: int = 2,
) -> List[str]:
    """Assign an entity name to every row of the forget split.

    A name that appears in at least `min_rows_per_entity` rows is treated as an
    entity, which drops one-off strings such as book titles or cities. Rows that
    mention no entity - TOFU's "What is the full name of the author born in
    ...?" - inherit the name of the surrounding rows, since the QA of one author
    is stored contiguously.

    Args:
        block_size: when known (20 for TOFU) rows are grouped into fixed blocks
            and the majority name of each block wins. Leave `None` to infer the
            grouping from the text alone.

    Returns:
        A list of entity names aligned with `rows`.
    """
    per_row_candidates: List[List[str]] = []
    for row in rows:
        names = candidate_names(str(row.get(answer_key, "")))
        names += candidate_names(str(row.get(question_key, "")))
        per_row_candidates.append(names)

    # row frequency (a name counts once per row) drives both filtering and ties
    row_counts: Counter = Counter()
    for names in per_row_candidates:
        row_counts.update(set(names))
    canonical = _canonicalize(row_counts)

    merged: Counter = Counter()
    for names in per_row_candidates:
        merged.update({canonical[n] for n in set(names)})
    frequent = {n for n, c in merged.items() if c >= min_rows_per_entity} or set(merged)

    best_per_row: List[Optional[str]] = []
    for names in per_row_candidates:
        options = {canonical[n] for n in names} & frequent
        best_per_row.append(
            max(options, key=lambda n: (merged[n], len(n))) if options else None
        )

    if block_size is not None:
        return _resolve_by_block(best_per_row, block_size)
    return _resolve_by_contiguity(best_per_row)


def _resolve_by_block(best_per_row: Sequence[Optional[str]], block_size: int) -> List[str]:
    resolved: List[str] = []
    for start in range(0, len(best_per_row), block_size):
        block = best_per_row[start : start + block_size]
        names = [n for n in block if n]
        if not names:
            raise ValueError(
                f"No entity name found in rows [{start}, {start + block_size}). "
                "Pass explicit entity names to the pipeline."
            )
        counts = Counter(names)
        top = max(counts.values())
        winner = max((n for n, c in counts.items() if c == top), key=len)
        resolved.extend([winner] * len(block))
    return resolved


def _resolve_by_contiguity(best_per_row: Sequence[Optional[str]]) -> List[str]:
    """Fill nameless rows from their neighbours (authors occupy contiguous rows)."""
    if not any(best_per_row):
        raise ValueError(
            "No entity name could be inferred from any row; pass explicit entity names."
        )
    resolved: List[Optional[str]] = list(best_per_row)
    last: Optional[str] = None
    for i, name in enumerate(resolved):
        if name is None:
            resolved[i] = last
        else:
            last = name
    nxt: Optional[str] = None
    for i in range(len(resolved) - 1, -1, -1):
        if resolved[i] is None:
            resolved[i] = nxt
        else:
            nxt = resolved[i]
    return [str(n) for n in resolved]


def group_rows_by_entity(entity_names: Sequence[str]) -> List[Tuple[str, List[int]]]:
    """`[(entity_name, [row indices]), ...]` in first-appearance order."""
    groups: Dict[str, List[int]] = {}
    for idx, name in enumerate(entity_names):
        groups.setdefault(name, []).append(idx)
    return list(groups.items())


# --------------------------------------------------------------------------
# attribute extraction
# --------------------------------------------------------------------------
class HeuristicAttributeExtractor:
    """Regex-based profile extraction from the QA text of one entity."""

    def __call__(self, name: str, texts: Sequence[str]) -> Attributes:
        blob = "\n".join(texts)
        birth_city, country = self._birthplace(blob)
        year = self._birth_year(blob)
        attrs = Attributes(
            name=name,
            nationality=country_to_nationality(country) if country else "",
            birth_decade=decade_of(year),
            birth_city=birth_city,
            birth_date=self._birth_date(blob),
            gender=self._gender(blob),
            profession=self._profession(blob),
            genre=self._genre(blob),
            award=self._award(blob, name),
            parent_professions=self._parents(blob),
            book_titles=self._titles(blob, name),
        )
        if country:
            attrs.extra["birth_country"] = country
        if year:
            attrs.extra["birth_year"] = year
        return attrs

    # -- individual fields -------------------------------------------------
    @staticmethod
    def _birthplace(blob: str) -> Tuple[str, str]:
        match = _BORN_IN_RE.search(blob)
        if not match:
            return "", ""
        place = re.sub(r"\s+", " ", match.group(1)).strip(" ,")
        parts = [p.strip() for p in place.split(",") if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[-1]
        return (parts[0], parts[0]) if parts else ("", "")

    @staticmethod
    def _birth_year(blob: str) -> Optional[int]:
        # the birth year is the earliest year mentioned near a "born" phrase
        window = blob
        match = re.search(r"\bborn\b", blob)
        if match:
            window = blob[match.start() : match.start() + 200]
        years = [int(y) for y in _YEAR_RE.findall(window)]
        if not years:
            years = [int(y) for y in _YEAR_RE.findall(blob)]
        return min(years) if years else None

    @staticmethod
    def _birth_date(blob: str) -> str:
        match = re.search(r"\bborn\b[^.]{0,120}", blob)
        window = match.group(0) if match else blob
        date = _DATE_RE.search(window)
        return date.group(0).strip() if date else ""

    @staticmethod
    def _gender(blob: str) -> str:
        lowered = blob.lower()
        male = len(re.findall(r"\b(he|his|him|male|father of)\b", lowered))
        female = len(re.findall(r"\b(she|her|hers|female)\b", lowered))
        if male == female:
            return ""
        return "male" if male > female else "female"

    @staticmethod
    def _profession(blob: str) -> str:
        match = _PROFESSION_RE.search(blob)
        return match.group(1).lower() if match else "author"

    @staticmethod
    def _genre(blob: str) -> str:
        for regex in (_GENRE_RE, _WRITES_RE):
            match = regex.search(blob)
            if match:
                genre = re.sub(r"\s+", " ", match.group(1)).strip().strip(".,")
                genre = re.sub(r"^(the|a|an)\s+", "", genre, flags=re.IGNORECASE)
                if 2 < len(genre) < 60:
                    return genre
        return ""

    @staticmethod
    def _parents(blob: str) -> List[str]:
        out = []
        for regex in (_FATHER_RE, _MOTHER_RE):
            match = regex.search(blob)
            if match:
                out.append(re.sub(r"\s+", " ", match.group(1)).strip())
        return out

    @staticmethod
    def _award(blob: str, name: str) -> str:
        for match in _AWARD_RE.finditer(blob):
            award = match.group(1).strip()
            if name and name.split()[0] in award:
                continue  # "the <Author> Award" is an identity-specific artefact
            return award
        return ""

    @staticmethod
    def _titles(blob: str, name: str) -> List[str]:
        titles = [t.strip() for t in _TITLE_RE.findall(blob)]
        titles = [t for t in titles if t and t.lower() != name.lower() and len(t) > 3]
        return dedup_preserve_order(titles)[:8]


ATTRIBUTE_SYSTEM_PROMPT = (
    "You extract structured biographical profiles from question-answer pairs "
    "about fictional authors. Answer with a single JSON object and nothing else."
)

ATTRIBUTE_USER_TEMPLATE = """\
Below are question-answer pairs about the author "{name}".

{qa_block}

Extract this JSON object (use "" or [] when the text does not say):
{{
  "nationality": "adjective, e.g. Kuwaiti",
  "birth_decade": "e.g. 1950s",
  "birth_city": "city of birth",
  "birth_date": "date of birth exactly as written in the text",
  "gender": "male|female|",
  "profession": "e.g. author",
  "genre": "main writing genre",
  "award": "a named award the author received",
  "parent_professions": ["father's job", "mother's job"],
  "book_titles": ["title", ...]
}}"""


class LLMAttributeExtractor:
    """Ask an external LLM for the profile, with heuristic back-fill."""

    def __init__(self, client: Any, max_pairs: int = 20, fallback: Optional[Any] = None):
        self.client = client
        self.max_pairs = max_pairs
        self.fallback = fallback or HeuristicAttributeExtractor()

    def __call__(self, name: str, texts: Sequence[str]) -> Attributes:
        base = self.fallback(name, texts)
        qa_block = "\n".join(f"- {t}" for t in list(texts)[: self.max_pairs])
        prompt = ATTRIBUTE_USER_TEMPLATE.format(name=name, qa_block=qa_block)
        try:
            raw = self.client.complete(prompt, system=ATTRIBUTE_SYSTEM_PROMPT)
            data = _parse_json_object(raw)
        except Exception as e:  # noqa: BLE001 - the heuristic profile is a valid fallback
            logger.warning("LLM attribute extraction failed for `%s`: %s", name, e)
            return base

        merged = base.to_dict()
        merged.pop("extra", None)
        for key, value in data.items():
            if key not in Attributes.__dataclass_fields__ or key == "extra":
                continue
            if isinstance(value, str) and value.strip():
                merged[key] = value.strip()
            elif isinstance(value, list) and value:
                merged[key] = [str(v).strip() for v in value if str(v).strip()]
        merged["name"] = name
        attrs = Attributes.from_dict(merged)
        attrs.extra.update(base.extra)
        return attrs


def _parse_json_object(raw: str) -> Dict[str, Any]:
    """Tolerant JSON extraction: strips code fences and surrounding prose."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM reply: {raw[:200]!r}")
    return json.loads(text[start : end + 1])


def parse_json_array(raw: str) -> List[Any]:
    """Tolerant JSON-array extraction (used by the neighbour generator)."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON array found in LLM reply: {raw[:200]!r}")
        data = json.loads(text[start : end + 1])
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return value
        return [data]
    return list(data)


def extract_attributes(
    rows: Sequence[Dict[str, str]],
    entity_names: Sequence[str],
    extractor: Optional[Any] = None,
    question_key: str = "question",
    answer_key: str = "answer",
) -> Dict[str, Attributes]:
    """Profile every entity from the rows that mention it."""
    extractor = extractor or HeuristicAttributeExtractor()
    texts: Dict[str, List[str]] = {}
    for row, name in zip(rows, entity_names):
        q, a = str(row.get(question_key, "")), str(row.get(answer_key, ""))
        texts.setdefault(name, []).append(f"Q: {q} A: {a}")
    return {name: extractor(name, qa) for name, qa in texts.items()}
