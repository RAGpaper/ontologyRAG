"""
Build health recommendation embeddings from the SustainHealthyFood Ontology v2.

Extracts all five food-health recommendation properties:
  - recommendedFor / RecommendedFor        – food is a good choice for a health state
  - highlyRecommendedFor                   – food is a top-priority choice
  - notRecommendedFor                      – food should be avoided
  - FoodInLimitedAmount                    – food allowed only in limited amounts
  - FoodToBeConsumedInHighAmount           – food especially encouraged

Creates three document kinds for rich semantic retrieval:
  - health_food      : one per food — natural-language sentences for each relation
  - health_state     : one per health state — combined summary of all levels
  - health_state_rel : one per (health state × relation level) — fine-grained retrieval

Run once before starting the app:
    python embeddingsStorage_ontology_health.py
"""
from __future__ import annotations

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS
from langchain_ollama import OllamaEmbeddings

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from typing import Dict, List, Optional, Set, Tuple

ONTOLOGY_FILE = "SHFO_extension_v2.ttl"
ONTOLOGY_NAMESPACE = "http://SustainHealthyFoodOntology#"
CHROMA_DIR = "ontology_chroma_db_health"
CHROMA_COLLECTION = "ontology_health_collection"
EMBED_MODEL = "nomic-embed-text"

# ---------------------------------------------------------------------------
# Property registry
# Each entry: prop_local_name -> (canonical_label, action_phrase, avoidance)
#   canonical_label  : used in reverse-index bucketing (two aliases share one key)
#   action_phrase    : extra synonyms embedded in text for semantic recall
#   avoidance        : True when the relation means "do NOT eat this food"
# ---------------------------------------------------------------------------
RECOMMENDATION_PROPS: Dict[str, Tuple[str, str, bool]] = {
    "recommendedFor": (
        "recommended for",
        "beneficial for, good choice if you have, suitable for",
        False,
    ),
    "RecommendedFor": (
        "recommended for",                          # same bucket as recommendedFor
        "beneficial for, good choice if you have, suitable for",
        False,
    ),
    "highlyRecommendedFor": (
        "highly recommended for",
        "highly beneficial for, top food choice for, best food if you have, "
        "strongly advised for, especially good for",
        False,
    ),
    "notRecommendedFor": (
        "not recommended for",
        "avoid if you have, should not eat for, bad for, harmful for, "
        "unsuitable for, contraindicated for, foods to avoid for",
        True,
    ),
    "FoodInLimitedAmount": (
        "should be consumed in limited amounts for",
        "limit intake for, eat sparingly if you have, moderation required for, "
        "restrict if you have, limited amount for",
        False,
    ),
    "FoodToBeConsumedInHighAmount": (
        "should be consumed in high amounts for",
        "increase intake for, eat more of this if you have, "
        "high consumption recommended for, prioritise for",
        False,
    ),
    "isIdealFor": (
        "ideal for",
        "good for, suitable for, best for, recommended for, perfect for, "
        "ideal meal for, good meal choice for, what to eat for, "
        "breakfast ideas, lunch ideas, dinner ideas, snack ideas",
        False,
    ),
}

# Bucket labels used in the combined health-state summary doc (ordered for display)
LEVEL_ORDER: List[str] = [
    "highly recommended for",
    "recommended for",
    "ideal for",
    "should be consumed in high amounts for",
    "should be consumed in limited amounts for",
    "not recommended for",
]

graph = Graph()
graph.parse(ONTOLOGY_FILE, format="turtle")
SHFO = Namespace(ONTOLOGY_NAMESPACE)
graph.bind("shfo", SHFO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def local_name(node) -> str:
    text = str(node)
    if "#" in text:
        return text.split("#")[-1]
    if "/" in text:
        return text.rsplit("/", 1)[-1]
    return text


def prettify(text: str) -> str:
    text = text.replace("_", " ").replace("-", " ")
    out: List[str] = []
    for i, ch in enumerate(text):
        if i > 0 and ch.isupper() and text[i - 1].islower():
            out.append(" ")
        out.append(ch)
    return "".join(out).strip()


def label(node) -> str:
    lbl = graph.value(node, RDFS.label)
    return str(lbl) if lbl else prettify(local_name(node))


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------

def build_food_health_document(food: URIRef, rec_data: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Per-food doc: one natural-language sentence per relationship type found.
    Multiple phrasings per relation improve semantic recall across query styles.
    """
    food_lbl = label(food)
    lines = [
        f"Food: {food_lbl}",
        f"{food_lbl} health recommendations:",
    ]
    for canonical_label, states in rec_data.items():
        _, action_phrase, avoidance = _canon_to_meta[canonical_label]
        states_str = ", ".join(sorted(states))
        if avoidance:
            lines.append(
                f"{food_lbl} is {canonical_label}: {states_str}. "
                f"Avoid {food_lbl} if you have: {states_str}. ({action_phrase})"
            )
        else:
            lines.append(
                f"{food_lbl} is {canonical_label}: {states_str}. "
                f"If you have {states_str}, {food_lbl} is a good dietary choice. ({action_phrase})"
            )
    return {
        "id": str(food),
        "kind": "health_food",
        "text": "\n".join(lines),
    }


def build_health_state_summary_document(
    state_lbl: str,
    buckets: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    Combined summary doc per health state — broad semantic match for general queries
    like "what should I eat for anemia?" or "diet for diabetes".
    """
    lines = [
        f"Health condition or dietary context: {state_lbl}",
        f"Food recommendations for {state_lbl}:",
    ]
    for level in LEVEL_ORDER:
        foods = buckets.get(level, [])
        if not foods:
            continue
        foods_str = ", ".join(sorted(foods))
        if level == "not recommended for":
            lines.append(
                f"Foods to avoid for {state_lbl} ({level}): {foods_str}. "
                f"Do not eat these if you have {state_lbl}."
            )
        elif level == "highly recommended for":
            lines.append(
                f"Highly recommended foods for {state_lbl} (top priority, eat more): {foods_str}. "
                f"Best food choices if you have {state_lbl}."
            )
        elif level == "recommended for":
            lines.append(
                f"Recommended foods for {state_lbl}: {foods_str}. "
                f"Beneficial foods if you have {state_lbl}."
            )
        elif level == "ideal for":
            lines.append(
                f"Foods ideal for {state_lbl}: {foods_str}. "
                f"Good food choices for {state_lbl}. Best foods to eat for {state_lbl}."
            )
        elif level == "should be consumed in limited amounts for":
            lines.append(
                f"Foods to eat in limited amounts for {state_lbl}: {foods_str}. "
                f"Consume these in moderation if you have {state_lbl}."
            )
        elif level == "should be consumed in high amounts for":
            lines.append(
                f"Foods to increase for {state_lbl}: {foods_str}. "
                f"Eat more of these if you have {state_lbl}."
            )
    return {
        "id": f"health_state__{state_lbl.replace(' ', '_')}",
        "kind": "health_state",
        "text": "\n".join(lines),
    }


def build_health_state_rel_document(
    state_lbl: str,
    level: str,
    foods: List[str],
) -> Dict[str, str]:
    """
    Fine-grained doc: one per (health state, relation level).
    Enables precise retrieval for targeted queries, e.g.
    "which foods should I AVOID for diabetes?" vs "which foods are RECOMMENDED for anemia?".
    """
    foods_str = ", ".join(sorted(foods))
    foods_short = ", ".join(sorted(foods)[:10])

    if level == "not recommended for":
        text = (
            f"Foods not recommended for {state_lbl} — foods to avoid:\n"
            f"If you have {state_lbl}, avoid these foods: {foods_str}.\n"
            f"Do not eat these if you have {state_lbl}: {foods_short}.\n"
            f"These foods are contraindicated or unsuitable for {state_lbl}."
        )
    elif level == "highly recommended for":
        text = (
            f"Foods highly recommended for {state_lbl} — top priority foods to eat:\n"
            f"If you have {state_lbl}, these foods are especially beneficial: {foods_str}.\n"
            f"Best food choices for {state_lbl}: {foods_short}.\n"
            f"Strongly advised foods if you have {state_lbl}."
        )
    elif level == "recommended for":
        text = (
            f"Foods recommended for {state_lbl}:\n"
            f"If you have {state_lbl}, these foods are good dietary choices: {foods_str}.\n"
            f"Beneficial foods for {state_lbl}: {foods_short}."
        )
    elif level == "ideal for":
        text = (
            f"Foods ideal for {state_lbl}:\n"
            f"Good food choices for {state_lbl}: {foods_str}.\n"
            f"What to eat for {state_lbl}: {foods_short}.\n"
            f"Best foods to eat for {state_lbl}. Suggested {state_lbl} foods."
        )
    elif level == "should be consumed in limited amounts for":
        text = (
            f"Foods to eat in limited amounts for {state_lbl} — consume in moderation:\n"
            f"If you have {state_lbl}, limit intake of these foods: {foods_str}.\n"
            f"Restrict but do not completely avoid these foods if you have {state_lbl}: {foods_short}."
        )
    elif level == "should be consumed in high amounts for":
        text = (
            f"Foods to increase consumption for {state_lbl}:\n"
            f"If you have {state_lbl}, eat more of these foods: {foods_str}.\n"
            f"Increase intake of these foods if you have {state_lbl}: {foods_short}."
        )
    else:
        text = (
            f"Food relation '{level}' for {state_lbl}: {foods_str}."
        )

    safe_level = level.replace(" ", "_").replace("/", "_")
    return {
        "id": f"health_state_rel__{state_lbl.replace(' ', '_')}__{safe_level}",
        "kind": "health_state_rel",
        "text": text,
    }


# Build a reverse lookup: canonical_label -> (prop_name, action_phrase, avoidance)
# Used in build_food_health_document.
_canon_to_meta: Dict[str, Tuple[str, str, bool]] = {}
for _prop, (_canon, _action, _avoid) in RECOMMENDATION_PROPS.items():
    if _canon not in _canon_to_meta:
        _canon_to_meta[_canon] = (_prop, _action, _avoid)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def build_health_embedding_documents() -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []

    # Reverse index: state_label -> {canonical_label -> [food_labels]}
    health_states: Dict[str, Dict[str, List[str]]] = {}

    # Pass 1 – iterate over Food individuals and collect ObjectProperty triples
    for food in sorted(graph.subjects(RDF.type, SHFO.Food), key=lambda x: label(x).lower()):
        if not isinstance(food, URIRef):
            continue
        food_lbl = label(food)
        # rec_data: canonical_label -> [state_labels]  (deduplicated)
        rec_data: Dict[str, List[str]] = {}

        for prop_name, (canonical_lbl, _, _) in RECOMMENDATION_PROPS.items():
            for _, _, target in graph.triples((food, SHFO[prop_name], None)):
                if not isinstance(target, URIRef):
                    continue
                state_lbl = label(target)
                # accumulate food -> states
                if canonical_lbl not in rec_data:
                    rec_data[canonical_lbl] = []
                if state_lbl not in rec_data[canonical_lbl]:
                    rec_data[canonical_lbl].append(state_lbl)
                # accumulate reverse index
                if state_lbl not in health_states:
                    health_states[state_lbl] = {lv: [] for lv in LEVEL_ORDER}
                bucket = health_states[state_lbl].setdefault(canonical_lbl, [])
                if food_lbl not in bucket:
                    bucket.append(food_lbl)

        if rec_data:
            docs.append(build_food_health_document(food, rec_data))

    # Pass 2 – handle FoodInLimitedAmount / FoodToBeConsumedInHighAmount used as
    # DatatypeProperty on HealthState individuals (value = food name string)
    _dt_props = {
        "FoodInLimitedAmount":          "should be consumed in limited amounts for",
        "FoodToBeConsumedInHighAmount":  "should be consumed in high amounts for",
    }
    for prop_name, canonical_lbl in _dt_props.items():
        for state, _, obj in graph.triples((None, SHFO[prop_name], None)):
            if not isinstance(state, URIRef) or not isinstance(obj, Literal):
                continue
            state_lbl = label(state)
            food_name = str(obj).strip()
            if not food_name:
                continue
            if state_lbl not in health_states:
                health_states[state_lbl] = {lv: [] for lv in LEVEL_ORDER}
            bucket = health_states[state_lbl].setdefault(canonical_lbl, [])
            if food_name not in bucket:
                bucket.append(food_name)

    # Build health-state summary + per-level granular docs
    for state_lbl, buckets in sorted(health_states.items()):
        if not any(buckets.values()):
            continue
        docs.append(build_health_state_summary_document(state_lbl, buckets))
        for level, foods in buckets.items():
            if foods:
                docs.append(build_health_state_rel_document(state_lbl, level, foods))

    return docs


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def upsert_documents(documents: List[Dict[str, str]]) -> None:
    texts = [doc["text"] for doc in documents]
    metadatas = [{"id": doc["id"], "kind": doc["kind"]} for doc in documents]
    ids = [doc["id"] for doc in documents]

    embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embedding_model,
    )
    db.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def main() -> None:
    docs = build_health_embedding_documents()
    food_docs = sum(1 for d in docs if d["kind"] == "health_food")
    state_docs = sum(1 for d in docs if d["kind"] == "health_state")
    rel_docs = sum(1 for d in docs if d["kind"] == "health_state_rel")
    print(f"Built {len(docs)} health recommendation documents.")
    print(f"  Per-food documents:                  {food_docs}")
    print(f"  Per-health-state summary documents:  {state_docs}")
    print(f"  Per-health-state-relation documents: {rel_docs}")
    upsert_documents(docs)
    print(f"Saved embeddings to '{CHROMA_DIR}' / collection '{CHROMA_COLLECTION}'")


if __name__ == "__main__":
    main()
