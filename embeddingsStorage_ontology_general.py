from __future__ import annotations

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL
from langchain_ollama import OllamaEmbeddings

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from typing import Dict, Iterable, List, Optional, Set, Tuple

ONTOLOGY_FILE = "SHFO_extension_v2.ttl"
ONTOLOGY_NAMESPACE = "http://SustainHealthyFoodOntology#"
CHROMA_DIR = "ontology_chroma_db_general"
CHROMA_COLLECTION = "ontology_general_collection"
EMBED_MODEL = "nomic-embed-text"


graph = Graph()
graph.parse(ONTOLOGY_FILE, format="turtle")
SHFO = Namespace(ONTOLOGY_NAMESPACE)
graph.bind("shfo", SHFO)


FOOD_ROOT = SHFO.Food
MOLECULE_ROOT = SHFO.Molecules
QUANTITY_ROOT = SHFO.Quantity

IGNORE_PREDICATES: Set[URIRef] = {
    RDF.type,
    RDFS.label,
    RDFS.comment,
    RDFS.subClassOf,
    OWL.sameAs,
}


def local_name(node) -> str:
    text = str(node)
    if "#" in text:
        return text.split("#")[-1]
    if "/" in text:
        return text.rsplit("/", 1)[-1]
    return text


def prettify(text: str) -> str:
    text = text.replace("_", " ").replace("-", " ")
    out = []
    for i, ch in enumerate(text):
        if i > 0 and ch.isupper() and text[i - 1].islower():
            out.append(" ")
        out.append(ch)
    return "".join(out).strip()


def label(node) -> str:
    lbl = graph.value(node, RDFS.label)
    return str(lbl) if lbl else prettify(local_name(node))


def comment(node) -> Optional[str]:
    c = graph.value(node, RDFS.comment)
    return str(c) if c else None


def transitive_subclasses(root: URIRef) -> Set[URIRef]:
    result: Set[URIRef] = set()
    frontier = [root]
    while frontier:
        cur = frontier.pop()
        for child in graph.subjects(RDFS.subClassOf, cur):
            if isinstance(child, URIRef) and child not in result:
                result.add(child)
                frontier.append(child)
    return result


def class_hierarchy_path(cls: URIRef) -> str:
    """Walk rdfs:subClassOf upward to build the full path, e.g. Molecules → Micromolecules → AminoAcids → EssentialAminoAcids."""
    path = []
    current: Optional[URIRef] = cls
    visited: Set[URIRef] = set()
    while current and current not in visited:
        visited.add(current)
        path.append(label(current))
        parents = [p for p in graph.objects(current, RDFS.subClassOf) if isinstance(p, URIRef)]
        current = parents[0] if parents else None
    path.reverse()
    return " → ".join(path)


def entity_hierarchy_path(entity: URIRef, target_classes: Set[URIRef]) -> str:
    """Return the full hierarchy path ending at the entity itself."""
    ent_types = [cls for cls in graph.objects(entity, RDF.type)
                 if isinstance(cls, URIRef) and cls in target_classes]
    if not ent_types:
        return ""
    # Pick the most specific type (the one with the deepest hierarchy path).
    best = max(ent_types, key=lambda cls: len(class_hierarchy_path(cls).split(" → ")))
    return f"{class_hierarchy_path(best)} → {label(entity)}"


def direct_parents(cls: URIRef) -> List[str]:
    return sorted({label(parent) for parent in graph.objects(cls, RDFS.subClassOf) if isinstance(parent, URIRef)})


def direct_children(cls: URIRef) -> List[str]:
    return sorted({label(child) for child in graph.subjects(RDFS.subClassOf, cls) if isinstance(child, URIRef)})


def class_individuals(cls: URIRef) -> List[URIRef]:
    return sorted({s for s in graph.subjects(RDF.type, cls) if isinstance(s, URIRef)}, key=lambda x: label(x).lower())


def datatype_properties_for_entity(entity: URIRef) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for pred, obj in graph.predicate_objects(entity):
        if pred in IGNORE_PREDICATES:
            continue
        if isinstance(obj, Literal):
            rows.append((label(pred), str(obj)))
    rows.sort(key=lambda x: (x[0].lower(), x[1].lower()))
    return rows


def object_properties_for_entity(entity: URIRef) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for pred, obj in graph.predicate_objects(entity):
        if pred in IGNORE_PREDICATES:
            continue
        if isinstance(obj, URIRef):
            rows.append((label(pred), label(obj)))
    rows.sort(key=lambda x: (x[0].lower(), x[1].lower()))
    return rows


def types_for_entity(entity: URIRef, target_classes: Set[URIRef]) -> List[str]:
    rows = []
    for cls in graph.objects(entity, RDF.type):
        if isinstance(cls, URIRef) and cls in target_classes:
            rows.append(label(cls))
    return sorted(set(rows))


def build_class_document(cls: URIRef) -> Dict[str, str]:
    cls_label = label(cls)
    lines = [
        f"Entity: {cls_label}",
        "Kind: ontology class",
        f"IRI: {cls}",
    ]
    c = comment(cls)
    if c:
        lines.append(f"Description: {c}")

    parents = direct_parents(cls)
    if parents:
        lines.append(f"Direct parents: {', '.join(parents)}")

    children = direct_children(cls)
    if children:
        lines.append(f"Direct subclasses: {', '.join(children[:20])}")

    inds = class_individuals(cls)
    if inds:
        lines.append(f"Example individuals: {', '.join(label(x) for x in inds[:20])}")
        lines.append(f"Individual count: {len(inds)}")

    return {
        "id": str(cls),
        "kind": "class",
        "text": "\n".join(lines),
    }


_FOOD_SOURCE_PRED = "food source"


def _is_food_source_rel(pred: str) -> bool:
    return _FOOD_SOURCE_PRED in pred.lower()


def build_individual_document(entity: URIRef, target_classes: Set[URIRef]) -> Dict[str, str]:
    ent_label = label(entity)
    hierarchy = entity_hierarchy_path(entity, target_classes)

    # Name repeated at top so exact-name queries ("what is tryptophan?") score highest.
    lines = [
        f"Name: {ent_label}. {ent_label} is an ontology individual named {ent_label}.",
        f"Entity: {ent_label}",
        "Kind: ontology individual",
        f"IRI: {entity}",
    ]

    if hierarchy:
        lines.append(f"Category: {hierarchy}")

    ent_types = types_for_entity(entity, target_classes)
    if ent_types:
        lines.append(f"Ontology classes: {', '.join(ent_types)}")

    c = comment(entity)
    if c:
        lines.append(f"Description: {c}")

    obj_rows = object_properties_for_entity(entity)
    other_rels = [(p, o) for p, o in obj_rows if not _is_food_source_rel(p)]
    food_source_rels = [(p, o) for p, o in obj_rows if _is_food_source_rel(p)]

    if other_rels:
        rels = [f"{pred} -> {obj}" for pred, obj in other_rels[:15]]
        lines.append(f"Relationships: {'; '.join(rels)}")

    # Keep only a short preview of food sources; the full list goes in a separate document.
    if food_source_rels:
        preview = [obj for _, obj in food_source_rels[:5]]
        lines.append(f"Example food sources ({len(food_source_rels)} total): {', '.join(preview)}")

    data_rows = datatype_properties_for_entity(entity)
    if data_rows:
        props = [f"{pred} = {val}" for pred, val in data_rows[:40]]
        lines.append(f"Data properties: {'; '.join(props)}")

    return {
        "id": str(entity),
        "kind": "individual",
        "text": "\n".join(lines),
    }


def build_individual_sources_document(entity: URIRef) -> Optional[Dict[str, str]]:
    """Separate document listing all food sources so they don't clutter the main description."""
    obj_rows = object_properties_for_entity(entity)
    food_source_rels = [(p, o) for p, o in obj_rows if _is_food_source_rel(p)]
    if not food_source_rels:
        return None

    ent_label = label(entity)
    lines = [
        f"Food sources for {ent_label}:",
        f"Entity: {ent_label}",
        "Kind: food sources list",
        f"IRI: {entity}",
    ]
    for _, food_name in food_source_rels:
        lines.append(f"- {food_name}")

    return {
        "id": f"{entity}__sources",
        "kind": "individual_sources",
        "text": "\n".join(lines),
    }


def collect_target_classes() -> Set[URIRef]:
    return (
        {FOOD_ROOT, MOLECULE_ROOT, QUANTITY_ROOT}
        | transitive_subclasses(FOOD_ROOT)
        | transitive_subclasses(MOLECULE_ROOT)
        | transitive_subclasses(QUANTITY_ROOT)
    )


def collect_target_individuals(target_classes: Set[URIRef]) -> Set[URIRef]:
    inds: Set[URIRef] = set()
    for cls in target_classes:
        for ent in graph.subjects(RDF.type, cls):
            if isinstance(ent, URIRef):
                inds.add(ent)
    return inds


def build_general_embedding_documents() -> List[Dict[str, str]]:
    target_classes = collect_target_classes()
    docs: List[Dict[str, str]] = []

    for cls in sorted(target_classes, key=lambda x: label(x).lower()):
        docs.append(build_class_document(cls))

    target_individuals = collect_target_individuals(target_classes)
    for ent in sorted(target_individuals, key=lambda x: label(x).lower()):
        docs.append(build_individual_document(ent, target_classes))
        sources_doc = build_individual_sources_document(ent)
        if sources_doc:
            docs.append(sources_doc)

    return docs


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
    docs = build_general_embedding_documents()
    class_docs = sum(1 for d in docs if d["kind"] == "class")
    individual_docs = sum(1 for d in docs if d["kind"] == "individual")
    sources_docs = sum(1 for d in docs if d["kind"] == "individual_sources")
    print(f"Built {len(docs)} general ontology documents.")
    print(f" - Class documents:          {class_docs}")
    print(f" - Individual documents:     {individual_docs}")
    print(f" - Food-sources documents:   {sources_docs}")
    upsert_documents(docs)
    print(f"Saved embeddings to {CHROMA_DIR} / {CHROMA_COLLECTION}")


if __name__ == "__main__":
    main()
