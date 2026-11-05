import streamlit as st
import re
import difflib
import json
import os
import csv
import io
import statistics
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Set, Tuple

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL
try:
    from langchain_ollama import OllamaLLM as Ollama, OllamaEmbeddings
except ImportError:
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma


PREDEFINED_QUESTIONS = [
    # 1. General nutrition based on profile
    "What are my daily calorie and macronutrient targets based on my profile?",
    "What food is recommended for my life stage and diet preference?",

    # 2. Recipe & meal planning ideas
    "Can you help me meal prep for the week with 3 balanced recipes?",
    "What are some healthy breakfast ideas that match my diet preference?",

    # 3. Health condition recommendations
    "What foods are recommended for someone with iron deficiency(anemia)?",
    "Which foods should I limit and which should I eat more of if I have type 2 diabetes ?",
    

    # 4. Mental health & wellbeing
    "What food or nutrition help to manage stress",
    "Which nutrients support better sleep quality?",
    "Which nutrients and foods help to reduce anxiety?",

    # 5. Pre & post workout nutrition
    "What should I eat before and after training?",
    "Which foods help with muscle recovery after intense exercise?",

    # 6. Allergies & food substitutions
    "Are there foods I should avoid given my allergies and diet preference?",
    "What should I avoid if I am lactose intolerant?",

    # 7. Nutrient & ingredient deep-dive
    "What is omega-3, why does it matter, and which foods are richest in it?",
    "What are the best plant-based protein sources for someone following a vegan diet?",
    "Which foods are high in protein but low in fat?",

    # 8. Environmental impact / sustainability
    "What is the environmental sustainability score of banana?",
]


ONTOLOGY_FILE = "SHFO_extension_v2.ttl"
ONTOLOGY_NAMESPACE = "http://SustainHealthyFoodOntology#"

GENERAL_CHROMA_DIR = "ontology_chroma_db_general"
GENERAL_CHROMA_COLLECTION = "ontology_general_collection"

HEALTH_CHROMA_DIR = "ontology_chroma_db_health"
HEALTH_CHROMA_COLLECTION = "ontology_health_collection"

MENTAL_STATE_CHROMA_DIR = "mental_state_chroma_db"
MENTAL_STATE_COLLECTION = "mental_state_model"

NUTRIENT_PROPERTIES: Dict[str, str] = {
    "hasAmountOfCalories": "Calories",
    "hasAmountOfProtein": "Protein",
    "hasAmountOfProteinPlant": "Protein (Plant)",
    "hasAmountOfProteinAnimal": "Protein (Animal)",
    "hasAmountOfCarbohydrates": "Carbohydrates",
    "hasAmountOfStarch": "Starch",
    "hasAmountOfSugar": "Sugar",
    "hasAmountOfPolyols": "Polyols",
    "hasAmountOfFat": "Fat",
    "hasAmountOfFattyAcids": "Fatty Acids",
    "hasAmountOfSaturatedFattyAcids": "Saturated Fatty Acids",
    "hasAmountOfFiber": "Fiber",
    "hasAmountOfDietaryFiber": "Dietary Fiber",
    "hasAmountOfWater": "Water",
    "hasAmountOfTryptophan": "Tryptophan",
    "hasAmountOfCalcium": "Calcium",
    "hasAmountOfIron": "Iron",
    "hasAmountOfMagnesium": "Magnesium",
    "hasAmountOfZinc": "Zinc",
    "hasAmountOfPotassium": "Potassium",
    # Kalium is NEVO's (Dutch) name for potassium — same values, but on 1445
    # foods vs 796 for hasAmountOfPotassium, so it fills the gaps. Listed
    # after Potassium: get_food_nutrient_number / get_all_nutrients_for_food
    # take the first property that has a value.
    "hasAmountOfKalium": "Potassium",
    "hasAmountOfSodium": "Sodium",
    "hasAmountOfVitaminA": "Vitamin A",
    "hasAmountOfVitaminB6": "Vitamin B6",
    "hasAmountOfVitaminC": "Vitamin C",
    "hasAmountOfVitaminD": "Vitamin D",
    "hasAmountOfVitaminE": "Vitamin E",
    "hasAmountOfVitaminK": "Vitamin K",
    "hasNutrientScore": "Nutrient Score",
}

NUTRIENT_UNITS: Dict[str, str] = {
    "Calories": "kcal",
    "Protein": "g",
    "Protein (Plant)": "g",
    "Protein (Animal)": "g",
    "Carbohydrates": "g",
    "Starch": "g",
    "Sugar": "g",
    "Polyols": "g",
    "Fat": "g",
    "Fatty Acids": "g",
    "Saturated Fatty Acids": "g",
    "Fiber": "g",
    "Dietary Fiber": "g",
    "Water": "g",
    "Tryptophan": "mg",
    "Calcium": "mg",
    "Iron": "mg",
    "Magnesium": "mg",
    "Zinc": "mg",
    "Potassium": "mg",
    "Sodium": "mg",
    "Vitamin A": "µg",
    "Vitamin B6": "mg",
    "Vitamin C": "mg",
    "Vitamin D": "µg",
    "Vitamin E": "mg",
    "Vitamin K": "µg",
    "Nutrient Score": "",
}

# Only ~15% of Food individuals have these — sourced from a df_sus/OWID environmental
# footprint dataset matched onto NEVO food labels, so most foods legitimately have none.
ENVIRONMENTAL_PROPERTIES: Dict[str, str] = {
    "AmountOfCO2emissions": "GHG (CO2) Emissions",
    "AmountOfEutrophication": "Eutrophying Emissions",
    "AmountOfLandUse": "Land Use",
}

llm = Ollama(model="llama3.2")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

def _call_llm(prompt: str, temperature: float = 0.0, num_predict: int = 500) -> str:
    """Invoke the LLM with per-call temperature/num_predict overrides.
    langchain_ollama.OllamaLLM doesn't accept these as call-time kwargs,
    so we create a lightweight copy with the desired settings."""
    try:
        configured = llm.model_copy(update={"temperature": temperature, "num_predict": num_predict})
    except AttributeError:
        configured = llm.copy(update={"temperature": temperature, "num_predict": num_predict})
    return configured.invoke(prompt)

ROUTER_PROMPT = """You are a question classifier for a dietary ontology Q&A system.

Classify the user's question into EXACTLY ONE category:
- nutrient_lookup: asks for nutrient/calorie values of a specific named food or ingredient
- environmental_impact: asks about the environmental impact, sustainability, carbon footprint, greenhouse gas (CO2) emissions, eutrophying emissions, or land use of a specific named food or ingredient
- recipe_lookup: asks for recipe ingredients, amounts, or how to prepare a specific named dish
(the ontology provides the ingredient list; cooking instructions are reasoned by the LLM from those ingredients)
- health_recommendation: asks which foods or nutrients are recommended for a specific health condition, deficiency, dietary context, or wellbeing state — including sleep quality, mood, low mood, anxiety, appetite, stress, depression, motivation, or energy levels (e.g. "what foods are good for anemia", "foods to avoid if vegetarian", "recommended foods for iron deficiency", "what should someone with vitamin B12 deficiency eat", "which nutrients support better sleep quality", "what nutrients or foods help reduce anxiety", "what foods should I consume if I am stressed", "which foods help with depression", "what to eat for a better mood", "foods that improve appetite", "what to eat to boost motivation", "foods that help with low energy")
- definition_lookup: asks what something is — including food concepts,
food classes, ontology terms, molecules (e.g. omega-3, antioxidants), hormones (e.g. cortisol, insulin),
neurotransmitters (e.g. serotonin, dopamine), vitamins, minerals, or any biological or nutritional concept
- profile_reasoning: asks for personalized dietary advice based on the user's own profile, life stage,
 goals, diet type, allergies, stress, activity, or current context
- food_suitability: names ONE specific food, dish, ingredient or recipe and asks whether the user personally
 can eat it, should eat it, or whether it is recommended / safe / OK / allowed for them (e.g. "can I eat a
 chocolate cookie?", "should I eat salmon?", "is brie OK for me?", "I want to eat X — is it recommended?",
 "am I allowed peanuts?")
- meal_suggestion: asks which foods or recipes are ideal for a specific meal type — breakfast, lunch, dinner, or snack (e.g. "what should I eat for breakfast?", "give me dinner food ideas", "what are good lunch options?", "what can I have as a snack?", "ideal foods for breakfast")
- general_cooking: open-ended cooking or general nutrition questions not covered by the ontology — e.g. "what should I cook", "which foods help with muscle recovery", "best foods for energy", "what to eat before training"

Rules (apply in order):
1. If the question names a specific food, dish, ingredient or recipe AND asks whether the user personally
can eat it, should eat it, or whether it is recommended / safe / OK / allowed for them (e.g. "can I eat a
chocolate cookie?", "should I eat salmon?", "is brie OK for me?", "I want to eat X — is it recommended?")
→ food_suitability. ENTITY = that food/dish/recipe name. This takes priority over EVERY rule below,
including the personal-profile rule.
2. If the question contains any of: "my profile", "my life stage", "my goal", "my diet type", "given my",
"based on my", "for me personally", "should I", "can you suggest", "daily meal", "adjust my diet", "my current",
"my activity", "my stress", "my allergies", "my weight" → use profile_reasoning
3. If the question asks about environmental impact, sustainability, carbon/greenhouse gas (CO2) emissions,
eutrophying emissions, or land use of a specific food or ingredient → environmental_impact. This takes priority
over nutrient_lookup even if the question also uses phrases like "how much" or "amount of".
4. If a specific food or ingredient name is asked about with nutrients/calories → nutrient_lookup
5. If the question is about sleep quality, mood, low mood, anxiety, appetite, stress, depression, motivation, or energy levels as a general state — e.g. "which nutrients support better sleep quality", "what nutrients or foods help reduce anxiety", "what foods should I consume if I am stressed", "which foods help with depression", "what to eat for a better mood", "foods that improve appetite", "what to eat to boost motivation", "foods that help with low energy" → health_recommendation. ENTITY must use the exact ontology name: "Anxious" for anxiety, "Stressed" for stress, "Depressed" for depression, "LowMood" for low mood or low energy, "MoodSwings" for mood swings, "EmotionalRecovery" for emotional recovery. For sleep or appetite extract "Sleep" or "Appetite".
6. If asking for multiple recipes, meal prep, weekly meals, or open-ended cooking help → general_cooking
7. If a single specific named recipe is asked about → recipe_lookup
8. If asking about foods recommended/not recommended for a health condition or dietary type → health_recommendation
9. If asking which foods or recipes are ideal/good/suitable for a meal type (breakfast, lunch, dinner, snack) → meal_suggestion. ENTITY should be the meal type name (e.g. "Breakfast", "Lunch", "Dinner", "Snack")
10. If asking what something is (a concept, class, molecule, hormone, neurotransmitter, vitamin, mineral) → definition_lookup

If the question contains obvious spelling mistakes, correct them when extracting the ENTITY name.
For example: "goualsh" → "Goulash", "anemeia" → "Anemia", "avokado" → "Avocado".

Respond ONLY in this exact format, no other text:
CATEGORY: <category>
ENTITY: <exact food/recipe name if applicable, otherwise NONE>

Question: {question}"""


@dataclass
class ResponseTrace:
    question: str
    timestamp: str
    routing_path: str
    retrieval_method: str
    matched_entity: str
    matched_entity_uri: str
    context_snippet: str
    context_length: int
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.0
    llm_num_predict: int = 300


def render_response_trace(trace: ResponseTrace) -> None:
    with st.expander("🔍 How was this response generated?"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Routing path:** `{trace.routing_path}`")
            st.markdown(f"**Retrieval method:** `{trace.retrieval_method}`")
            st.markdown(f"**Context length:** {trace.context_length} chars")
        with col2:
            st.markdown(f"**Model:** `{trace.llm_model}`")
            st.markdown(f"**Temperature:** `{trace.llm_temperature}` · **Max tokens:** `{trace.llm_num_predict}`")
            st.markdown(f"**Time:** {trace.timestamp}")
        if trace.matched_entity:
            st.markdown(f"**Matched entity:** {trace.matched_entity}")
        if trace.matched_entity_uri:
            st.markdown(f"**Entity URI:** `{trace.matched_entity_uri}`")
        with st.expander("Context passed to LLM"):
            st.text(trace.context_snippet)


@st.cache_resource
def load_ontology() -> Tuple[Graph, Namespace]:
    graph = Graph()
    graph.parse(ONTOLOGY_FILE, format="turtle")
    shfo = Namespace(ONTOLOGY_NAMESPACE)
    graph.bind("shfo", shfo)
    return graph, shfo


ontology_graph, SHFO = load_ontology()


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def fuzzy_label_score(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def classify_question(question: str) -> tuple:
    """Ask the LLM to classify the question and extract the relevant entity name.
    Returns (category, entity_name) where category is one of:
    nutrient_lookup, environmental_impact, recipe_lookup, definition_lookup, profile_reasoning, unknown.
    """
    try:
        response = _call_llm(ROUTER_PROMPT.format(question=question), temperature=0.0, num_predict=60)
        category = "unknown"
        entity = ""
        for line in response.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("CATEGORY:"):
                category = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("ENTITY:"):
                entity = line.split(":", 1)[1].strip()
                if entity.upper() in ("NONE", "N/A", ""):
                    entity = ""
        valid_categories = {"nutrient_lookup", "environmental_impact", "recipe_lookup", "definition_lookup", "profile_reasoning", "health_recommendation", "general_cooking", "meal_suggestion", "food_suitability"}
        if category not in valid_categories:
            category = "unknown"
        return category, entity
    except Exception:
        return "unknown", ""


def is_general_cooking_question(question: str) -> bool:
    q = question.lower()
    # These wellbeing/mental health states are handled by health_recommendation, not general_cooking
    wellbeing_terms = [
        "sleep", "anxiety", "mood", "low mood", "appetite",
        "stressed", "stress", "depression", "depressed",
        "motivation", "energy level", "amount of energy", "low energy",
    ]
    if any(term in q for term in wellbeing_terms):
        return False
    triggers = [
        # Meal planning & prep (generic, no specific recipe count)
        "meal plan", "weekly meal", "week of meals",
        "what should i cook", "what can i cook", "what to cook",
        "easy recipes", "quick recipes", "healthy recipes",
        "suggest meals", "suggest recipes", "give me recipes",
        "cook for the week",
        "recipe with chicken and", "recipe with fish and", "recipe with pasta and",
        # "Foods that help / support / boost X" patterns
        "which foods help", "what foods help", "foods that help",
        "which foods support", "what foods support", "foods that support",
        "which foods boost", "what foods boost", "foods that boost",
        "which foods are good for", "what foods are good for",
        "best foods for", "foods for recovery", "foods for muscle",
        "foods for energy", "foods for sleep", "foods for focus",
        "foods for performance", "foods for endurance", "foods for strength",
        "what should i eat after", "what should i eat before",
        "what to eat after", "what to eat before",
        "nutrition for", "diet for recovery", "diet for muscle",
        "eat for recovery", "eat for performance",
    ]
    # Catch "N different recipes" only when NOT combined with meal prep/balanced (those go to ontology)
    if re.search(r"\d+\s+(different\s+)?recipes?", q) and not any(
        t in q for t in ["meal prep", "balanced", "prep for the week", "week"]
    ):
        return True
    # Catch "foods to eat for X" or "foods to avoid for X"
    if re.search(r"foods?\s+to\s+(eat|avoid|include|prioritise|prioritize)\s+", q):
        return True
    return any(t in q for t in triggers)


def is_profile_reasoning_question(question: str) -> bool:
    """Keyword fallback for profile-based reasoning questions."""
    q = question.lower()
    triggers = [
        "my profile", "my life stage", "my diet type", "my goal",
        "my activity level", "my stress", "my allergies", "given my",
        "based on my", "for someone with my", "my current",
        "adjust my diet", "meal pattern", "meal plan", "fit my goal",
        "should i pay attention", "which nutrients should",
        "dietary guidelines for me", "general dietary guidelines",
    ]
    return any(t in q for t in triggers)


def prettify_fragment(fragment: str) -> str:
    fragment = fragment.replace("_", " ").replace("-", " ")
    fragment = re.sub(r"(?<!^)(?=[A-Z])", " ", fragment)
    return re.sub(r"\s+", " ", fragment).strip()


def get_entity_label(entity) -> str:
    label = ontology_graph.value(entity, RDFS.label)
    if label:
        return str(label)
    entity_str = str(entity)
    if "#" in entity_str:
        return prettify_fragment(entity_str.split("#")[-1])
    return prettify_fragment(entity_str.rsplit("/", 1)[-1])


def parse_numeric_literal(value) -> Optional[float]:
    try:
        return float(value.toPython())
    except Exception:
        try:
            return float(str(value))
        except Exception:
            return None


def format_nutrient_value(nutrient_name: str, value) -> str:
    number = parse_numeric_literal(value)
    unit = NUTRIENT_UNITS.get(nutrient_name, "")
    if number is None:
        return f"{value}{(' ' + unit) if unit else ''}".strip()

    if abs(number - round(number)) < 1e-9:
        number_text = str(int(round(number)))
    else:
        number_text = f"{number:.2f}".rstrip("0").rstrip(".")

    return f"{number_text}{(' ' + unit) if unit else ''}".strip()


def list_food_candidates() -> List[Tuple[str, object]]:
    candidates = []
    seen = set()
    for food in ontology_graph.subjects(RDF.type, SHFO.Food):
        if food in seen:
            continue
        seen.add(food)
        candidates.append((get_entity_label(food), food))
    return candidates


FOOD_CANDIDATES = list_food_candidates()


def build_general_entity_index() -> Dict[str, URIRef]:
    """Build normalized-label → URI index for all named individuals and classes."""
    index: Dict[str, URIRef] = {}
    for s, _, o in ontology_graph.triples((None, RDFS.label, None)):
        if isinstance(s, URIRef):
            norm = normalize_text(str(o))
            if norm:
                index[norm] = s
    for s in ontology_graph.subjects(RDF.type, None):
        if isinstance(s, URIRef):
            norm = normalize_text(str(s).split("#")[-1])
            if norm and norm not in index:
                index[norm] = s
    return index

GENERAL_ENTITY_INDEX: Dict[str, URIRef] = build_general_entity_index()


def extract_query_subject(question: str) -> Optional[str]:
    """Extract the subject name from 'what is X?', 'tell me about X', etc."""
    q = question.lower().strip()
    patterns = [
        r"what (?:is|are) (?:a |an |the )?(.+?)[\?\.]*$",
        r"tell me about (.+?)[\?\.]*$",
        r"explain (.+?)[\?\.]*$",
        r"describe (.+?)[\?\.]*$",
    ]
    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            return m.group(1).strip(" ?.,!")
    return None


# Curated food-source fallback for nutrient/molecule entities that carry NO
# shf:hasFoodSource triples in the ontology (e.g. shf:Omega3). Without this the
# "what is X / which foods are richest in it" answer has no grounded food to
# give and the LLM invents one from training data — which then bypasses the
# diet/allergy filter (e.g. only "fatty fish" for a vegan). Keyed by the
# entity's URI local name, lower-cased. Values are ontology food labels so the
# profile exclusion filter below can still drop conflicting ones.
_ENTITY_FOOD_SOURCE_FALLBACK: Dict[str, List[str]] = {
    "omega3": [
        "Salmon farmed raw", "Salmon smoked", "Linseeds", "Chia seeds dried",
        "Walnuts unsalted", "Hemp seed", "Oil rapeseed", "Pumpkin seeds",
    ],
    "omega-3": [
        "Salmon farmed raw", "Salmon smoked", "Linseeds", "Chia seeds dried",
        "Walnuts unsalted", "Hemp seed", "Oil rapeseed", "Pumpkin seeds",
    ],
}


def build_general_entity_context(entity_uri: URIRef, user_profile: Optional[dict] = None) -> str:
    """Build context for a general entity directly from the ontology graph.

    user_profile, if given, filters the food_sources list below against the
    user's diet type/allergies (same exclusion set meal_suggestion uses) —
    without this, a "what is omega-3 / which foods are richest in it"
    question surfaces the exact same raw food list regardless of who's
    asking, including foods that conflict with their profile.
    """
    entity_label = get_entity_label(entity_uri)
    lines = [f"Entity: {entity_label}", f"IRI: {entity_uri}"]

    desc = ontology_graph.value(entity_uri, RDFS.comment)
    if desc:
        lines.append(f"Description: {desc}")

    types = sorted({
        get_entity_label(o)
        for _, _, o in ontology_graph.triples((entity_uri, RDF.type, None))
        if isinstance(o, URIRef) and o != OWL.NamedIndividual and o != entity_uri
    })
    if types:
        lines.append(f"Ontology classes: {', '.join(types)}")

    data_props = sorted(
        f"{get_entity_label(p)} = {o}"
        for p, o in ontology_graph.predicate_objects(entity_uri)
        if isinstance(o, Literal) and p not in {RDFS.label, RDFS.comment}
    )
    if data_props:
        lines.append(f"Data properties: {'; '.join(data_props[:10])}")

    # hasFoodSource is excluded from the generic relationships dump below (it can
    # list hundreds of foods for a molecule like Gluten) but summarized here instead
    # — without this, molecules/nutrients with no rdfs:comment (Gluten included)
    # had nothing food-related in their context at all, and the LLM would fill the
    # gap from its own training data despite being told not to.
    food_sources = sorted({
        get_entity_label(o)
        for _, _, o in ontology_graph.triples((entity_uri, SHFO.hasFoodSource, None))
        if isinstance(o, URIRef)
    })
    fallback_used = False
    if not food_sources:
        key = str(entity_uri).split("#")[-1].lower()
        fb = _ENTITY_FOOD_SOURCE_FALLBACK.get(key) or _ENTITY_FOOD_SOURCE_FALLBACK.get(
            normalize_text(entity_label)
        )
        if fb:
            food_sources = list(fb)
            fallback_used = True
    if user_profile:
        profile_excluded = _get_diet_and_allergy_exclusions(user_profile)
        food_sources = [f for f in food_sources if f not in profile_excluded]
        # hasFoodSource can point at broad category individuals ("Fish", "Meat",
        # "Dairy") that the label-based exclusion set won't catch — drop those
        # too for a vegan/vegetarian profile.
        diet_key = _DIET_URI_TO_KEY.get(
            normalize_text(user_profile.get("diet_type", "") or "")
        )
        if diet_key in ("vegan", "vegetarian"):
            animal_cats = {"fish", "meat", "poultry", "seafood", "shellfish",
                           "beef", "pork", "chicken", "lamb", "gelatin"}
            if diet_key == "vegan":
                animal_cats |= {"dairy", "milk", "eggs", "egg", "honey", "cheese"}
            food_sources = [f for f in food_sources
                            if normalize_text(f) not in animal_cats]
    if food_sources:
        shown = food_sources[:8]
        remainder = len(food_sources) - len(shown)
        suffix = f"; ... and {remainder} more" if remainder > 0 else ""
        note = " (well-established dietary sources)" if fallback_used else ""
        lines.append(f"Food sources ({len(food_sources)} total){note}: {', '.join(shown)}{suffix}")

    if user_profile:
        _dk = _DIET_URI_TO_KEY.get(normalize_text(user_profile.get("diet_type", "") or ""))
        if _dk in ("vegan", "vegetarian"):
            lines.append(
                f"NOTE — the user's diet is {user_profile.get('diet_type')}: when naming "
                f"food sources, give ONLY the plant-based ones from the 'Food sources' line "
                f"above. Do NOT present fish, seafood or meat as a source for this user, even "
                f"if the Description text mentions them."
            )

    obj_props = sorted(
        f"{get_entity_label(p)} → {get_entity_label(o)}"
        for p, o in ontology_graph.predicate_objects(entity_uri)
        if isinstance(o, URIRef) and p != RDF.type and p != SHFO.hasFoodSource
        and "food source" not in get_entity_label(p).lower()
    )
    if obj_props:
        lines.append(f"Relationships: {'; '.join(obj_props[:10])}")

    return "\n".join(lines)


def find_food_entity(question: str):
    q_norm = normalize_text(question)
    best_match = None
    best_score = 0

    for food_label, food_uri in FOOD_CANDIDATES:
        label_norm = normalize_text(food_label)
        if not label_norm:
            continue

        if label_norm in q_norm:
            score = len(label_norm)
            if score > best_score:
                best_match = food_uri
                best_score = score

        iri_tail = normalize_text(str(food_uri).split("#")[-1])
        if iri_tail and iri_tail in q_norm:
            score = len(iri_tail)
            if score > best_score:
                best_match = food_uri
                best_score = score

    return best_match

def extract_food_phrase(question: str) -> str:
    q = question.lower().strip()

    patterns = [
        r"what are the nutrients of (.+)",
        r"what is the nutrition of (.+)",
        r"what are the nutrients in (.+)",
        r"how many calories in (.+)",
        r"how much protein in (.+)",
        r"tell me the nutrients of (.+)",
        r"nutrition of (.+)",
        r"nutrients of (.+)",
        r"what is the environmental sustainability score of (.+)",
        r"what is the environmental impact of (.+)",
        r"what is the environmental footprint of (.+)",
        r"what is the carbon footprint of (.+)",
        r"environmental sustainability score of (.+)",
        r"environmental impact of (.+)",
        r"environmental footprint of (.+)",
        r"carbon footprint of (.+)",
        r"how sustainable is (.+)",
        # "how much tryptophan is in sausage raw beef" / "how much iron does
        # spinach have" / "amount of zinc in X" — without these the whole
        # sentence was matched and every "sausage ..." label tied.
        r"how (?:much|many) [a-z0-9 ()\-]+? (?:is|are) (?:there )?in (?:a |an |the )?(.+)",
        r"how (?:much|many) [a-z0-9 ()\-]+? (?:does|do) (?:a |an |the )?(.+?) (?:have|contain)",
        r"(?:amount|content|level|quantity) of [a-z0-9 ()\-]+? in (?:a |an |the )?(.+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            return re.sub(r"^(?:a|an|the)\s+", "", m.group(1).strip(" ?.,!"))

    return q.strip(" ?.,!")

def find_food_candidates(question: str):
    food_phrase = extract_food_phrase(question)

    # normalize common spelling variants
    food_phrase_norm = normalize_text(
        food_phrase.lower()
        .replace("yoghurt", "yogurt")
    )

    matches = []

    for food_label, food_uri in FOOD_CANDIDATES:
        label_text = food_label.lower().replace("yoghurt", "yogurt")
        label_norm = normalize_text(label_text)

        iri_tail = str(food_uri).split("#")[-1].lower().replace("yoghurt", "yogurt")
        iri_tail_norm = normalize_text(iri_tail)

        score = 0

        # strongest: exact phrase match
        if food_phrase_norm == label_norm:
            score = 100
        elif food_phrase_norm == iri_tail_norm:
            score = 95

        # strong: phrase appears inside label
        elif food_phrase_norm and food_phrase_norm in label_norm:
            score = 85
        elif food_phrase_norm and food_phrase_norm in iri_tail_norm:
            score = 80

        # weaker: one important token overlap only from extracted food phrase
        else:
            phrase_tokens = set(re.findall(r"[a-z0-9]+", food_phrase.lower().replace("yoghurt", "yogurt")))
            label_tokens = set(re.findall(r"[a-z0-9]+", label_text))
            overlap = phrase_tokens.intersection(label_tokens)

            # only accept meaningful overlap
            stopwords = {"food", "foods", "the", "a", "an", "of", "in", "type"}
            overlap = {t for t in overlap if t not in stopwords and len(t) >= 4}

            if overlap:
                score = 50 + len(overlap) * 10
            else:
                # fuzzy fallback — catches spelling mistakes
                fuzz = fuzzy_label_score(food_phrase_norm, label_norm)
                if fuzz >= 0.72:
                    score = int(fuzz * 65)

        if score > 0:
            matches.append((score, food_label, food_uri))

    matches.sort(key=lambda x: (-x[0], x[1]))
    return matches

#Build nutrient answer for one chosen food
def build_nutrient_report_for_food(food_uri, intro_text: str = "", user_profile: Optional[dict] = None) -> str:
    food_label, nutrient_rows = get_all_nutrients_for_food(food_uri)
    groups = get_food_groups(food_uri)
    types = get_food_types(food_uri)

    nutrient_map = {name: value for name, value in nutrient_rows}

    lines = []
    if intro_text:
        lines.append(intro_text)

    lines.append(f"Chosen ontology food: {food_label}")
    lines.append("Serving basis: per 100g")

    if groups:
        lines.append(f"Food groups: {', '.join(groups)}")
    if types:
        lines.append(f"Ontology categories: {', '.join(types)}")

    lines.append("Calories and nutrient score:")
    for nutrient_name in ["Calories", "Nutrient Score"]:
        value = nutrient_map.get(nutrient_name, "not available in the ontology context")
        lines.append(f"- {nutrient_name}: {value}")

    lines.append("Macronutrients:")
    for nutrient_name in ["Protein", "Carbohydrates", "Fat", "Fiber", "Sugar"]:
        value = nutrient_map.get(nutrient_name, "not available in the ontology context")
        lines.append(f"- {nutrient_name}: {value}")

    lines.append("Micronutrients:")
    for nutrient_name in ["Calcium", "Iron", "Magnesium", "Zinc", "Vitamin A", "Vitamin B6", "Vitamin C", "Vitamin D", "Vitamin E"]:
        value = nutrient_map.get(nutrient_name, "not available in the ontology context")
        lines.append(f"- {nutrient_name}: {value}")

    # Everything else the food has (Tryptophan, Potassium, Starch, ...) — left
    # out of the fixed lists above, so "how much tryptophan is in X" had no
    # value in context and the LLM listed (and invented) other nutrients.
    shown = {"Calories", "Nutrient Score", "Protein", "Carbohydrates", "Fat", "Fiber", "Sugar",
             "Calcium", "Iron", "Magnesium", "Zinc", "Vitamin A", "Vitamin B6", "Vitamin C", "Vitamin D",
             "Vitamin E"}
    other = [(n, v) for n, v in nutrient_rows if n not in shown]
    if other:
        lines.append("Other nutrients:")
        lines += [f"- {n}: {v}" for n, v in other]

    return "\n".join(lines) + _profile_conflict_note([food_label], user_profile)


#Build environmental impact answer for one chosen food
def build_environmental_report_for_food(food_uri, intro_text: str = "", user_profile: Optional[dict] = None) -> str:
    food_label, env_rows = get_environmental_impact_for_food(food_uri)
    env_map = {name: value for name, value in env_rows}

    lines = []
    if intro_text:
        lines.append(intro_text)

    lines.append(f"Chosen ontology food: {food_label}")

    if not env_map:
        lines.append(
            "Environmental sustainability data: not available in the ontology for this food item."
        )
        return "\n".join(lines) + _profile_conflict_note([food_label], user_profile)

    lines.append("Environmental sustainability data:")
    for env_name in ENVIRONMENTAL_PROPERTIES.values():
        value = env_map.get(env_name, "not available in the ontology context")
        lines.append(f"- {env_name}: {value}")

    return "\n".join(lines) + _profile_conflict_note([food_label], user_profile)

#detect general question
def is_general_ontology_question(question: str) -> bool:
    q = question.lower()

    general_keywords = [
        "what is", "what are", "tell me about", "explain",
        "relationship", "relationships", "related to",
        "class", "subclass", "subclasses",
        "ontology", "entity", "entities",
        "foodhighincarbs", "foodhighinprotein", "foodhighinfat",
        "molecule", "molecules",
        "food group", "belongs to",
        "individual", "individuals", "amino acids", "hasEffectOn"
    ]

    nutrient_keywords = [
        "calories", "kcal", "protein", "carbs", "carbohydrates", "fat",
        "fiber", "fibre", "sugar", "iron", "zinc", "calcium", "magnesium",
        "vitamin", "nutrient", "nutrients", "per 100g"
    ]

    recipe_keywords = [
        "recipe", "ingredients", "ingredient", "how to make", "show recipe",
        "recipe calories", "recipe nutrients"
    ]

    has_general = any(k in q for k in general_keywords)
    has_nutrient = any(k in q for k in nutrient_keywords)
    has_recipe = any(k in q for k in recipe_keywords)

    return has_general and not has_nutrient and not has_recipe


_DEFINITION_INTENT_PATTERNS = [
    r"^what is\b", r"^what are\b", r"^what's\b",
    r"^tell me about\b", r"^explain\b", r"^describe\b",
]


def is_definition_style_question(question: str) -> bool:
    """True for 'what is X' / 'tell me about X' style phrasing, regardless of
    what entity X turns out to be."""
    q = question.lower().strip()
    return any(re.search(p, q) for p in _DEFINITION_INTENT_PATTERNS)


def find_commented_entity(question: str, llm_entity: str = "") -> Optional[URIRef]:
    """For definition-style questions, resolve the subject to an ontology entity
    and return its URI ONLY if that entity has an rdfs:comment annotation.
    This lets 'what is X' questions be answered from the ontology's own
    description even when X also matches another routing path (e.g. a Diet
    entity like KetoDiet that is also a target of RecommendedFor/notRecommendedFor
    and would otherwise be intercepted by health_recommendation routing before
    ever reaching the definition lookup)."""
    if not is_definition_style_question(question):
        return None
    subject = llm_entity if llm_entity else extract_query_subject(question)
    if not subject:
        return None
    entity_uri = GENERAL_ENTITY_INDEX.get(normalize_text(subject))
    if not entity_uri:
        return None
    if ontology_graph.value(entity_uri, RDFS.comment):
        return entity_uri
    return None


#detect nutrient question
def is_nutrient_question(question: str) -> bool:
    q = question.lower()
    keywords = [
        "nutrient", "nutrients", "nutrition", "calories", "kcal",
        "protein", "carb", "carbs", "carbohydrate", "carbohydrates",
        "fat", "fibre", "fiber", "sugar", "vitamin", "mineral",
        "iron", "magnesium", "zinc", "calcium",
        "how much", "amount of", "per 100g", "nutrient score"
    ]
    return any(keyword in q for keyword in keywords)


#detect environmental impact question
def is_environmental_question(question: str) -> bool:
    q = question.lower()
    keywords = [
        "environmental impact", "environmental footprint", "environmental score",
        "environmentally friendly", "environment", "sustainability", "sustainable",
        "carbon footprint", "carbon emissions", "co2", "co₂", "greenhouse gas", "ghg",
        "emissions", "eutrophication", "eutrophying", "land use", "climate impact",
        "eco score", "ecoscore", "eco-friendly", "how sustainable",
    ]
    return any(keyword in q for keyword in keywords)





def get_all_nutrients_for_food(food_uri) -> Tuple[str, List[Tuple[str, str]]]:
    food_label = get_entity_label(food_uri)
    nutrient_rows: List[Tuple[str, str]] = []
    seen: Set[str] = set()

    for prop_name, nutrient_name in NUTRIENT_PROPERTIES.items():
        if nutrient_name in seen:   # Kalium after Potassium — one row per nutrient
            continue
        prop_uri = SHFO[prop_name]
        for _, _, value in ontology_graph.triples((food_uri, prop_uri, None)):
            nutrient_rows.append((nutrient_name, format_nutrient_value(nutrient_name, value)))
            seen.add(nutrient_name)
            break

    return food_label, nutrient_rows


def get_food_nutrient_number(food_uri, nutrient_name: str) -> Optional[float]:
    """Numeric per-100 g value of one nutrient, trying every property that
    maps to it (Potassium, then Kalium)."""
    for prop_name, name in NUTRIENT_PROPERTIES.items():
        if name != nutrient_name:
            continue
        for obj in ontology_graph.objects(food_uri, SHFO[prop_name]):
            number = parse_numeric_literal(obj)
            if number is not None:
                return number
    return None


def get_environmental_impact_for_food(food_uri) -> Tuple[str, List[Tuple[str, str]]]:
    """Look up the environmental footprint properties (GHG/CO2, eutrophication, land
    use) for a food individual. Most foods have none of these — the ontology only has
    this data for foods matched to the environmental footprint dataset — so an empty
    result list is expected and must be reported as "not available", not treated as
    an error."""
    food_label = get_entity_label(food_uri)
    env_rows: List[Tuple[str, str]] = []

    for prop_name, env_name in ENVIRONMENTAL_PROPERTIES.items():
        prop_uri = SHFO[prop_name]
        for _, _, value in ontology_graph.triples((food_uri, prop_uri, None)):
            env_rows.append((env_name, format_nutrient_value(env_name, value)))

    return food_label, env_rows


def get_food_groups(food_uri) -> List[str]:
    groups = []
    for _, _, group in ontology_graph.triples((food_uri, SHFO.belongsToFoodGroup, None)):
        groups.append(get_entity_label(group))
    return sorted(set(groups))


def get_food_types(food_uri) -> List[str]:
    types = []
    for _, _, class_uri in ontology_graph.triples((food_uri, RDF.type, None)):
        if class_uri == SHFO.Food:
            continue
        class_label = get_entity_label(class_uri)
        if class_label.lower() != "named individual":
            types.append(class_label)
    return sorted(set(types))


def build_nutrient_context_from_question(question: str, user_profile: Optional[dict] = None) -> str:
    matches = find_food_candidates(question)

    food_phrase = extract_food_phrase(question).lower().replace("yoghurt", "yogurt")
    filtered_matches = []

    for score, label, uri in matches:
        label_check = label.lower().replace("yoghurt", "yogurt")
        if food_phrase in label_check or any(tok in label_check for tok in food_phrase.split() if len(tok) >= 4):
            filtered_matches.append((score, label, uri))

    if filtered_matches:
        matches = filtered_matches

    # Score 60 = single token overlap only — not reliable for multi-word
    # queries; zero matches is even less reliable. Either way, before giving
    # up on a FOOD match, check whether the phrase instead names a RECIPE —
    # recipes are a separate ontology class that never appears in
    # FOOD_CANDIDATES at all (built only from SHFO.Food individuals), so a
    # clean, precise entity name like "Goulash" can legitimately score ZERO
    # food matches (nothing to even weakly overlap with) even though the
    # ontology has real, correct data for it as a Recipe. This has to run
    # for the zero-match case too, not just the weak-match case — a lookup
    # phrase that's an exact recipe name with no stray words is MORE likely
    # to score zero food matches than a messy full-sentence query is (there's
    # no extra vocabulary left to coincidentally overlap with an unrelated
    # food label), so skipping this check on "not matches" let exactly the
    # cleanest, most confident entity extractions fall through ungrounded.
    if not matches or matches[0][0] < 70:
        recipe_uri = find_recipe_entity(question)
        if recipe_uri:
            recipe_label = get_entity_label(recipe_uri)
            # Dedicated variable, NOT last_food_matches — that list feeds the
            # "show me another food instance" follow-up (build_next_food_
            # from_memory), which expects Food URIs specifically and calls
            # Food-specific context builders on them. Stuffing a Recipe URI
            # in there to make the trace panel show the right matched_entity
            # would risk breaking that unrelated feature for a cosmetic fix.
            st.session_state.last_nutrient_lookup_entity_uri = recipe_uri
            intro = (
                f"According to the SustainHealthyFood Ontology, '{recipe_label}' is a recipe.\n"
                f"Here are its nutrient values and details:"
            )
            return build_single_recipe_context(recipe_uri, intro, user_profile=user_profile)

    if not matches:
        return (
            "No food entity from the SustainHealthyFood Ontology was matched to the user's question."
        )

    if matches[0][0] < 70:
        close_names = [lbl for _, lbl, _ in matches[:5]]
        return (
            f"No food entity closely matching '{food_phrase.title()}' was found in the "
            f"SustainHealthyFood Ontology. "
            f"The closest partial matches are: {', '.join(close_names)}. "
            f"If one of these is what you meant, please ask again using that exact name."
        )

    chosen_food_uri = matches[0][2]
    chosen_food_label = matches[0][1]

    st.session_state.last_food_matches = [uri for _, _, uri in matches]
    st.session_state.last_food_index = 0
    st.session_state.last_nutrient_lookup_entity_uri = chosen_food_uri

    other_names = [label for _, label, _ in matches[1:6]]

    intro = (
        "According to the SustainHealthyFood Ontology, I found one or more food instances matching the user's question.\n"
        f"I selected one ontology food to present now: {chosen_food_label}"
    )

    context = build_nutrient_report_for_food(chosen_food_uri, intro, user_profile=user_profile)

    if other_names:
        context += (
            "\nOther matching ontology food instances include: "
            + ", ".join(other_names)
            + ". If the user wants, show another matching food instance from the ontology."
        )

    return context


def render_asked_nutrient_answer(question: str, food_uri: str, ontology_context: str) -> Optional[str]:
    """Answer "how much <nutrient> is in <food>" straight from the graph.

    Given the full report, llama3.2 answered "how much tryptophan" with a
    generic nutrient list — and Sodium/Cholesterol values the ontology doesn't
    even store. When the question names specific nutrients and a Food was
    resolved, there is nothing to compose. Returns None otherwise (general
    "nutrients of X" questions, recipes), leaving those to the LLM."""
    if not food_uri or (URIRef(food_uri), RDF.type, SHFO.Food) not in ontology_graph:
        return None
    q = question.lower()
    found: Dict[str, int] = {}   # nutrient -> position in the question
    for keyword, name in _comparison_nutrient_keywords():
        pattern = r"\b" + re.escape(keyword) + r"s?\b"
        m = re.search(pattern, q)
        if m:
            found.setdefault(name, m.start())
            # blank it out (same length) so "fat" can't re-match inside
            # "saturated fat", without shifting later positions
            q = re.sub(pattern, lambda x: " " * len(x.group(0)), q)
    if not found:
        return None
    asked = sorted(found, key=found.get)

    label, rows = get_all_nutrients_for_food(URIRef(food_uri))
    values = dict(rows)

    def _name(n: str) -> str:
        return n if n.startswith("Vitamin") else n.lower()

    have = [
        values[n] if n == "Calories" else f"{values[n]} of {_name(n)}"
        for n in asked if n in values and n != "Nutrient Score"
    ]
    missing = [_name(n) for n in asked if n not in values]

    lines = []
    if have:
        lines.append(
            f"According to the SustainHealthyFood Ontology, {label} contains "
            f"{' and '.join(have) if len(have) <= 2 else ', '.join(have[:-1]) + ' and ' + have[-1]} per 100 g."
        )
    if "Nutrient Score" in asked and "Nutrient Score" in values:
        lines.append(f"The nutrient score of {label} in the ontology is {values['Nutrient Score']}.")
    if missing:
        lines.append(f"The ontology has no {', '.join(missing)} value for {label}.")
    m = re.search(r"Other matching ontology food instances include: (.+?)\. If the user wants", ontology_context)
    if m:
        lines.append(
            f"\nOther matching items in the ontology: {m.group(1)}. "
            f"Ask about one of them by name if you meant a different item."
        )
    return "\n".join(lines)


# ── Nutrient comparison ("which has more iron, chicken or beef?") ──────────
# The single-food lookup above can't answer these: it matches ONE food against
# the whole question, so "chicken or beef" matched nothing and the LLM filled
# the gap from memory while still crediting the ontology. And even with both
# values in context, llama3.2 sometimes named the wrong winner (1.4 mg "more
# than" 2.8 mg). So each food is resolved separately, against EVERY matching
# NEVO item rather than one arbitrary pick (all "chicken ..." labels tie at
# score 85, so the "chosen" one is just alphabetical), and the verdict is
# computed here and prepended to the answer — same pattern as
# food_suitability's VERDICT_SENTENCE.
_COMPARISON_NUTRIENT_ALIASES: Dict[str, str] = {
    "calorie": "Calories", "kcal": "Calories",
    "carb": "Carbohydrates", "carbohydrate": "Carbohydrates",
    "fibre": "Fiber",
    "saturated fat": "Saturated Fatty Acids",
    "plant protein": "Protein (Plant)", "animal protein": "Protein (Animal)",
}
_COMPARISON_CUE_RE = re.compile(
    r"\b(more|less|fewer|higher|lower|richer|most|least|compare[sd]?|comparing|comparison|versus|vs)\b"
)
_COMPARISON_SPLIT_RE = re.compile(r"\s*(?:\bor\b|\bvs\b\.?|\bversus\b|\band\b|\bthan\b|,)\s*")
_COMPARISON_FILLER = {
    "which", "what", "food", "foods", "item", "one", "has", "have", "had", "contains", "contain",
    "containing", "does", "do", "is", "are", "there", "the", "a", "an", "in", "of", "between",
    "per", "100g", "100", "g", "gram", "grams", "amount", "amounts", "content", "value", "values",
    "level", "levels", "tell", "show", "me", "please", "better", "source", "sources", "with", "for",
    "rich", "high", "low", "nutrient", "nutrients",
    # "is it true that X has more iron than Y?" — myth-check phrasing
    "it", "true", "false", "that", "really", "correct", "right", "actually", "i", "heard", "think",
}


def _comparison_nutrient_keywords() -> List[Tuple[str, str]]:
    keywords = {name.lower(): name for name in NUTRIENT_PROPERTIES.values()}
    keywords.update(_COMPARISON_NUTRIENT_ALIASES)
    # Longest first, so "saturated fat" is consumed before "fat".
    return sorted(keywords.items(), key=lambda kv: -len(kv[0]))


def parse_nutrient_comparison(question: str) -> Optional[Tuple[List[str], List[str]]]:
    """Return (nutrient names, food phrases) for a question comparing a nutrient
    across two or more named foods, else None."""
    q = question.lower()
    if not _COMPARISON_CUE_RE.search(q):
        return None
    # "more iron that beef" / "then beef" — a common typo for "than". Without
    # it the two foods never split, the question falls to nutrient_lookup and
    # only the first food's values are shown. Only rewritten after a
    # comparative word, and only when there's no real "than".
    if not re.search(r"\bthan\b", q):
        q = re.sub(r"\b(more|less|fewer|higher|lower|richer)\b([^?]*?)\b(?:that|then)\b",
                   r"\1\2than", q, count=1)

    nutrients: List[str] = []
    for keyword, name in _comparison_nutrient_keywords():
        pattern = r"\b" + re.escape(keyword) + r"s?\b"
        if re.search(pattern, q):
            if name not in nutrients:
                nutrients.append(name)
            q = re.sub(pattern, " ", q)
    if not nutrients:
        return None

    q = re.sub(r"[?!.:;()]", " ", q).replace(",", " , ")
    words = [
        w for w in q.split()
        if w not in _COMPARISON_FILLER
        and (w in ("vs", "versus") or not _COMPARISON_CUE_RE.fullmatch(w))
    ]
    foods = [p.strip() for p in _COMPARISON_SPLIT_RE.split(" ".join(words)) if p.strip()]
    if len(foods) < 2:
        return None
    return nutrients, foods[:4]


def _resolve_comparison_food(phrase: str) -> Tuple[Optional[List[Tuple[str, object]]], List[str]]:
    """All ontology foods tied for the best match to `phrase` (preferring labels
    that start with it, so "beef" means "Beef ..." items, not "Corned beef").
    Returns (None, closest labels) when nothing matches well enough."""
    # "beef meat" / "chicken meat": the trailing "meat" makes the phrase match
    # soups and meatballs ("Soup clear w meat (beef/chicken)") instead of the
    # 38 plain "Beef ..." items. Drop it when what's left still names foods.
    m = re.fullmatch(r"(.+?)\s+meats?", phrase.strip())
    if m:
        stripped, _ = _resolve_comparison_food(m.group(1))
        if stripped:
            return stripped, []
    matches = find_food_candidates(phrase)
    if not matches or matches[0][0] < 70:
        return None, [label for _, label, _ in matches[:5]]
    top = matches[0][0]
    tier = [(label, uri) for score, label, uri in matches if score == top]
    starts = [(label, uri) for label, uri in tier if label.lower().startswith(phrase.lower())]
    return (starts or tier), []


def build_nutrient_comparison_context(question: str, user_profile: Optional[dict] = None) -> Optional[str]:
    parsed = parse_nutrient_comparison(question)
    if not parsed:
        return None
    nutrients, phrases = parsed

    resolved = []
    for phrase in phrases:
        items, closest = _resolve_comparison_food(phrase)
        if items is None:
            hint = f" The closest partial matches are: {', '.join(closest)}." if closest else ""
            return (
                f"No food entity closely matching '{phrase.title()}' was found in the "
                f"SustainHealthyFood Ontology, so this comparison can't be made from ontology data.{hint} "
                f"If one of these is what you meant, please ask again using that exact name."
            )
        resolved.append((phrase, items))

    lines = ["NUTRIENT COMPARISON CONTEXT (SustainHealthyFood Ontology, NEVO values per 100 g)"]
    verdicts = []

    for nutrient in nutrients:
        fmt = lambda n: format_nutrient_value(nutrient, n)
        stats = []  # (display name, [(label, value)], median)
        for phrase, items in resolved:
            values = []
            for label, uri in items:
                number = get_food_nutrient_number(uri, nutrient)
                if number is not None:
                    values.append((label, number))
            name = values[0][0] if len(values) == 1 else phrase.capitalize()
            stats.append((name, sorted(values, key=lambda lv: -lv[1]),
                          statistics.median(v for _, v in values) if values else None))

        lines.append(f"\n{nutrient}:")
        for name, values, med in stats:
            if not values:
                lines.append(f"{name}: {nutrient} not available in the ontology")
                continue
            if len(values) == 1:
                lines.append(f"{name}: {fmt(values[0][1])}")
            else:
                lines.append(
                    f"{name} ({len(values)} ontology items): median {fmt(med)}, "
                    f"range {fmt(values[-1][1])}–{fmt(values[0][1])}"
                )
                for label, value in values[:12]:
                    lines.append(f"  - {label}: {fmt(value)}")
                if len(values) > 12:
                    lines.append(f"  - ... and {len(values) - 12} more")

        missing = [name for name, values, _ in stats if not values]
        if missing:
            verdicts.append(
                f"The ontology has no {nutrient.lower()} values for {', '.join(missing)}, "
                f"so the {nutrient.lower()} comparison can't be made from ontology data."
            )
            continue

        multi = any(len(values) > 1 for _, values, _ in stats)
        basis = "median per 100 g across the matching ontology items" if multi else "per 100 g"
        ranked = sorted(stats, key=lambda s: -s[2])
        if len(ranked) == 2:
            (hi_name, hi_vals, hi_med), (lo_name, lo_vals, lo_med) = ranked
            if hi_med == lo_med:
                verdict = (f"{hi_name} and {lo_name} have the same amount of {nutrient.lower()} "
                           f"({fmt(hi_med)}, {basis}).")
            else:
                verdict = (f"{hi_name} has more {nutrient.lower()} than {lo_name} "
                           f"({fmt(hi_med)} vs {fmt(lo_med)}, {basis}).")
                # Not a plain range-overlap test: one outlier (e.g. "Beef fat",
                # 0 mg iron) would trigger it for almost every comparison.
                if multi and max(v for _, v in lo_vals) >= hi_med:
                    verdict += (f" Some {lo_name.lower()} items reach that level, so individual "
                                f"cuts/preparations can differ.")
        else:
            verdict = (f"Ranked by {nutrient.lower()} ({basis}): "
                       + " > ".join(f"{name} {fmt(med)}" for name, _, med in ranked) + ".")
        verdicts.append(verdict)

    for verdict in verdicts:
        lines.append(f"COMPARISON_VERDICT: {verdict}")
    return "\n".join(lines)


def render_nutrient_comparison_answer(context: str, examples_per_food: int = 3) -> str:
    """The final answer for nutrient_comparison, built from its context with no
    LLM call: everything in it is already a number from the context, and the
    LLM was seen filing one food's items under the other food's heading."""
    verdicts = re.findall(r"^COMPARISON_VERDICT:\s*(.+)$", context, re.M)
    out = verdicts + [""]
    shown = 0
    for line in context.splitlines()[1:]:
        if not line.strip() or line.startswith("COMPARISON_VERDICT:"):
            continue
        if line.startswith("  - "):
            if shown < examples_per_food and not line.startswith("  - ... and"):
                out.append(line)
                shown += 1
            continue
        shown = 0
        out.append(f"**{line[:-1]}**" if line.endswith(":") else f"- {line}")
    out.append("\nAll values are per 100 g, from the NEVO data in the SustainHealthyFood Ontology.")
    return "\n".join(out)


# ── Nutrient ranking ("which fruit has the highest amount of vitamin c?") ──
# A superlative over a whole food group, not two named foods — the comparison
# route above needs >= 2 foods, so this fell to nutrient_lookup, which picked
# one fruit (Orange) and had the LLM list its nutrients. Ranked here over every
# food in the group, no LLM call.
_RANKING_HIGH_RE = re.compile(r"\b(highest|most|richest|best source|best sources|top|greatest|largest|maximum)\b")
_RANKING_LOW_RE = re.compile(r"\b(lowest|least|fewest|smallest|minimum|poorest)\b")
# Question word -> (food group individual or None for all foods, display noun)
_RANKING_SCOPES: List[Tuple[str, Optional[str], str]] = [
    (r"fruits?", "foodgroup_fruits", "fruit"),
    (r"vegetables?|veggies?|veg", "foodgroup_vegetables", "vegetable"),
    (r"nuts?|seeds?", "foodgroup_nuts_and_seeds", "nut or seed"),
    (r"legumes?|beans?|pulses?", "foodgroup_legumes", "legume"),
    (r"meats?", "foodgroup_meat", "meat"),
    (r"cheeses?", "foodgroup_cheese", "cheese"),
    (r"dairy|milk products?", "foodgroup_milk_and_milk_products", "dairy product"),
    (r"eggs?", "foodgroup_eggs", "egg product"),
    (r"breads?", "foodgroup_bread", "bread"),
    (r"cereals?|grains?", "foodgroup_cereal_products_and_types_of_flour", "cereal product"),
    (r"potato(?:es)?|starchy vegetables?", "foodgroup_starchyvegetables", "starchy vegetable"),
    (r"drinks?|beverages?", "foodgroup_nonalcoholicbeverages", "drink"),
    (r"foods?|items?|products?|ingredients?", None, "food"),
]


def parse_nutrient_ranking(question: str) -> Optional[Tuple[str, str, Optional[str], str, int]]:
    """(nutrient, 'high'|'low', food group or None, scope noun, n) for a
    "which <group> has the highest/lowest <nutrient>" question, else None."""
    q = question.lower()
    high, low = _RANKING_HIGH_RE.search(q), _RANKING_LOW_RE.search(q)
    if not (high or low):
        return None
    # Meal-type questions ("most protein for breakfast") have their own route.
    if re.search(r"\b(breakfast|lunch|dinner|snack|pre ?workout|post ?workout)\b", q):
        return None
    nutrient = None
    for keyword, name in _comparison_nutrient_keywords():
        if re.search(r"\b" + re.escape(keyword) + r"s?\b", q):
            nutrient = name
            break
    if nutrient is None:
        return None
    for pattern, group, noun in _RANKING_SCOPES:
        if re.search(rf"\b(?:{pattern})\b", q):
            break
    else:
        return None
    n = 5
    m = re.search(r"\btop\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:\w+\s+)?(?:fruits?|vegetables?|foods?|items?)\b", q)
    if m:
        n = max(1, min(int(m.group(1) or m.group(2)), 15))
    # "top 3 foods with the lowest sodium": "top" only sets the count.
    direction = "low" if low else "high"
    return nutrient, direction, group, noun, n


def build_nutrient_ranking_context(question: str, user_profile: Optional[dict] = None) -> Optional[str]:
    parsed = parse_nutrient_ranking(question)
    if not parsed:
        return None
    nutrient, direction, group, noun, n = parsed
    if group:
        foods = [
            (get_entity_label(f), f)
            for f in ontology_graph.subjects(SHFO.belongsToFoodGroup, SHFO[group])
            if isinstance(f, URIRef)
        ]
    else:
        foods = list(FOOD_CANDIDATES)
    ranked = [
        (label, v) for label, uri in foods
        if (v := get_food_nutrient_number(uri, nutrient)) is not None
    ]
    if not ranked:
        return None
    ranked.sort(key=lambda lv: (-lv[1], lv[0]) if direction == "high" else (lv[1], lv[0]))
    top = ranked[:n]
    word = "most" if direction == "high" else "least"
    best_label, best_value = top[0]
    lines = [
        "NUTRIENT RANKING CONTEXT (SustainHealthyFood Ontology, NEVO values per 100 g)",
        f"RANKING_VERDICT: In the SustainHealthyFood Ontology, the {noun} with the {word} "
        f"{nutrient if nutrient.startswith('Vitamin') else nutrient.lower()} is {best_label} "
        f"({format_nutrient_value(nutrient, best_value)} per 100 g).",
        f"Top {len(top)} of {len(ranked)} {noun} items with a {nutrient} value:",
    ]
    lines += [f"{i}. {label}: {format_nutrient_value(nutrient, v)}" for i, (label, v) in enumerate(top, 1)]
    return "\n".join(lines) + _profile_conflict_note([label for label, _ in top], user_profile)


def render_nutrient_ranking_answer(context: str) -> str:
    """Final answer for nutrient_ranking — straight from the context, no LLM."""
    verdict = re.search(r"^RANKING_VERDICT:\s*(.+)$", context, re.M).group(1)
    rows = [
        l for l in context.splitlines()
        if re.match(r"^(\d+\. |Top \d+ of )", l)
    ]
    rows = [f"**{rows[0]}**"] + rows[1:]
    return "\n".join([verdict, ""] + rows + [
        "\nAll values are per 100 g, from the NEVO data in the SustainHealthyFood Ontology."
    ])


def build_environmental_context_from_question(question: str, user_profile: Optional[dict] = None) -> str:
    """Same food-matching logic as build_nutrient_context_from_question, but reports
    the environmental footprint properties instead of nutrients. Environmental data
    only exists for a minority of foods, so a matched food with none is reported
    explicitly as unavailable rather than silently omitted."""
    matches = find_food_candidates(question)

    food_phrase = extract_food_phrase(question).lower().replace("yoghurt", "yogurt")
    filtered_matches = []

    for score, label, uri in matches:
        label_check = label.lower().replace("yoghurt", "yogurt")
        if food_phrase in label_check or any(tok in label_check for tok in food_phrase.split() if len(tok) >= 4):
            filtered_matches.append((score, label, uri))

    if filtered_matches:
        matches = filtered_matches

    if not matches:
        return (
            "No food entity from the SustainHealthyFood Ontology was matched to the user's question."
        )

    if matches[0][0] < 70:
        close_names = [lbl for _, lbl, _ in matches[:5]]
        return (
            f"No food entity closely matching '{food_phrase.title()}' was found in the "
            f"SustainHealthyFood Ontology. "
            f"The closest partial matches are: {', '.join(close_names)}. "
            f"If one of these is what you meant, please ask again using that exact name."
        )

    chosen_food_uri = matches[0][2]
    chosen_food_label = matches[0][1]

    st.session_state.last_food_matches = [uri for _, _, uri in matches]
    st.session_state.last_food_index = 0

    intro = (
        "According to the SustainHealthyFood Ontology, I found one or more food instances matching the user's question.\n"
        f"I selected one ontology food to present now: {chosen_food_label}"
    )

    return build_environmental_report_for_food(chosen_food_uri, intro, user_profile=user_profile)


#detect recipe question
def is_recipe_question(question: str) -> bool:
    q = question.lower()
    recipe_keywords = [
        "recipe", "ingredients", "ingredient", "how to make",
        "what is in", "show recipe", "give me the recipe",
        "amount of ingredients", "recipe nutrients", "recipe calories"
    ]
    return any(k in q for k in recipe_keywords)

#Find the recipe entity from the user question

def find_recipe_entity(question: str):
    q_norm = normalize_text(question)

    for recipe in ontology_graph.subjects(RDF.type, SHFO.Recipe):
        recipe_local = str(recipe).split("#")[-1]
        if normalize_text(recipe_local) in q_norm:
            return recipe
        recipe_label = ontology_graph.value(recipe, RDFS.label)
        if recipe_label and normalize_text(str(recipe_label)) in q_norm:
            return recipe

    # Fuzzy fallback — catches spelling mistakes (e.g. "goualsh" → "Goulash")
    best_match = None
    best_score = 0.0
    for recipe in ontology_graph.subjects(RDF.type, SHFO.Recipe):
        recipe_label = ontology_graph.value(recipe, RDFS.label)
        label_norm = normalize_text(str(recipe_label)) if recipe_label else normalize_text(str(recipe).split("#")[-1])
        score = fuzzy_label_score(q_norm, label_norm)
        if score > best_score and score >= 0.70:
            best_score = score
            best_match = recipe
    return best_match

#Retrieve all recipes
def get_all_recipes_sparql():
    q = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX shfo: <http://SustainHealthyFoodOntology#>

    SELECT ?recipe ?label
    WHERE {
        ?recipe rdf:type shfo:Recipe .
        OPTIONAL { ?recipe rdfs:label ?label . }
    }
    """
    results = ontology_graph.query(q)
    return [
        {
            "uri": str(row.recipe),
            "label": str(row.label) if row.label else str(row.recipe).split("#")[-1]
        }
        for row in results
    ]

#Retrieve ingredients for a recipe
def get_recipe_ingredients(recipe_uri):
    ingredients = []

    for _, _, ingredient in ontology_graph.triples((recipe_uri, SHFO.hasIngredient, None)):
        ingredient_label = ontology_graph.value(ingredient, RDFS.label)
        ingredients.append({
            "uri": str(ingredient),
            "label": str(ingredient_label) if ingredient_label else str(ingredient).split("#")[-1]
        })

    return ingredients

#Retrieve ingredient amounts for a recipe
def get_recipe_ingredient_amounts(recipe_uri):
    amounts = []
    for _, _, amount in ontology_graph.triples((recipe_uri, SHFO.hasIngredientAmount, None)):
        amounts.append(str(amount))
    return amounts

#retrieve recipe nutrients totals
RECIPE_NUTRIENT_PROPERTIES = {
    "hasAmountOfCalories": "Calories",
    "hasAmountOfProtein": "Protein",
    "hasAmountOfProteinPlant": "Protein (Plant)",
    "hasAmountOfProteinAnimal": "Protein (Animal)",
    "hasAmountOfCarbohydrates": "Carbohydrates",
    "hasAmountOfStarch": "Starch",
    "hasAmountOfSugar": "Sugar",
    "hasAmountOfPolyols": "Polyols",
    "hasAmountOfFat": "Fat",
    "hasAmountOfFattyAcids": "Fatty Acids",
    "hasAmountOfSaturatedFattyAcids": "Saturated Fatty Acids",
    "hasAmountOfFiber": "Fiber",
    "hasAmountOfDietaryFiber": "Dietary Fiber",
    "hasAmountOfWater": "Water",
    "hasAmountOfTryptophan": "Tryptophan",
    "hasAmountOfCalcium": "Calcium",
    "hasAmountOfIron": "Iron",
    "hasAmountOfMagnesium": "Magnesium",
    "hasAmountOfZinc": "Zinc",
    "hasAmountOfPotassium": "Potassium",
    "hasAmountOfKalium": "Potassium",   # see NUTRIENT_PROPERTIES
    "hasAmountOfSodium": "Sodium",
    "hasAmountOfVitaminA": "Vitamin A",
    "hasAmountOfVitaminB6": "Vitamin B6",
    "hasAmountOfVitaminC": "Vitamin C",
    "hasAmountOfVitaminD": "Vitamin D",
    "hasAmountOfVitaminE": "Vitamin E",
    "hasAmountOfVitaminK": "Vitamin K",
    "hasNutrientScore": "Nutrient Score",
}

def get_all_nutrients_for_recipe(recipe_uri):
    nutrients = []
    seen = set()

    for prop_name, nice_name in RECIPE_NUTRIENT_PROPERTIES.items():
        if nice_name in seen:   # Kalium after Potassium — one row per nutrient
            continue
        prop_uri = SHFO[prop_name]
        for _, _, value in ontology_graph.triples((recipe_uri, prop_uri, None)):
            nutrients.append((nice_name, str(value)))
            seen.add(nice_name)
            break

    return nutrients


def build_single_recipe_context(recipe_uri, intro_text: str = "", user_profile: Optional[dict] = None) -> str:
    recipe_label = get_entity_label(recipe_uri)
    ingredients = get_recipe_ingredients(recipe_uri)
    ingredient_amounts = get_recipe_ingredient_amounts(recipe_uri)
    nutrients = get_all_nutrients_for_recipe(recipe_uri)

    lines = []
    if intro_text:
        lines.append(intro_text)

    lines.append(f"Chosen ontology recipe: {recipe_label}")

    if ingredients:
        lines.append("Ingredients:")
        for ing in ingredients:
            lines.append(f"- {ing['label']}")
    else:
        lines.append("Ingredients: not available in ontology context.")

    if ingredient_amounts:
        lines.append("Ingredient amounts:")
        for amt in ingredient_amounts:
            lines.append(f"- {amt}")
    else:
        lines.append("Ingredient amounts: not available in ontology context.")

    if nutrients:
        lines.append("Recipe nutrient values:")
        for nutrient_name, value in nutrients:
            lines.append(f"- {nutrient_name}: {value}")
    else:
        lines.append("Recipe nutrient values: not available in ontology context.")

    ingredient_labels = [recipe_label] + [ing["label"] for ing in ingredients]
    return "\n".join(lines) + _profile_conflict_note(ingredient_labels, user_profile)

def detect_meal_prep_recipe_request(question: str) -> Optional[int]:
    """Return number of recipes requested if question asks for N balanced/varied recipes for meal prep, else None."""
    q = question.lower()
    match = re.search(r"(\d+)\s+(balanced|varied|different|healthy|diverse)?\s*recipes?", q)
    if match and any(t in q for t in ["meal prep", "prep for the week", "week", "balanced recipes", "balanced meal"]):
        return int(match.group(1))
    if "meal prep" in q and "recipe" in q:
        return 3
    return None


def build_meal_prep_context(n: int = 3, user_profile: Optional[dict] = None) -> str:
    """Pick n diverse recipes from the ontology and build combined context for meal prep."""
    import random
    all_recipes = list(ontology_graph.subjects(RDF.type, SHFO.Recipe))
    if not all_recipes:
        return "No recipes found in the ontology."
    chosen = random.sample(all_recipes, min(n, len(all_recipes)))
    sections = []
    for i, recipe_uri in enumerate(chosen, 1):
        sections.append(f"--- Recipe {i} ---")
        sections.append(build_single_recipe_context(recipe_uri, user_profile=user_profile))
    return "\n".join(sections)


#complete recipe function
def build_recipe_context_from_question(question: str, user_profile: Optional[dict] = None) -> str:
    recipe_uri = find_recipe_entity(question)

    if not recipe_uri:
        all_recipes = get_all_recipes_sparql()
        if all_recipes:
            recipe_names = ", ".join(r["label"] for r in all_recipes[:20])
            return f"No recipe entity matched this question. Available recipe examples in the ontology include: {recipe_names}"
        return "No recipe entity found in the ontology."

    recipe_label = ontology_graph.value(recipe_uri, RDFS.label)
    recipe_label = str(recipe_label) if recipe_label else str(recipe_uri).split("#")[-1]

    ingredients = get_recipe_ingredients(recipe_uri)
    ingredient_amounts = get_recipe_ingredient_amounts(recipe_uri)
    nutrients = get_all_nutrients_for_recipe(recipe_uri)

    lines = [f"According to the SustainHealthyFood Ontology, the recipe '{recipe_label}' has the following ontology facts."]

    if ingredients:
        lines.append("Ingredients:")
        for ing in ingredients:
            lines.append(f"- {ing['label']}")
    else:
        lines.append("Ingredients: not available in ontology context.")

    if ingredient_amounts:
        lines.append("Ingredient amounts:")
        for amt in ingredient_amounts:
            lines.append(f"- {amt}")
    else:
        lines.append("Ingredient amounts: not available in ontology context.")

    if nutrients:
        lines.append("Recipe nutrient values:")
        for nutrient_name, value in nutrients:
            lines.append(f"- {nutrient_name}: {value}")
    else:
        lines.append("Recipe nutrient values: not available in ontology context.")

    ingredient_labels = [recipe_label] + [ing["label"] for ing in ingredients]
    return "\n".join(lines) + _profile_conflict_note(ingredient_labels, user_profile)

#finding a recipe based on ingredient
def is_recipe_by_ingredient_question(question: str) -> bool:
    q = question.lower()
    triggers = [
        "recipe with ",
        "recipes with ",
        "recipe containing ",
        "recipes containing ",
        "recipe using ",
        "recipes using ",
        "dish with ",
        "dessert with "
    ]
    return any(t in q for t in triggers)


#Extract ingredient phrase from the question
def extract_ingredient_phrase(question: str) -> str:
    q = question.lower().strip()

    patterns = [
        r"recipe[s]?\s+with\s+(.+)",
        r"recipe[s]?\s+containing\s+(.+)",
        r"recipe[s]?\s+using\s+(.+)",
        r"dessert\s+with\s+(.+)",
        r"dish\s+with\s+(.+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            phrase = m.group(1).strip(" ?.,!")
            return phrase

    return q.strip(" ?.,!")

#Find ingredient candidates by partial label match
def find_ingredient_candidates(question: str):
    ingredient_phrase = extract_ingredient_phrase(question)
    phrase_norm = normalize_text(ingredient_phrase)

    matches = []
    seen = set()

    for food_label, food_uri in FOOD_CANDIDATES:
        label_norm = normalize_text(food_label)
        iri_tail_norm = normalize_text(str(food_uri).split("#")[-1])

        score = 0

        # exact normalized match
        if phrase_norm == label_norm or phrase_norm == iri_tail_norm:
            score = 100
        # user phrase contained in label
        elif phrase_norm and phrase_norm in label_norm:
            score = 80
        elif phrase_norm and phrase_norm in iri_tail_norm:
            score = 70
        # label contained in user phrase
        elif label_norm and label_norm in phrase_norm:
            score = 60
        elif iri_tail_norm and iri_tail_norm in phrase_norm:
            score = 50

        if score > 0 and food_uri not in seen:
            seen.add(food_uri)
            matches.append((score, food_label, food_uri))

    matches.sort(key=lambda x: (-x[0], x[1]))
    return matches

#Get recipes for any matching ingredient candidate
def get_recipes_by_any_ingredient_candidates(ingredient_candidates):
    recipe_hits = []
    seen = set()

    candidate_uris = {uri for _, _, uri in ingredient_candidates}

    for recipe, _, ingredient in ontology_graph.triples((None, SHFO.hasIngredient, None)):
        if ingredient in candidate_uris and recipe not in seen:
            seen.add(recipe)
            recipe_hits.append(recipe)

    return recipe_hits




def build_recipe_by_ingredient_context(question: str, user_profile: Optional[dict] = None) -> str:
    ingredient_candidates = find_ingredient_candidates(question)

    if not ingredient_candidates:
        return (
            "No ingredient entity from the SustainHealthyFood Ontology matched the user's question."
        )

    matched_ingredient_names = [label for _, label, _ in ingredient_candidates[:10]]
    recipes = get_recipes_by_any_ingredient_candidates(ingredient_candidates)

    if not recipes:
        return (
            "According to the SustainHealthyFood Ontology, ingredient candidates were matched "
            f"({', '.join(matched_ingredient_names)}), but no recipes were found using them."
        )

    st.session_state.last_recipe_matches = recipes
    st.session_state.last_recipe_index = 0
    st.session_state.last_recipe_query_type = "ingredient"

    chosen_recipe = recipes[0]
    chosen_label = get_entity_label(chosen_recipe)
    matched_recipe_names = [get_entity_label(r) for r in recipes[:5]]

    intro = (
        "According to the SustainHealthyFood Ontology, multiple recipes matched the requested ingredient.\n"
        f"Matched ingredient candidates: {', '.join(matched_ingredient_names)}\n"
        f"Matched recipes: {', '.join(matched_recipe_names)}\n"
        f"I selected one ontology recipe to present now: {chosen_label}"
    )

    context = build_single_recipe_context(chosen_recipe, intro, user_profile=user_profile)

    if len(recipes) > 1:
        context += "\nIf the user wants, I can show another matching recipe from the ontology."

    return context


def find_recipes_by_title_keyword(question: str):
    q = question.lower().strip()

    patterns = [
        r"recipe for (.+)",
        r"recipe of (.+)",
        r"give me a recipe for (.+)",
        r"show me a recipe for (.+)",
        r"can you give a recipe for (.+)",
        r"can you give me a recipe for (.+)",
    ]

    keyword = None
    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            keyword = m.group(1).strip(" ?.,!")
            break

    if not keyword:
        return []

    keyword_norm = normalize_text(keyword)
    matches = []

    for recipe in ontology_graph.subjects(RDF.type, SHFO.Recipe):
        recipe_label = get_entity_label(recipe)
        label_norm = normalize_text(recipe_label)
        iri_tail_norm = normalize_text(str(recipe).split("#")[-1])

        if keyword_norm and (keyword_norm in label_norm or keyword_norm in iri_tail_norm):
            matches.append(recipe)

    return matches

def _extract_recipe_keyword(question: str) -> Optional[str]:
    """Extract the target dish/ingredient phrase from a recipe-for-X question."""
    q = question.lower().strip()
    patterns = [
        r"give me a recipe for (.+)",
        r"show me a recipe for (.+)",
        r"can you give me a recipe for (.+)",
        r"can you give a recipe for (.+)",
        r"recipe for (.+)",
        r"recipe of (.+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            return m.group(1).strip(" ?.,!")
    return None


def _find_ingredient_candidates_for_phrase(phrase: str):
    """
    Find ingredient candidates for a phrase, trying the full phrase first,
    then falling back to individual words (handles multi-word names like 'banana bread').
    """
    candidates = find_ingredient_candidates(f"recipe with {phrase}")
    if candidates:
        return candidates
    # Word-by-word fallback for multi-word phrases
    seen_uris: set = set()
    merged: list = []
    for word in phrase.split():
        if len(word) < 4:
            continue
        for cand in find_ingredient_candidates(f"recipe with {word}"):
            uri_str = str(cand[2])
            if uri_str not in seen_uris:
                seen_uris.add(uri_str)
                merged.append(cand)
    merged.sort(key=lambda x: -x[0])
    return merged


def build_recipe_title_search_context(question: str, user_profile: Optional[dict] = None) -> str:
    # Single keyword extraction: try pattern ("recipe for X"), else treat entire input as the keyword.
    # This handles both full questions ("give me a recipe for goulash") and
    # direct entity names passed by the LLM classifier ("Minced Beef Ball").
    keyword = _extract_recipe_keyword(question) or question.strip(" ?.,!")
    keyword_norm = normalize_text(keyword)

    # Level 1: keyword search in recipe labels and IRIs
    matching_recipes = []
    if keyword_norm:
        for recipe in ontology_graph.subjects(RDF.type, SHFO.Recipe):
            label_norm = normalize_text(get_entity_label(recipe))
            iri_norm = normalize_text(str(recipe).split("#")[-1])
            if keyword_norm in label_norm or keyword_norm in iri_norm:
                matching_recipes.append(recipe)

    if matching_recipes:
        st.session_state.last_recipe_matches = matching_recipes
        st.session_state.last_recipe_index = 0
        st.session_state.last_recipe_query_type = "title"
        chosen_recipe = matching_recipes[0]
        chosen_label = get_entity_label(chosen_recipe)
        matched_names = [get_entity_label(r) for r in matching_recipes[:5]]
        intro = (
            "According to the SustainHealthyFood Ontology, multiple recipe titles matched the user's request.\n"
            f"Matched recipes: {', '.join(matched_names)}\n"
            f"I selected one ontology recipe to present now: {chosen_label}"
        )
        context = build_single_recipe_context(chosen_recipe, intro, user_profile=user_profile)
        if len(matching_recipes) > 1:
            context += "\nIf the user wants, I can show another matching recipe from the ontology."
        return context

    # Level 1.5: fuzzy title match — catches spelling mistakes (e.g. "goualsh" → "Goulash")
    if not matching_recipes and keyword_norm:
        best_recipe = None
        best_score = 0.0
        for recipe in ontology_graph.subjects(RDF.type, SHFO.Recipe):
            label_norm = normalize_text(get_entity_label(recipe))
            score = fuzzy_label_score(keyword_norm, label_norm)
            if score > best_score and score >= 0.70:
                best_score = score
                best_recipe = recipe
        if best_recipe:
            matching_recipes = [best_recipe]
            st.session_state.last_recipe_matches = matching_recipes
            st.session_state.last_recipe_index = 0
            st.session_state.last_recipe_query_type = "title"
            chosen_label = get_entity_label(best_recipe)
            intro = (
                "According to the SustainHealthyFood Ontology, a recipe closely matching the request was found.\n"
                f"I selected one ontology recipe to present now: {chosen_label}"
            )
            return build_single_recipe_context(best_recipe, intro, user_profile=user_profile)

    # Level 2: ingredient-based fallback — handles single words and multi-word phrases
    ingredient_candidates = _find_ingredient_candidates_for_phrase(keyword)
    if ingredient_candidates:
        recipes = get_recipes_by_any_ingredient_candidates(ingredient_candidates)
        if recipes:
            st.session_state.last_recipe_matches = recipes
            st.session_state.last_recipe_index = 0
            st.session_state.last_recipe_query_type = "ingredient_fallback"
            chosen = recipes[0]
            chosen_label = get_entity_label(chosen)
            matched_ing_names = [label for _, label, _ in ingredient_candidates[:5]]
            matched_recipe_names = [get_entity_label(r) for r in recipes[:5]]
            intro = (
                f"No recipe named '{keyword}' was found in the SustainHealthyFood Ontology.\n"
                f"However, recipes containing matching ingredients were found.\n"
                f"Matched ingredient candidates: {', '.join(matched_ing_names)}\n"
                f"Matched recipes: {', '.join(matched_recipe_names)}\n"
                f"I selected one ontology recipe to present now: {chosen_label}"
            )
            context = build_single_recipe_context(chosen, intro, user_profile=user_profile)
            if len(recipes) > 1:
                context += "\nIf the user wants, I can show another matching recipe from the ontology."
            return context

    # Level 3: nothing matched — list available recipes so the user can pick
    all_recipes = get_all_recipes_sparql()
    if all_recipes:
        recipe_names = ", ".join(r["label"] for r in all_recipes[:20])
        return (
            f"No recipe matching '{keyword}' was found in the SustainHealthyFood Ontology. "
            f"Available recipes in the ontology include: {recipe_names}"
        )
    return f"No recipe matching '{keyword}' was found in the SustainHealthyFood Ontology."

def is_recipe_title_search_question(question: str) -> bool:
    q = question.lower()
    triggers = [
        "recipe for",
        "recipe of",
        "give me a recipe for",
        "show me a recipe for",
        "can you give a recipe for",
        "can you give me a recipe for",
    ]
    return any(t in q for t in triggers)

def is_recipe_followup_request(question: str) -> bool:
    q = question.lower().strip()
    triggers = [
        "show another",
        "another one",
        "another recipe",
        "show me another",
        "yes show me another",
        "different recipe",
        "other recipe",
        "next recipe",
    ]
    return any(t in q for t in triggers)

def build_next_recipe_from_memory(user_profile: Optional[dict] = None) -> str:
    matches = st.session_state.get("last_recipe_matches", [])
    idx = st.session_state.get("last_recipe_index", 0)

    if not matches:
        return "There is no previous set of matched recipes to continue from."

    next_idx = idx + 1
    if next_idx >= len(matches):
        return "There are no more matching recipes in the ontology for the previous request."

    st.session_state.last_recipe_index = next_idx
    recipe_uri = matches[next_idx]

    intro = (
        "According to the SustainHealthyFood Ontology, here is another matching recipe.\n"
        f"This is option {next_idx + 1} of {len(matches)}."
    )

    context = build_single_recipe_context(recipe_uri, intro, user_profile=user_profile)

    if next_idx < len(matches) - 1:
        context += "\nIf the user wants, I can show another matching recipe from the ontology."

    return context

def is_food_followup_request(question: str) -> bool:
    q = question.lower().strip()
    triggers = [
        "show another type",
        "another type",
        "another yogurt",
        "another yoghurt",
        "another one",
        "show another one",
        "different yogurt",
        "different yoghurt",
        "different type",
        "other types",
    ]
    return any(t in q for t in triggers)

def build_next_food_from_memory(user_profile: Optional[dict] = None) -> str:
    matches = st.session_state.get("last_food_matches", [])
    idx = st.session_state.get("last_food_index", 0)

    if not matches:
        return "There is no previous set of matched food instances to continue from."

    next_idx = idx + 1
    if next_idx >= len(matches):
        return "There are no more matching food instances in the ontology for the previous request."

    st.session_state.last_food_index = next_idx
    food_uri = matches[next_idx]

    remaining_names = [get_entity_label(uri) for uri in matches[next_idx + 1: next_idx + 6]]

    intro = (
        "According to the SustainHealthyFood Ontology, here is another matching food instance.\n"
        f"This is option {next_idx + 1} of {len(matches)}."
    )

    context = build_nutrient_report_for_food(food_uri, intro, user_profile=user_profile)

    if remaining_names:
        context += (
            "\nOther matching ontology food instances include: "
            + ", ".join(remaining_names)
            + ". If the user wants, show another matching food instance from the ontology."
        )

    return context


FOOD_CLASS_KEYWORDS: Dict[str, str] = {
    "high in protein": "FoodHighInProtein",
    "high protein": "FoodHighInProtein",
    "rich in protein": "FoodHighInProtein",
    "low in protein": "FoodLowInProtein",
    "low protein": "FoodLowInProtein",
    "high in carbs": "FoodHighInCarbs",
    "high in carbohydrates": "FoodHighInCarbs",
    "high carb": "FoodHighInCarbs",
    "low in carbs": "FoodLowInCarbs",
    "low in carbohydrates": "FoodLowInCarbs",
    "low carb": "FoodLowInCarbs",
    "high in fat": "FoodHighInFat",
    "high fat": "FoodHighInFat",
    "low in fat": "FoodLowInFat",
    "low fat": "FoodLowInFat",
    "high in fiber": "FoodHighInFiber",
    "high in fibre": "FoodHighInFiber",
    "high fiber": "FoodHighInFiber",
    "high fibre": "FoodHighInFiber",
    "low in fiber": "FoodLowInFiber",
    "low in fibre": "FoodLowInFiber",
    "low fiber": "FoodLowInFiber",
    "high in iron": "FoodHighInIron",
    "high iron": "FoodHighInIron",
    "low in iron": "FoodLowInIron",
    "low iron": "FoodLowInIron",
    "high in sugar": "FoodHighInSugar",
    "high sugar": "FoodHighInSugar",
    "low in sugar": "FoodLowInSugar",
    "low sugar": "FoodLowInSugar",
    "high in salt": "FoodHighInSalt",
    "high salt": "FoodHighInSalt",
    "high sodium": "FoodHighInSalt",
    "low in salt": "FoodLowInSalt",
    "low salt": "FoodLowInSalt",
    "low sodium": "FoodLowInSalt",
    "fat free": "FoodFatFree",
    "fat-free": "FoodFatFree",
    "fermented": "FermentedFood",
    "fortified": "FortifiedFood",
    "prebiotic": "PrebioticFood",
}


def detect_food_class_query(question: str) -> Optional[str]:
    """Return the ontology class name if the question asks about a food category, else None."""
    q = question.lower()
    # Sort by length descending so longer phrases match before shorter substrings
    for keyword in sorted(FOOD_CLASS_KEYWORDS, key=len, reverse=True):
        if keyword in q:
            return FOOD_CLASS_KEYWORDS[keyword]
    return None


FOOD_CLASS_NUTRIENT_FALLBACK: Dict[str, Tuple[str, str, float]] = {
    "FoodHighInProtein":    ("hasAmountOfProtein",       ">", 20.0),
    "FoodLowInProtein":     ("hasAmountOfProtein",       "<",  3.0),
    "FoodHighInCarbs":      ("hasAmountOfCarbohydrates", ">", 40.0),
    "FoodLowInCarbs":       ("hasAmountOfCarbohydrates", "<",  5.0),
    "FoodHighInFat":        ("hasAmountOfFat",           ">", 17.5),
    "FoodLowInFat":         ("hasAmountOfFat",           "<",  3.0),
    "FoodHighInFiber":      ("hasAmountOfFiber",         ">",  6.0),
    "FoodLowInFiber":       ("hasAmountOfFiber",         "<",  2.0),
    "FoodHighInSugar":      ("hasAmountOfSugar",         ">", 22.5),
    "FoodLowInSugar":       ("hasAmountOfSugar",         "<",  5.0),
    "FoodHighInIron":       ("hasAmountOfIron",          ">",  2.1),
    "FoodLowInIron":        ("hasAmountOfIron",          "<",  0.5),
}


def get_foods_by_class(class_name: str, limit: int = 30) -> List[Tuple[str, URIRef]]:
    class_uri = SHFO[class_name]
    seen: set = set()
    foods: List[Tuple[str, URIRef]] = []
    for individual in ontology_graph.subjects(RDF.type, class_uri):
        if individual in seen:
            continue
        seen.add(individual)
        foods.append((get_entity_label(individual), individual))
    return sorted(foods, key=lambda x: x[0])[:limit]


def get_foods_by_nutrient_threshold(
    prop_name: str, operator: str, threshold: float, limit: int = 25
) -> List[Tuple[str, URIRef, float]]:
    """Return (label, uri, value) for Food individuals whose `prop_name` satisfies the threshold."""
    prop_uri = SHFO[prop_name]
    results: List[Tuple[str, URIRef, float]] = []
    for label, food_uri in FOOD_CANDIDATES:
        for _, _, raw_value in ontology_graph.triples((food_uri, prop_uri, None)):
            numeric = parse_numeric_literal(raw_value)
            if numeric is None:
                continue
            if (operator == ">" and numeric > threshold) or (operator == "<" and numeric < threshold):
                results.append((label, food_uri, numeric))
    results.sort(key=lambda x: x[2], reverse=(operator == ">"))
    return results[:limit]


def get_recipes_by_class(class_name: str, limit: int = 20) -> List[Tuple[str, URIRef]]:
    """Return Recipe individuals directly typed as the given food class."""
    class_uri = SHFO[class_name]
    seen: set = set()
    recipes: List[Tuple[str, URIRef]] = []
    for individual in ontology_graph.subjects(RDF.type, class_uri):
        if individual in seen:
            continue
        seen.add(individual)
        if (individual, RDF.type, SHFO.Recipe) in ontology_graph:
            recipes.append((get_entity_label(individual), individual))
    return sorted(recipes, key=lambda x: x[0])[:limit]


def get_recipes_by_nutrient_threshold(
    prop_name: str, operator: str, threshold: float, limit: int = 15
) -> List[Tuple[str, URIRef, float]]:
    """Return (label, uri, value) for Recipe individuals whose `prop_name` satisfies the threshold."""
    prop_uri = SHFO[prop_name]
    results: List[Tuple[str, URIRef, float]] = []
    seen: set = set()
    for recipe in ontology_graph.subjects(RDF.type, SHFO.Recipe):
        if recipe in seen:
            continue
        seen.add(recipe)
        for _, _, raw_value in ontology_graph.triples((recipe, prop_uri, None)):
            numeric = parse_numeric_literal(raw_value)
            if numeric is None:
                continue
            if (operator == ">" and numeric > threshold) or (operator == "<" and numeric < threshold):
                results.append((get_entity_label(recipe), recipe, numeric))
    results.sort(key=lambda x: x[2], reverse=(operator == ">"))
    return results[:limit]


def build_food_class_context(class_name: str, user_profile: Optional[dict] = None) -> str:
    class_uri = SHFO[class_name]
    class_label = get_entity_label(class_uri)

    # Primary path: individuals directly typed as this class
    foods = get_foods_by_class(class_name)
    recipes = get_recipes_by_class(class_name)
    if foods or recipes:
        lines = [
            f"According to the SustainHealthyFood Ontology, the class '{class_label}' "
            f"contains the following instances:",
        ]
        if foods:
            lines.append(f"Foods ({len(foods)}):")
            lines.extend([f"- {label}" for label, _ in foods])
        if recipes:
            lines.append(f"Recipes ({len(recipes)}):")
            lines.extend([f"- {label}" for label, _ in recipes])
        all_labels = [label for label, _ in foods] + [label for label, _ in recipes]
        return "\n".join(lines) + _profile_conflict_note(all_labels, user_profile)

    # Fallback: query nutrient data from all Food and Recipe individuals
    fallback = FOOD_CLASS_NUTRIENT_FALLBACK.get(class_name)
    if fallback:
        prop_name, operator, threshold = fallback
        nutrient_name = NUTRIENT_PROPERTIES.get(prop_name, prop_name)
        unit = NUTRIENT_UNITS.get(nutrient_name, "")
        threshold_str = f"{threshold}{(' ' + unit) if unit else ''}"
        op_phrase = f"more than {threshold_str}" if operator == ">" else f"less than {threshold_str}"

        threshold_foods = get_foods_by_nutrient_threshold(prop_name, operator, threshold)
        threshold_recipes = get_recipes_by_nutrient_threshold(prop_name, operator, threshold)

        if threshold_foods or threshold_recipes:
            lines = [
                f"The ontology class '{class_label}' has no directly assigned instances.",
                f"Based on NEVO nutrient data in the SustainHealthyFood Ontology, "
                f"items containing {op_phrase} of {nutrient_name} per 100 g:",
            ]
            if threshold_foods:
                lines.append(f"Foods:")
                lines.extend([
                    f"- {label}: {value:.1f}{(' ' + unit) if unit else ''}"
                    for label, _, value in threshold_foods
                ])
            if threshold_recipes:
                lines.append(f"Recipes:")
                lines.extend([
                    f"- {label}: {value:.1f}{(' ' + unit) if unit else ''}"
                    for label, _, value in threshold_recipes
                ])
            threshold_labels = [label for label, _, _ in threshold_foods] + [label for label, _, _ in threshold_recipes]
            return "\n".join(lines) + _profile_conflict_note(threshold_labels, user_profile)

    return (
        f"The ontology class '{class_label}' has no directly assigned instances "
        f"and no nutrient-threshold data is available for this category."
    )


FOOD_CLASS_INTERSECTIONS: List[Tuple[List[str], List[str]]] = [
    (["high in protein", "low in fat"],  ["FoodHighInProtein", "FoodLowInFat"]),
    (["high protein", "low fat"],        ["FoodHighInProtein", "FoodLowInFat"]),
    (["high protein", "low in fat"],     ["FoodHighInProtein", "FoodLowInFat"]),
    (["high in protein", "low fat"],     ["FoodHighInProtein", "FoodLowInFat"]),
    (["high in fiber", "low in fat"],    ["FoodHighInFiber",   "FoodLowInFat"]),
    (["high in carbs", "low in fat"],    ["FoodHighInCarbs",   "FoodLowInFat"]),
]


def detect_food_class_intersection_query(question: str) -> Optional[List[str]]:
    """Return list of class names when question combines multiple food class conditions, else None."""
    q = question.lower()
    for keywords, classes in FOOD_CLASS_INTERSECTIONS:
        if all(kw in q for kw in keywords):
            return classes
    return None


def get_foods_by_class_intersection(class_names: List[str], limit: int = 25) -> List[Tuple[str, URIRef]]:
    """Return foods that are instances of ALL given ontology classes."""
    sets = [set(ontology_graph.subjects(RDF.type, SHFO[n])) for n in class_names]
    intersection = sets[0].intersection(*sets[1:])
    foods = [(get_entity_label(uri), uri) for uri in intersection]
    return sorted(foods, key=lambda x: x[0])[:limit]


def build_food_class_intersection_context(class_names: List[str]) -> str:
    """Build ontology context for a multi-class intersection (e.g. high protein AND low fat)."""
    class_labels = [get_entity_label(SHFO[n]) for n in class_names]
    description = " and ".join(class_labels)

    def _enrich(uri) -> Tuple[Optional[float], Optional[float]]:
        protein = next(
            (parse_numeric_literal(v) for _, _, v in ontology_graph.triples((uri, SHFO["hasAmountOfProtein"], None))),
            None,
        )
        fat = next(
            (parse_numeric_literal(v) for _, _, v in ontology_graph.triples((uri, SHFO["hasAmountOfFat"], None))),
            None,
        )
        return protein, fat

    def _food_groups(uri) -> List[str]:
        return get_food_groups(uri) or ["Other"]

    def _format_entry(label, uri, protein, fat) -> str:
        groups = _food_groups(uri)
        parts = []
        if protein is not None:
            parts.append(f"protein: {protein:.1f}g")
        if fat is not None:
            parts.append(f"fat: {fat:.1f}g")
        nutrient_str = f" ({', '.join(parts)})" if parts else ""
        group_str = f" [food group: {', '.join(groups)}]"
        return f"- {label}{nutrient_str}{group_str}"

    # Primary: foods typed as ALL classes in the ontology
    foods = get_foods_by_class_intersection(class_names)
    if foods:
        lines = [
            f"According to the SustainHealthyFood Ontology, foods classified as both {description} (per 100 g):",
        ]
        for label, uri in foods:
            protein, fat = _enrich(uri)
            lines.append(_format_entry(label, uri, protein, fat))
        return "\n".join(lines)

    # Fallback: numeric threshold intersection across all Food individuals
    uri_sets = []
    for name in class_names:
        fb = FOOD_CLASS_NUTRIENT_FALLBACK.get(name)
        if fb:
            prop_name, operator, threshold = fb
            matched = get_foods_by_nutrient_threshold(prop_name, operator, threshold, limit=300)
            uri_sets.append({uri for _, uri, _ in matched})

    if len(uri_sets) == len(class_names):
        intersection_uris = uri_sets[0].intersection(*uri_sets[1:])
        candidates = []
        for uri in intersection_uris:
            protein, fat = _enrich(uri)
            candidates.append((get_entity_label(uri), uri, protein or 0.0, fat or 0.0))
        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = candidates[:25]

        if candidates:
            lines = [
                f"Based on NEVO nutrient data in the SustainHealthyFood Ontology,",
                f"foods that are {description} (per 100 g):",
            ]
            for label, uri, prot, fat in candidates:
                lines.append(_format_entry(label, uri, prot, fat))
            return "\n".join(lines)

    return f"No foods found in the ontology matching: {description}."


def calculate_bmi(weight: float, height: float) -> float:
    height_m = height / 100
    return round(weight / (height_m ** 2), 1)


def calculate_body_fat_percentage(bmi: float, age: int, sex: int, decimal_places: int = 2) -> float:
    if not (isinstance(bmi, (int, float)) and bmi > 0):
        raise ValueError("BMI must be a positive number.")
    if not (isinstance(age, int) and age > 0):
        raise ValueError("Age must be a positive integer.")
    if sex not in [0, 1]:
        raise ValueError("sex must be 1 for male or 0 for female.")
    body_fat = (1.20 * bmi) + (0.23 * age) - (10.8 * sex) - 5.4
    return round(body_fat, decimal_places)


def calculate_total_calories(
    weight: float, height: float, age: int, sex: int, activity_level: str,
    method: str = "Harris-Benedict",
) -> float:
    if method == "Harris-Benedict" and sex == 1:
        rmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    elif method == "Harris-Benedict" and sex == 0:
        rmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    else:
        raise ValueError("Invalid method or sex input.")
    activity_factors = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725,
        "Extra Active": 1.9,
    }
    return round(rmr * activity_factors.get(activity_level, 1.2), 2)


def calculate_efsa_macros(total_calories: float, weight_kg: float) -> dict:
    kcal_per_g = {"carbs": 4, "protein": 4, "fat": 9}
    carb_pct_range = (0.45, 0.60)
    fat_pct_range = (0.20, 0.35)
    protein_range = (0.83, 1.5)
    carbs_min = (carb_pct_range[0] * total_calories) / kcal_per_g["carbs"]
    carbs_max = (carb_pct_range[1] * total_calories) / kcal_per_g["carbs"]
    fat_min = (fat_pct_range[0] * total_calories) / kcal_per_g["fat"]
    fat_max = (fat_pct_range[1] * total_calories) / kcal_per_g["fat"]
    protein_min = protein_range[0] * weight_kg
    protein_max = protein_range[1] * weight_kg
    return {
        "Carbohydrates (g)": (round(carbs_min, 1), round(carbs_max, 1)),
        "Fats (g)": (round(fat_min, 1), round(fat_max, 1)),
        "Protein (g)": (round(protein_min, 1), round(protein_max, 1)),
    }


_ACTIVITY_LEVEL_MAP = {
    "Sedentary": "Sedentary",
    "Moderately Active": "Moderately Active",
    "Active": "Very Active",
    "Athlete": "Extra Active",
}


def compute_daily_needs(profile: dict) -> Optional[dict]:
    sex_int = 1 if profile["sex"] == "Male" else (0 if profile["sex"] == "Female" else None)
    if sex_int is None or profile["age"] <= 0 or profile["weight_kg"] <= 0 or profile["height_cm"] <= 0:
        return None
    try:
        bmi = calculate_bmi(profile["weight_kg"], profile["height_cm"])
        body_fat = calculate_body_fat_percentage(bmi, int(profile["age"]), sex_int)
        activity = _ACTIVITY_LEVEL_MAP.get(profile["activity_level"], "Sedentary")
        total_calories = calculate_total_calories(
            profile["weight_kg"], profile["height_cm"], int(profile["age"]), sex_int, activity
        )
        macros = calculate_efsa_macros(total_calories, profile["weight_kg"])
        return {
            "bmi": bmi,
            "body_fat": body_fat,
            "total_calories": total_calories,
            "macros": macros,
        }
    except (ValueError, ZeroDivisionError):
        return None


def build_profile_summary_text(user_profile: dict) -> str:
    """Plain-text profile summary fed to the LLM prompts. Pulled out of the
    sidebar module-level code so both the Streamlit UI and non-UI callers
    (e.g. evaluation scripts) build it identically from a given profile dict."""
    concern_text = (
        f"The user's main concerns are: {user_profile['user_concerns']}"
        if user_profile["user_concerns"] else "The user did not indicate any specific concerns."
    )

    profile_summary = (
        f"The user is a {user_profile['age']}-year-old {user_profile['sex'].lower()} "
        f"in the {user_profile['life_stage']} stage. "
        f"Weight: {user_profile['weight_kg']} kg. Height: {user_profile['height_cm']} cm. "
        f"Goal: {user_profile['goal']}. Diet preference: {user_profile['diet_type']}. "
        f"Allergies/intolerances: {', '.join(user_profile['allergies']) if user_profile['allergies'] else 'none'}. "
        f"Activity level: {user_profile['activity_level']}. "
        f"Cultural background: {user_profile['cultural_background'] or 'not specified'}. "
        f"Occupation: {user_profile['occupation'] or 'not specified'}. "
        f"{concern_text}. "
        f"Stress level: {user_profile['stress_level']}. "
        f"Sleep quality: {user_profile['sleep_quality']}. "
        f"Motivation level: {user_profile['motivation_level']}. "
        f"Mental persona: {user_profile['mental_persona']}. "
        f"Appetite pattern: {user_profile['appetite_pattern']}. "
        f"Support goal: {user_profile['support_goal']}."
    )

    daily_needs = compute_daily_needs(user_profile)
    if daily_needs:
        profile_summary += (
            f" Calculated daily energy need: {daily_needs['total_calories']} kcal/day"
            f" (Harris-Benedict + activity factor). "
            f"BMI: {daily_needs['bmi']}. Estimated body fat: {daily_needs['body_fat']}%. "
            f"EFSA daily macronutrient targets: "
            f"Carbohydrates {daily_needs['macros']['Carbohydrates (g)'][0]}–{daily_needs['macros']['Carbohydrates (g)'][1]} g, "
            f"Protein {daily_needs['macros']['Protein (g)'][0]}–{daily_needs['macros']['Protein (g)'][1]} g, "
            f"Fat {daily_needs['macros']['Fats (g)'][0]}–{daily_needs['macros']['Fats (g)'][1]} g."
        )
    return profile_summary


@st.cache_resource
def init_general_retriever():
    vectorstore = Chroma(
        persist_directory=GENERAL_CHROMA_DIR,
        collection_name=GENERAL_CHROMA_COLLECTION,
        embedding_function=embedding_model,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 8})

@st.cache_resource
def init_health_retriever():
    vectorstore = Chroma(
        persist_directory=HEALTH_CHROMA_DIR,
        collection_name=HEALTH_CHROMA_COLLECTION,
        embedding_function=embedding_model,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 8})


@st.cache_resource
def init_mental_state_vectorstore():
    """Raw Chroma handle (not a retriever) for the mental state chart —
    kept unwrapped because it needs similarity_search with metadata filters
    (state / from_state / type), which a plain retriever can't do per-call."""
    return Chroma(
        persist_directory=MENTAL_STATE_CHROMA_DIR,
        collection_name=MENTAL_STATE_COLLECTION,
        embedding_function=embedding_model,
    )


def build_mental_state_chart_context(chart_state: str, question: str) -> str:
    """Retrieve state-profile, transition, and wellbeing-evidence docs from the
    mental state chart (mental_state_model.jsonl + wellbeing PDFs, embedded into
    the mental_state_chroma_db Chroma collection) for eating-pattern / nutrient-gap
    context tied to a resolved mood state."""
    vs = init_mental_state_vectorstore()
    try:
        state_profile_docs = vs.similarity_search(
            f"{chart_state} dietary advice", k=2,
            filter={"state": {"$eq": chart_state}},
        )
        transition_docs = vs.similarity_search(
            f"{chart_state} recovery", k=2,
            filter={"from_state": {"$eq": chart_state}},
        )
        evidence_docs = vs.similarity_search(
            f"{chart_state} {question}", k=3,
            filter={"type": {"$eq": "evidence_pdf"}},
        )
    except Exception:
        # Fallback: plain semantic search if metadata filters unsupported
        state_profile_docs = vs.similarity_search(f"{chart_state} state profile dietary advice", k=3)
        transition_docs = vs.similarity_search(f"{chart_state} recovery dietary intervention", k=2)
        evidence_docs = vs.similarity_search(f"{chart_state} {question}", k=3)

    chart_text = "\n\n".join(d.page_content for d in (state_profile_docs + transition_docs)[:3])
    evidence_text = "\n\n".join(d.page_content for d in evidence_docs[:2])

    lines = [
        "MENTAL STATE CHART CONTEXT "
        f"(entry signals, ongoing eating pattern, and nutrient gaps typical of {chart_state}):",
        chart_text or "No mental state chart entry found for this state.",
    ]
    if evidence_text:
        lines.append("\nSupporting wellbeing/nutrition research evidence:")
        lines.append(evidence_text)
    return "\n".join(lines)


# Keywords that indicate a question is about mood/mental state, distinct from the
# narrower wellbeing_terms list in is_general_cooking_question (which exists to
# route AWAY from general_cooking, not to gate mental-chart retrieval).
_MOOD_QUESTION_KEYWORDS = [
    "mood", "sleep", "anxiety", "anxious", "appetite", "stressed", "stress",
    "depression", "depressed", "motivation", "energy", "emotional",
    "wellbeing", "well-being", "burnout", "recovering", "recovery",
    "mental state", "current state", "my state", "how i feel", "how i'm feeling",
    "mental health",
]


def is_mood_related_question(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in _MOOD_QUESTION_KEYWORDS)


# Keywords that mean the question explicitly names a specific condition
# ("manage my stress") rather than asking generically about "my mood"/"how
# I'm feeling" with nothing specific named. Used to decide whether a mood
# question should be grounded in the user's own resolved mental state
# (generic phrasing) or trusted as naming its own target condition (specific
# phrasing) — see _is_generic_mood_question and the profile_reasoning branch.
_SPECIFIC_STATE_KEYWORDS = (
    "stress", "anxiety", "anxious", "depress", "energy", "low mood",
    "burnout", "motivation",
)


def _is_generic_mood_question(question: str) -> bool:
    """True for a mood-related question that names no specific condition —
    e.g. 'what food helps elevate my mood?' — as opposed to one that does,
    e.g. 'what food helps my stress?'. Generic questions like this used to be
    routed to health_recommendation, which has no awareness of the user's
    own profile at all: it asks the router LLM to guess a specific ontology
    entity from the bare question text alone, with no way to know the user's
    actual resolved state. E.g. a Stressed user asking this got an answer
    grounded in "Depressed" — whatever the LLM guessed from "elevate my
    mood" in isolation — with zero connection to their real state. A
    specifically-named condition ("my stress") is left alone; that one is
    reliably resolved from the question text itself, same as before.
    """
    if not is_mood_related_question(question):
        return False
    q = question.lower()
    return not any(k in q for k in _SPECIFIC_STATE_KEYWORDS)


# Display label + colour per resolved mood state — same mapping as the sidebar
# state indicator in retrieval_Food_personas_wellbeing_ontology.py.
STATE_DISPLAY: Dict[str, Tuple[str, str]] = {
    "Calm":        ("Calm",        "#16a34a"),
    "Stressed":    ("Stressed",    "#ea580c"),
    "Anxious":     ("Anxious",     "#ca8a04"),
    "Depressed":   ("Depressed",   "#7c3aed"),
    "Energized":   ("Energized",   "#0284c7"),
    "Low mood":    ("Low mood",    "#6b7280"),
    "Mood swings": ("Mood swings", "#d97706"),
    "Recovering":  ("Recovering",  "#0891b2"),
}


def resolve_mental_state_from_profile(profile: dict) -> str:
    """Mood resolution. Self-reported text (profile['self_reported_mood'])
    takes priority for Depressed and Anxious — those two states are reachable
    ONLY through what the user actually types there now, not inferred from
    the stress/sleep/motivation dropdowns. Everything else is still rule-
    based over those dropdowns, same approach as before. "Recovering" and
    "Mood swings" have been removed as possible states entirely.
    "Not defined"/empty values are treated as neutral — skip those rules.
    """
    self_report = (profile.get("self_reported_mood") or "").lower()
    if "depress" in self_report:
        return "Depressed"
    if "anxi" in self_report:   # catches "anxious" and "anxiety"
        return "Anxious"

    stress     = profile.get("stress_level", "Not defined")
    motivation = profile.get("motivation_level", "Not defined")
    appetite   = profile.get("appetite_pattern", "Not defined")

    # sleep_quality is no longer read by any rule below. "Low mood" used to
    # exclude sleep == "Poor" here, because that combination was claimed by
    # the rule-based "Depressed" branch (now removed, replaced by self-
    # report above). Without dropping that exclusion, a Low-motivation +
    # poor-sleep profile with no self-reported mood would fall all the way
    # through to the "Calm" default, which is clearly wrong for that profile.
    if motivation == "Low" and stress in ("Low", "Moderate"):
        return "Low mood"
    if stress == "High" or "sugar / caffeine" in appetite:
        return "Stressed"
    if motivation == "High" and stress == "Low":
        return "Energized"
    return "Calm"


FEEDBACK_LOG_FILE = "feedback_log.json"


def build_chat_txt(history: list, profile: dict) -> str:
    lines = [
        "=" * 60,
        "  Dietary Guidelines Q&A System — Conversation Export",
        f"  Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        "USER PROFILE",
        "-" * 40,
        f"Age: {profile.get('age')}  |  Sex: {profile.get('sex')}  |  Life stage: {profile.get('life_stage')}",
        f"Weight: {profile.get('weight_kg')} kg  |  Height: {profile.get('height_cm')} cm",
        f"Goal: {profile.get('goal')}  |  Activity: {profile.get('activity_level')}",
        f"Diet: {profile.get('diet_type')}  |  Allergies: {', '.join(profile.get('allergies', [])) or 'none'}",
        f"Stress: {profile.get('stress_level')}  |  Sleep: {profile.get('sleep_quality')}",
        "",
        "=" * 60,
        "  CONVERSATION",
        "=" * 60,
        "",
    ]
    for i, (q, a) in enumerate(history, 1):
        lines += [f"[{i}] You:", f"    {q}", "", "    Assistant:", ""]
        for line in a.splitlines():
            lines.append(f"    {line}")
        lines += ["", "-" * 60, ""]
    return "\n".join(lines)


def build_chat_html(history: list, profile: dict) -> str:
    def esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows = "".join(
        f"""
        <div class="turn">
          <div class="bubble user"><strong>You</strong><p>{esc(q)}</p></div>
          <div class="bubble assistant"><strong>Assistant</strong><p>{"<br>".join(esc(line) for line in a.splitlines())}</p></div>
        </div>"""
        for q, a in history
    )
    allergy_str = ", ".join(profile.get("allergies", [])) or "none"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Dietary Q&A Export</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 820px; margin: 40px auto; color: #1f2937; }}
  h1 {{ font-size: 1.4rem; color: #15803d; }}
  .meta {{ font-size: 0.85rem; color: #6b7280; margin-bottom: 1.5rem; }}
  .profile {{ background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px;
              padding: 0.8rem 1.2rem; margin-bottom: 1.5rem; font-size: 0.9rem; }}
  .turn {{ margin-bottom: 1.2rem; }}
  .bubble {{ border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.4rem; }}
  .user {{ background: #e0f2fe; }}
  .assistant {{ background: #f9fafb; border: 1px solid #e5e7eb; }}
  strong {{ font-size: 0.8rem; color: #6b7280; display: block; margin-bottom: 4px; }}
  @media print {{ body {{ margin: 20px; }} }}
</style></head><body>
<h1>🥗 Dietary Guidelines Q&A — Conversation Export</h1>
<p class="meta">Exported: {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;|&nbsp; {len(history)} exchange(s)</p>
<div class="profile">
  <strong>User Profile</strong>
  Age {profile.get('age')} &nbsp;|&nbsp; {profile.get('sex')} &nbsp;|&nbsp; {profile.get('life_stage')} &nbsp;|&nbsp;
  {profile.get('weight_kg')} kg / {profile.get('height_cm')} cm &nbsp;|&nbsp;
  Goal: {profile.get('goal')} &nbsp;|&nbsp; Activity: {profile.get('activity_level')} &nbsp;|&nbsp;
  Diet: {profile.get('diet_type')} &nbsp;|&nbsp; Allergies: {allergy_str}
</div>
{rows}
<p class="meta">— End of conversation. To save as PDF: File → Print → Save as PDF —</p>
</body></html>"""


def save_feedback(rating: str, question: str, answer: str, routing_path: str = "") -> None:
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rating": rating,
        "routing_path": routing_path,
        "question": question,
        "answer_snippet": answer[:300],
    }
    existing = []
    if os.path.exists(FEEDBACK_LOG_FILE):
        try:
            with open(FEEDBACK_LOG_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []
    existing.append(entry)
    with open(FEEDBACK_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def load_feedback_log() -> List[dict]:
    if not os.path.exists(FEEDBACK_LOG_FILE):
        return []
    try:
        with open(FEEDBACK_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False
if "state_history" not in st.session_state:
    st.session_state.state_history = []
if "current_state" not in st.session_state:
    st.session_state.current_state = "Calm"

if "last_recipe_matches" not in st.session_state:
    st.session_state.last_recipe_matches = []
if "last_recipe_index" not in st.session_state:
    st.session_state.last_recipe_index = 0
if "last_recipe_query_type" not in st.session_state:
    st.session_state.last_recipe_query_type = ""

if "last_food_matches" not in st.session_state:
    st.session_state.last_food_matches = []
if "last_food_index" not in st.session_state:
    st.session_state.last_food_index = 0
if "response_traces" not in st.session_state:
    st.session_state.response_traces = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False
if "retry_question" not in st.session_state:
    st.session_state.retry_question = None

st.markdown(
    """
    <style>
    .stApp { background: #f6f8fb; }
    .main-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.4rem; }
    .subtitle { font-size: 0.95rem; color: #6b7280; margin-bottom: 1.5rem; }
    .card {
        background-color: #ffffff; padding: 1.2rem 1.4rem; border-radius: 0.9rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06); margin-bottom: 1rem; border: 1px solid #e5e7eb;
    }
    .chat-user { background: #e0f2fe; border-radius: 1rem; padding: 0.8rem 1rem; margin-bottom: 0.5rem; }
    .chat-assistant {
        background: #f9fafb; border-radius: 1rem; padding: 0.8rem 1rem; margin-bottom: 1rem; border: 1px solid #e5e7eb;
    }
    .question-chip button {
        border-radius: 999px !important; padding: 0.25rem 0.8rem !important;
        font-size: 0.85rem !important; border: 1px solid #d1d5db !important;
    }
    .stCheckbox > label { font-weight: 500; }
    .feedback-row button {
        padding: 0.15rem 0.55rem !important;
        font-size: 0.85rem !important;
        border-radius: 999px !important;
        border: 1px solid #d1d5db !important;
        background: transparent !important;
        color: #6b7280 !important;
        min-height: 0 !important;
    }
    .feedback-row button:hover { background: #f3f4f6 !important; }
    div[data-testid="stCheckbox"] {
        background: #f0fdf4;
        border: 2px solid #16a34a;
        border-radius: 0.9rem;
        padding: 1rem 1.4rem;
        margin: 1rem 0;
    }
    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stCheckbox"] span {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #15803d !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-title">🥗 Dietary Guidelines Q&A System</div>
    <div class="subtitle">
        Personalized guidance grounded in the SustainHealthyFood ontology, your profile, and your lifestyle.
    </div>
    """,
    unsafe_allow_html=True,
)

# Required by the NEVO 2023 conditions of use: output that involves
# nutritional calculations must acknowledge NEVO.
st.caption("Based on data from NEVO online version 2023/8.0, RIVM, Bilthoven and other data sources.")

with st.expander("ℹ️ How to use this system", expanded=False):
    st.markdown("""
### Welcome! Here's how to get the most out of this system.

**This system combines a food ontology, namely the SustainHealthyFood Ontology, with an AI assistant to give you personalised dietary guidance.**

---

#### 🚀 Getting started — 3 simple steps

1. **Fill in your profile** in the sidebar on the left — age, weight, goal, diet type, allergies, stress level, and more.
2. **Confirm your profile** by ticking the *"Yes, this information is correct"* box below your summary.
3. **Ask a question** — type it in the chat box or click one of the suggested questions.

---

#### 💬 What you can ask

| Topic | Example questions |
|---|---|
| 👤 **Your nutrition targets** | *"What are my daily calorie and macro targets?"* |
| 🍽️ **Recipes & meal planning** | *"Help me meal prep for the week with 3 recipes"* |
| 🏥 **Health conditions** | *"What foods are good for iron deficiency?"* |
| 🧠 **Mental health & energy** | *"Which foods help with stress and better sleep?"* |
| 🏋️ **Workout nutrition** | *"What should I eat before and after training?"* |
| 🌿 **Allergies & substitutes** | *"What are good dairy-free alternatives?"* |
| 🔬 **Nutrient lookups** | *"What are the nutrients in avocado per 100g?"* |
| 📖 **Concept definitions** | *"What is omega-3 and why does it matter?"* |
| 🌱 **Environmental sustainability impact** | *"What is the environmental sustainability score of banana?"* |

---

#### 🧬 About the data

The SustainHealthyFood Ontology is operationalized as a knowledge graph: its food, recipe, and nutritional information is populated with data (triples) sourced from the [NEVO dataset](https://www.rivm.nl/en/dutch-food-composition-database) (Dutch Food Composition Database). Because food and recipe names are carried over from NEVO's own naming conventions rather than rewritten, you'll often see abbreviations in suggestions — e.g. "raw" / "prepared" for preparation state, or "w" / "wo" for "with" / "without" an ingredient.

Environmental sustainability data (GHG/CO2 emissions, eutrophying emissions, land use) are sourced from Our World in Data's [Environmental Impacts of Food dataset](https://ourworldindata.org/environmental-impacts-of-food), matched onto NEVO food names. That dataset covers around 40 broad commodity categories (e.g. "Beef", "Milk", "Rice") rather than specific prepared foods, so only a subset of foods in the ontology have this data — if it's missing for the food you ask about, the assistant will say so rather than estimate a number.

---

#### 💡 Tips for best results

- **Be specific** — the more detail in your question, the more tailored the answer.
- **Use your profile** — questions like *"given my goal"* or *"based on my activity level"* trigger personalised responses.
- **Explore the suggested questions** below — they are grouped by topic to help you discover what the system can do.
- **Check "🔍 How was this response generated?"** below any answer — it shows the routing path, the ontology entity matched, and the exact context passed to the AI, so you can verify how the answer was produced.
- **Answers are AI-generated from the context above** — the assistant is grounded in the ontology, but can occasionally drift from it. Treat health-related suggestions as a starting point, not medical advice.
""")

with st.sidebar:
    # ── Sticky profile completeness bar ────────────────────────────────────
    # Rendered as a single raw HTML block so position:sticky actually works.
    # Streamlit widget wrappers have intermediate divs with overflow:hidden
    # that silently break sticky — bypassing them with pure HTML fixes it.
    # Values are read from session_state (saved at end of sidebar) so they
    # always reflect the last submitted profile on every re-run.
    st.markdown("""
    <style>
    /* Remove overflow:hidden on sidebar containers so sticky can propagate */
    div[data-testid="stSidebarContent"],
    div[data-testid="stSidebarContent"] > div {
        overflow: visible !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _prev = st.session_state.get("user_profile", {})
    _cf = {
        "Age":              _prev.get("age", 0) > 0,
        "Weight":           _prev.get("weight_kg", 0) > 0,
        "Goal":             _prev.get("goal", "Not defined") != "Not defined",
        "Activity level":   _prev.get("activity_level", "Not defined") != "Not defined",
        "Diet preference":  _prev.get("diet_type", "None") != "None",
        "Current concerns": bool(_prev.get("user_concerns", "")),
        "Stress level":     _prev.get("stress_level", "Not defined") != "Not defined",
        "Sleep quality":    _prev.get("sleep_quality", "Not defined") != "Not defined",
        "Appetite pattern": _prev.get("appetite_pattern", "Not defined") != "Not defined",
        "Support goal":     _prev.get("support_goal", "Not defined") != "Not defined",
    }
    _filled  = sum(_cf.values())
    _total   = len(_cf)
    _pct     = int(_filled / _total * 100)
    _bar_color, _badge_bg, _status_text = (
        ("#2ecc71", "#d4edda", "✅ Profile complete — answers will be fully personalised!")
        if _pct == 100 else
        ("#f39c12", "#fff3cd", f"🟡 {_pct}% — almost there, keep going!")
        if _pct >= 70 else
        ("#e67e22", "#fde8d8", f"🟠 {_pct}% — more detail means better answers")
        if _pct >= 40 else
        ("#e74c3c", "#fde8e8", f"🔴 {_pct}% — open the sections below to get started")
    )
    _missing = [k for k, v in _cf.items() if not v]
    _done    = [k for k, v in _cf.items() if v]

    _pill = lambda text, done: (
        f'<span style="display:inline-block;margin:2px 3px 2px 0;padding:2px 8px;'
        f'border-radius:99px;font-size:0.72em;font-weight:500;'
        f'background:{"#d4edda" if done else "#f8d7da"};'
        f'color:{"#155724" if done else "#721c24"};">'
        f'{"✓" if done else "○"} {text}</span>'
    )
    _all_pills = "".join(_pill(k, True) for k in _done) + "".join(_pill(k, False) for k in _missing)

    st.markdown(f"""
    <div style="
        position: sticky;
        top: 0;
        z-index: 9999;
        background-color: rgb(240, 242, 246);
        padding: 14px 6px 12px 6px;
        border-bottom: 1px solid rgba(49,51,63,0.15);
        margin-bottom: 14px;
    ">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;">
            <span style="font-size:0.95em;font-weight:700;color:#31333f;">📋 Profile Completeness</span>
            <span style="font-size:0.82em;font-weight:600;color:{_bar_color};">{_filled}/{_total} fields</span>
        </div>
        <div style="background:#e0e0e0;border-radius:99px;height:11px;overflow:hidden;margin-bottom:7px;">
            <div style="width:{_pct}%;background:linear-gradient(90deg,{_bar_color}cc,{_bar_color});
                        height:100%;border-radius:99px;transition:width 0.4s ease;"></div>
        </div>
        <div style="background:{_badge_bg};border-radius:6px;padding:5px 8px;margin-bottom:8px;">
            <span style="font-size:0.8em;">{_status_text}</span>
        </div>
        <div style="line-height:1.8;">{_all_pills}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.header("👤 User Profile")

    user_profile = {}

    with st.expander("🧬 Basics", expanded=True):
        user_profile["age"] = st.number_input("Age", min_value=0, max_value=120, value=30,
            help="Your age in years. Used to calculate your daily calorie and nutrient needs.")
        user_profile["sex"] = st.selectbox("Sex", ["Female", "Male"],
            help="Your biological sex. Used in calorie and body composition calculations (Harris-Benedict equation).")
        user_profile["life_stage"] = st.selectbox("Life Stage", ["General Adult", "Pregnant", "Lactating", "Pedriatic"],
            help="Your current life stage. Pregnant and lactating women have significantly different nutritional requirements. Pedriatic refers to a person under 18 years old whose nutritional needs vary by age and growth stage.")
        user_profile["weight_kg"] = st.number_input("Weight (kg)", min_value=0.0, value=55.0, max_value=150.0,
            help="Your current body weight in kilograms. Used to calculate calorie needs and protein targets.")
        user_profile["height_cm"] = st.number_input("Height (cm)", min_value=10.0, value=165.0, max_value=215.0,
            help="Your height in centimetres. Used alongside weight to calculate your BMI and daily energy needs.")

    with st.expander("🎯 Goals & Lifestyle", expanded=True):
        user_profile["goal"] = st.selectbox("Goal", ["Not defined", "Maintain weight", "Lose weight", "Gain muscle"],
            help="What you want to achieve with your diet. This shapes the type of nutritional advice and food suggestions you receive.")
        user_profile["activity_level"] = st.selectbox(
            "Activity Level", ["Not defined", "Sedentary", "Moderately Active", "Active", "Athlete"],
            help=(
                "How physically active you are on a typical day — affects your total daily calorie needs.\n\n"
                "• Sedentary: desk job, little or no exercise\n"
                "• Moderately Active: light exercise 3–5 days/week\n"
                "• Active: intense exercise 6–7 days/week\n"
                "• Athlete: very intense daily training"
            )
        )
        user_profile["diet_type"] = st.selectbox("Diet Preference", ["None", "Vegetarian", "Vegan", "Halal", "Keto"],
            help="Any dietary pattern or restriction you follow. The system will tailor food and recipe recommendations accordingly.")
        user_profile["allergies"] = st.multiselect("Allergies/Intolerances", ["Nuts", "Dairy", "Soy", "Gluten", "Eggs"],
            help="Foods you cannot eat or need to avoid. The system will flag or exclude these from recommendations.")

    with st.expander("🌍 Context", expanded=False):
        user_profile["cultural_background"] = st.text_input("Cultural background",
            help="Your cultural or ethnic background. Helps the system suggest culturally familiar and appropriate foods.")
        user_profile["occupation"] = st.text_input("Occupation",
            help="Your job or daily role. Helps contextualise your energy demands — e.g. a manual worker needs more calories than a desk worker.")
        user_profile["user_concerns"] = st.text_area("Current concerns",
            placeholder="e.g., I'm worried about my iron levels...",
            help="Any specific health, dietary, or lifestyle concerns you want to address. The system will take these into account when answering.")

    with st.expander("🧠 Mood & Well-being", expanded=False):
        user_profile["self_reported_mood"] = st.text_input(
            "How are you feeling right now?",
            value="",
            placeholder="e.g. feeling pretty anxious today, a bit depressed lately",
            help="Describe how are you feeling"
        )
        user_profile["stress_level"] = st.selectbox("Stress level", ["Not defined", "Low", "Moderate", "High"],
            help="Your current level of psychological stress. High stress increases cortisol, affects digestion, and changes nutrient needs.")
        user_profile["sleep_quality"] = st.selectbox("Sleep quality", ["Not defined", "Good", "Fair", "Poor"],
            help="How well you are sleeping. Poor sleep disrupts appetite hormones (ghrelin/leptin) and lowers energy levels.")
        user_profile["motivation_level"] = st.selectbox("Energy & motivation", ["Not defined", "High", "Moderate", "Low"],
            help="Your current energy and motivation level. Helps the system suggest realistic meal plans and avoid overwhelming advice.")
        user_profile["mental_persona"] = st.text_input(
            "Relationship with food",
            value="None",
            placeholder="e.g. I stress-eat, I skip meals when busy, I meal prep every Sunday",
            help="Describe how you typically relate to food — any habits, patterns, or tendencies around eating."
        )
        user_profile["appetite_pattern"] = st.selectbox(
            "Typical appetite pattern",
            [
                "Not defined",
                "Normal / regular meals",
                "Craving sugar / caffeine",
                "Low appetite / skipping meals",
                "Only small/light snacks"
            ],
            help="Your usual appetite and eating pattern. Helps the system suggest appropriate meal sizes, snacks, or energy-boosting foods."
        )
        user_profile["support_goal"] = st.selectbox(
            "Current focus / goal",
            [
                "Not defined",
                "Feel calmer / reduce stress",
                "Steady my mood / reduce anxiety",
                "Improve energy / fight fatigue",
                "Support recovery after training",
                "Just maintain my good routine"
            ],
            help="What you are most trying to address right now. The system will tailor its focus and tone of advice accordingly."
        )

    st.session_state.user_profile = user_profile

    resolved_state = resolve_mental_state_from_profile(user_profile)
    if not st.session_state.state_history or st.session_state.state_history[-1] != resolved_state:
        st.session_state.state_history.append(resolved_state)
    st.session_state.current_state = resolved_state

    st.divider()
    _state_label, _state_color = STATE_DISPLAY[resolved_state]
    st.markdown(
        f"**Current mood state:** "
        f"<span style='color:{_state_color};font-weight:700'>{_state_label}</span>",
        unsafe_allow_html=True,
    )
    if len(st.session_state.state_history) > 1:
        with st.expander("State history this session"):
            for _s in st.session_state.state_history:
                _lbl, _clr = STATE_DISPLAY[_s]
                st.markdown(f"- <span style='color:{_clr}'>{_lbl}</span>", unsafe_allow_html=True)

    st.divider()
    st.subheader("📊 Feedback Dashboard")
    log = load_feedback_log()
    if not log:
        st.caption("No feedback recorded yet.")
    else:
        total = len(log)
        up = sum(1 for e in log if e["rating"] == "up")
        down = total - up
        st.metric("Total ratings", total)
        col_u, col_d = st.columns(2)
        col_u.metric("👍 Positive", up)
        col_d.metric("👎 Negative", down)

        with st.expander("📋 View all feedback"):
            for entry in reversed(log):
                icon = "👍" if entry["rating"] == "up" else "👎"
                st.markdown(
                    f"**{icon} {entry['timestamp']}** · `{entry.get('routing_path', '—')}`\n\n"
                    f"**Q:** {entry['question']}\n\n"
                    f"**A:** {entry['answer_snippet']}{'…' if len(entry['answer_snippet']) == 300 else ''}"
                )
                st.divider()

        csv_rows = [["timestamp", "rating", "routing_path", "question", "answer_snippet"]]
        csv_rows += [
            [
                e["timestamp"],
                "Positive" if e["rating"] == "up" else "Negative",
                e.get("routing_path", ""),
                e["question"],
                e["answer_snippet"],
            ]
            for e in log
        ]
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerows(csv_rows)
        st.download_button(
            label="⬇️ Download as CSV",
            data=buf.getvalue(),
            file_name="feedback_log.csv",
            mime="text/csv",
        )

daily_needs = compute_daily_needs(user_profile)
st.session_state.daily_needs = daily_needs

profile_summary = build_profile_summary_text(user_profile)

_profile_summary_html = f"""
<ul>
  <li><b>Age:</b> {user_profile['age']}</li>
  <li><b>Sex:</b> {user_profile['sex']}</li>
  <li><b>Life stage:</b> {user_profile['life_stage']}</li>
  <li><b>Weight / Height:</b> {user_profile['weight_kg']} kg, {user_profile['height_cm']} cm</li>
  <li><b>Goal:</b> {user_profile['goal']}</li>
  <li><b>Diet preference:</b> {user_profile['diet_type']}</li>
  <li><b>Allergies / intolerances:</b> {', '.join(user_profile['allergies']) if user_profile['allergies'] else 'none'}</li>
  <li><b>Activity level:</b> {user_profile['activity_level']}</li>
  <li><b>Cultural background:</b> {user_profile['cultural_background'] or 'not specified'}</li>
  <li><b>Occupation:</b> {user_profile['occupation'] or 'not specified'}</li>
  <li><b>Main concerns:</b> {user_profile['user_concerns'] or 'not specified'}</li>
  <li><b>Stress level:</b> {user_profile['stress_level']}</li>
  <li><b>Sleep quality:</b> {user_profile['sleep_quality']}</li>
  <li><b>Motivation / energy:</b> {user_profile['motivation_level']}</li>
  <li><b>Appetite pattern:</b> {user_profile['appetite_pattern']}</li>
  <li><b>Support goal:</b> {user_profile['support_goal']}</li>
  <li><b>Detected mental state:</b> <span style='color:{STATE_DISPLAY[st.session_state.current_state][1]};font-weight:700'>{st.session_state.current_state}</span></li>
</ul>
"""

if st.session_state.get("profile_confirmed"):
    with st.expander("📋 Your Profile Summary Today", expanded=False):
        st.markdown(_profile_summary_html, unsafe_allow_html=True)
else:
    st.markdown(
        f"""
<div class="card">
  <h3>📋 Your Profile Summary Today</h3>
  <p>Please check if the information below is correct. You can change anything in the sidebar on the left.</p>
  {_profile_summary_html}
</div>
""",
        unsafe_allow_html=True,
    )

if daily_needs:
    macros = daily_needs["macros"]
    with st.expander("🔢 Your Estimated Daily Nutritional Needs", expanded=False):
        st.markdown(
            f"""
<div class="card">
  <p style="font-size:0.85rem; color:#6b7280;">
    Calculated using the Harris-Benedict equation + EFSA (2017) macronutrient guidelines.
  </p>
  <ul>
    <li><b>BMI:</b> {daily_needs['bmi']}</li>
    <li><b>Estimated body fat:</b> {daily_needs['body_fat']}%</li>
    <li><b>Daily energy need:</b> {daily_needs['total_calories']} kcal</li>
    <li><b>Carbohydrates:</b> {macros['Carbohydrates (g)'][0]}–{macros['Carbohydrates (g)'][1]} g</li>
    <li><b>Protein:</b> {macros['Protein (g)'][0]}–{macros['Protein (g)'][1]} g</li>
    <li><b>Fat:</b> {macros['Fats (g)'][0]}–{macros['Fats (g)'][1]} g</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )
else:
    st.info("Daily nutritional needs cannot be calculated for the current profile (check age, weight, height, and sex).")

profile_confirmed = st.checkbox("✅ Yes, this information is correct", key="profile_confirmed")
if not st.session_state.get("profile_confirmed"):
    st.info("Please review your profile in the sidebar and tick the box above.")
    st.stop()

if st.session_state.chat_history:
    st.markdown("""
    <style>
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stBaseButton-secondary"] {
        width: 100% !important;
        min-width: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    col_title, col_txt, col_html, col_clear = st.columns([5, 1, 1, 1])
    with col_title:
        st.markdown("#### 💬 Conversation")
    with col_txt:
        _txt = build_chat_txt(st.session_state.chat_history, user_profile)
        st.download_button(
            label="⬇️ .txt",
            data=_txt,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            help="Download conversation as plain text",
            use_container_width=True,
        )
    with col_html:
        _html = build_chat_html(st.session_state.chat_history, user_profile)
        st.download_button(
            label="⬇️ .html",
            data=_html,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            help="Download as HTML — open in browser and use File → Print → Save as PDF",
            use_container_width=True,
        )
    with col_clear:
        if st.session_state.confirm_clear:
            if st.button("⚠️ Confirm", key="confirm_clear_btn", type="primary", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.response_traces = []
                st.session_state.feedback = {}
                st.session_state.last_question = ""
                st.session_state.last_food_matches = []
                st.session_state.last_food_index = 0
                st.session_state.last_recipe_matches = []
                st.session_state.last_recipe_index = 0
                st.session_state.confirm_clear = False
                st.rerun()
            st.caption("Click to confirm")
        else:
            if st.button("🗑️ Clear", key="clear_chat_btn", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()

for i, (q, a) in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
        traces = st.session_state.response_traces
        if i < len(traces):
            render_response_trace(traces[i])

        given_feedback = st.session_state.feedback.get(i)
        st.markdown('<div class="feedback-row">', unsafe_allow_html=True)
        traces = st.session_state.response_traces
        rpath = traces[i].routing_path if i < len(traces) else ""
        fb_col1, fb_col2, fb_col3, fb_col4 = st.columns([1, 1, 3, 7])
        with fb_col1:
            if given_feedback is None:
                if st.button("👍", key=f"fb_up_{i}", help="Good answer"):
                    st.session_state.feedback[i] = "up"
                    save_feedback("up", q, a, rpath)
                    st.rerun()
            elif given_feedback == "up":
                st.markdown("✅")
        with fb_col2:
            if given_feedback is None:
                if st.button("👎", key=f"fb_down_{i}", help="Bad answer"):
                    st.session_state.feedback[i] = "down"
                    save_feedback("down", q, a, rpath)
                    st.rerun()
            elif given_feedback == "down":
                st.markdown("📝")
        with fb_col3:
            if st.button("🔁 Try again", key=f"retry_{i}", help="Re-run this question to get a different answer"):
                st.session_state.retry_question = q
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        if given_feedback == "up":
            st.caption("✅ Thanks for the positive feedback!")
        elif given_feedback == "down":
            st.caption("📝 Thanks — we'll use this to improve the system.")

placeholder_text = (
    "Ask a question about foods, nutrients, or dietary guidance..."
    if not st.session_state.get("last_question")
    else "Ask me something else..."
)

QUESTION_CATEGORIES = [
    ("👤 Profile & Nutrition",    PREDEFINED_QUESTIONS[0:2]),
    ("🍽️ Recipes & Meal Planning", PREDEFINED_QUESTIONS[2:4]),
    ("🏥 Health Conditions",       PREDEFINED_QUESTIONS[4:6]),
    ("🧠 Mood & Mental Health",  PREDEFINED_QUESTIONS[6:9]),
    ("🏋️ Workout Nutrition",       PREDEFINED_QUESTIONS[9:11]),
    ("🌿 Allergies & Substitutes", PREDEFINED_QUESTIONS[11:13]),
    ("🔬 Nutrients & Ingredients", PREDEFINED_QUESTIONS[13:16]),
    ("🌱 Environmental Sustainability Impact",    PREDEFINED_QUESTIONS[16:17]),
]

st.markdown("#### 💡 Try one of these questions:")
for category_label, questions in QUESTION_CATEGORIES:
    with st.expander(category_label):
        for i, q in enumerate(questions):
            idx = PREDEFINED_QUESTIONS.index(q)
            if st.button(q, key=f"preset_q_{idx}"):
                st.session_state["input_box"] = q
                st.session_state["preset_question"] = q

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(placeholder_text, key="input_box")
    submitted = st.form_submit_button("Send")

if st.session_state.get("preset_question") and not submitted:
    user_input = st.session_state["preset_question"]
    submitted = True
    st.session_state["preset_question"] = None


_HEALTH_REC_PROPS = {
    "highlyRecommendedFor":          "highly recommended for",
    "recommendedFor":                "recommended for",
    "RecommendedFor":                "recommended for",
    "notRecommendedFor":             "not recommended for",
    # RestrictedFood only ever targets FoodAllergy individuals (Peanut_Allergy,
    # TreeNuts_Allergy, Milk_Allergy, ...) — it's the ontology's "avoid this food"
    # relation for allergies, same role notRecommendedFor plays for diets/health
    # states, so it feeds the same bucket. Including it here also means
    # _build_health_state_index (below) picks up allergy entities automatically,
    # so an allergy question resolves through this ontology-first path too.
    "RestrictedFood":                "not recommended for",
    "FoodInLimitedAmount":           "should be consumed in limited amounts for",
    "FoodToBeConsumedInHighAmount":  "should be consumed in high amounts for",
}


def _build_health_state_index() -> Dict[str, URIRef]:
    index: Dict[str, URIRef] = {}
    # Index targets of food recommendation properties (e.g. Anxious, Depressed, Anemia)
    for prop_name in _HEALTH_REC_PROPS:
        for _, _, target in ontology_graph.triples((None, SHFO[prop_name], None)):
            if not isinstance(target, URIRef):
                continue
            lbl = get_entity_label(target)
            for key in (normalize_text(lbl), normalize_text(str(target).split("#")[-1])):
                if key and key not in index:
                    index[key] = target
    # Also index targets of hasPositiveEffectOn so states like SleepRegulation
    # are discoverable even when no food recommendedFor triples exist for them.
    for _, _, target in ontology_graph.triples((None, SHFO.hasPositiveEffectOn, None)):
        if not isinstance(target, URIRef):
            continue
        lbl = get_entity_label(target)
        for key in (normalize_text(lbl), normalize_text(str(target).split("#")[-1])):
            if key and key not in index:
                index[key] = target
    return index


HEALTH_STATE_INDEX: Dict[str, URIRef] = _build_health_state_index()

# Maps user-facing wellbeing terms to the normalized ontology entity labels.
# Needed because the ontology uses "Anxious"/"Stressed"/"Depressed" while users
# naturally write "anxiety", "stress", "depression".
_WELLBEING_ALIAS_MAP: Dict[str, str] = {
    "anxiety":            "anxious",
    "anxious":            "anxious",
    "stress":             "stressed",
    "stressed":           "stressed",
    "depression":         "depressed",
    "depressed":          "depressed",
    "lowmood":            "lowmood",
    "moodswings":         "moodswings",
    "moodswing":          "moodswings",
    "emotionalrecovery":  "emotionalrecovery",
    "calm":               "calm",
    "sleep":              "sleepregulation",
    "sleepquality":       "sleepregulation",
    "appetite":           "appetiteregulation",
}

# Ontology allergy entities don't share vocabulary with the app's allergy
# dropdown (app "Dairy"/"Gluten" vs ontology "Milk_Allergy"/"Wheat_Allergy").
# Left to the fuzzy fallback below, "dairyallergy" scores HIGHER against the
# wrong "soyallergy" (0.73) than against the correct "milkallergy" (0.70 —
# just under the cutoff), so it silently resolves to the wrong allergy entity
# instead of failing loudly. These explicit aliases are checked first so that
# can't happen.
_ALLERGY_ALIAS_MAP: Dict[str, str] = {
    "nutsallergy":   "treenutsallergy",
    "dairyallergy":  "milkallergy",
    "glutenallergy": "wheatallergy",
}

# find_health_state_entity only ever returns a single URI, but a "Nuts"
# allergy in the app should exclude both tree nuts AND peanuts. Callers that
# need every matching entity (see _get_diet_and_allergy_exclusions) should
# also collect these.
_ALLERGY_EXTRA_ENTITIES: Dict[str, List[str]] = {
    "nutsallergy": ["Peanut_Allergy"],
}

# The ontology's RestrictedFood/notRecommendedFor allergy tagging is sparse
# and manually curated — for Gluten specifically, only 27 of the 224 foods
# tagged containsMolecule Gluten are also cross-linked to Wheat_Allergy via
# RestrictedFood. containsMolecule is otherwise unused for any other allergen
# (checked: Gluten is the only molecule ever targeted by it), so this can't
# be generalized to Dairy/Soy/Nuts/Eggs, but for Gluten it closes a large,
# real gap — see _get_diet_and_allergy_exclusions.
_ALLERGY_MOLECULE_MAP: Dict[str, str] = {
    "glutenallergy": "Gluten",
}

# Category-level backstop: for an allergy whose entire food group is
# unambiguously the allergen (every individual in it genuinely contains the
# thing), exclude the WHOLE group rather than relying only on the ontology's
# per-food RestrictedFood tagging or a keyword hit on the food's own name —
# a plain "Yoghurt low fat" carries no RestrictedFood/notRecommendedFor triple
# for Milk_Allergy at all in this ontology, so without this a dairy allergy
# only caught it by luck (the word "yoghurt" happening to appear in the
# label). Deliberately NOT done for Nuts_and_Seeds (mixes real nuts with
# seeds like chia/linseed/hemp that aren't a tree-nut allergen — would
# over-exclude) or for Gluten/Soy (no clean single food group exists for
# either; the existing keyword + containsMolecule handling already covers
# them). foodgroup_meat_and_milk_and_milk_products_substitutes is the
# dairy-free-alternatives group and is correctly left OUT of this list.
_ALLERGY_FOOD_GROUP_MAP: Dict[str, List[str]] = {
    "dairyallergy": ["foodgroup_milk_and_milk_products", "foodgroup_cheese"],
    "eggsallergy": ["foodgroup_eggs"],
}


def find_health_state_entity(query: str) -> Optional[URIRef]:
    q_norm = normalize_text(query)
    if q_norm in HEALTH_STATE_INDEX:
        return HEALTH_STATE_INDEX[q_norm]
    allergy_alias = _ALLERGY_ALIAS_MAP.get(q_norm)
    if allergy_alias and allergy_alias in HEALTH_STATE_INDEX:
        return HEALTH_STATE_INDEX[allergy_alias]
    # Alias lookup — maps "anxiety" → "anxious", "stress" → "stressed", etc.
    # Also handles compound questions by scanning the full query for any alias term.
    for alias_key, ontology_key in _WELLBEING_ALIAS_MAP.items():
        if alias_key in q_norm and ontology_key in HEALTH_STATE_INDEX:
            return HEALTH_STATE_INDEX[ontology_key]
    for key, uri in HEALTH_STATE_INDEX.items():
        if key in q_norm or q_norm in key:
            return uri
    # Fuzzy fallback — catches spelling mistakes (e.g. "anemeia" → "Anemia")
    best_uri: Optional[URIRef] = None
    best_score = 0.0
    for key, uri in HEALTH_STATE_INDEX.items():
        score = fuzzy_label_score(q_norm, key)
        if score > best_score and score >= 0.70:
            best_score = score
            best_uri = uri
    return best_uri


_HEALTH_LEVEL_ORDER: List[str] = [
    "highly recommended for",
    "recommended for",
    "should be consumed in high amounts for",
    "should be consumed in limited amounts for",
    "not recommended for",
]


_MAX_FOODS_PER_LEVEL = 30


# The ontology's FoodGroup class hierarchy (Dairy/Fish/Grains/Vegetables/... under
# owl:Class shf:FoodGroup) is only ever used as a hasFoodSource target on nutrient
# individuals (e.g. "Lysine hasFoodSource Dairy, Fish, Meat, ..."). No actual food is
# rdf:typed with those classes, and none is belongsToFoodGroup-tagged with them either.
# Real foods are tagged via belongsToFoodGroup against a separate, flatter set of
# lowercase individuals (foodgroup_vegetables, foodgroup_cheese, etc.). Vegan/
# Vegetarian/Pescatarian have no recommendedFor triples at all in this ontology —
# only notRecommendedFor (handled generically above) — so "recommended" foods for
# these diets are derived from food-group membership instead.
_DIET_FOOD_GROUP_KEYS: Dict[str, List[str]] = {
    "vegan": [
        "foodgroup_vegetables", "foodgroup_fruits",
        "foodgroup_cereal_products_and_types_of_flour", "foodgroup_bread",
    ],
    "vegetarian": [
        "foodgroup_vegetables", "foodgroup_fruits",
        "foodgroup_cereal_products_and_types_of_flour", "foodgroup_bread",
        "foodgroup_milk_and_milk_products", "foodgroup_cheese",
    ],
    "pescatarian": [
        "foodgroup_vegetables", "foodgroup_fruits",
        "foodgroup_cereal_products_and_types_of_flour", "foodgroup_bread",
        "foodgroup_milk_and_milk_products", "foodgroup_cheese",
    ],
}

# There is no "Fish" food group tagged on any real food (see note above), so
# Pescatarian's fish/seafood allowance is matched by name instead.
_FISH_NAME_KEYWORDS = {
    "fish", "salmon", "tuna", "cod", "herring", "mackerel", "sardine", "sardines",
    "trout", "haddock", "anchovy", "anchovies", "shrimp", "prawn", "prawns",
    "crab", "lobster", "mussel", "mussels", "oyster", "oysters", "squid", "seafood",
    "plaice", "sole", "eel", "kipper",
}

# Halal has only a handful of notRecommendedFor triples (and only on composite
# recipes, never on raw foods), so raw-food classification is done by name instead:
# pork/alcohol-named foods are excluded, everything else is allowed.
_HALAL_EXCLUDED_KEYWORDS = {
    "pork", "bacon", "ham", "lard", "gammon",
    "wine", "beer", "alcohol", "rum", "whisky", "whiskey", "vodka",
    "brandy", "liqueur", "liquor", "sherry", "port",
}
# Guards against false positives from the keyword list above: "wine gums" are an
# alcohol-free candy (matches "wine") and "beer, alcohol-free" is literally
# non-alcoholic (matches "beer"/"alcohol") — skip exclusion when these appear too.
_HALAL_EXCLUSION_EXCEPTIONS = {"gum", "gums", "free"}

_DIET_URI_TO_KEY: Dict[str, str] = {
    "vegan": "vegan",
    "vegetarian": "vegetarian",
    "pescatarian": "pescatarian",
    "halaldiet": "halal",
    "ketodiet": "keto",
}


def _label_has_keyword(label: str, keywords: Set[str]) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", label.lower()))
    return bool(tokens & keywords)


def _get_diet_food_group_recommended(diet_key: str) -> Set[str]:
    group_keys = _DIET_FOOD_GROUP_KEYS.get(diet_key)
    if not group_keys:
        return set()
    group_uris = {SHFO[key] for key in group_keys}
    recommended: Set[str] = set()
    for food, _, group in ontology_graph.triples((None, SHFO.belongsToFoodGroup, None)):
        if group in group_uris:
            recommended.add(get_entity_label(food))
    if diet_key == "pescatarian":
        for label, _uri in FOOD_CANDIDATES:
            if _label_has_keyword(label, _FISH_NAME_KEYWORDS):
                recommended.add(label)
    return recommended


def _is_halal_excluded(label: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", label.lower()))
    if not (tokens & _HALAL_EXCLUDED_KEYWORDS):
        return False
    return not (tokens & _HALAL_EXCLUSION_EXCEPTIONS)


def _get_halal_food_buckets() -> Tuple[Set[str], Set[str]]:
    recommended: Set[str] = set()
    not_recommended: Set[str] = set()
    for label, _uri in FOOD_CANDIDATES:
        if _is_halal_excluded(label):
            not_recommended.add(label)
        else:
            recommended.add(label)
    return recommended, not_recommended


# Vegan/Vegetarian have the same gap as Halal above, just worse: the ontology's
# notRecommendedFor triples for these diets only ever landed on raw meat cuts
# (e.g. "Beef T-bone steak raw" IS tagged notRecommendedFor Vegan/Vegetarian) —
# composite/processed dishes built from those same ingredients (Beef fat, Soup
# clear w meat, omelettes, milk-chocolate-coated raisins) and eggs/dairy were
# never tagged at all, so they silently leaked into every Vegan/Vegetarian
# persona's iron-deficiency and pre/post-workout answers. Same name-based
# fallback as Halal, with two guards this dataset specifically needs:
# - "wo X" ("without X", e.g. "Croissant prepared wo butter", "Pasta wo egg")
#   negates just that one keyword hit, not the whole label.
# - a plant-qualifier immediately before the keyword ("coconut milk", "coconut
#   meat", "peanut butter") means it isn't actually the animal product.
_VEGAN_EXCLUDED_KEYWORDS = {
    # meat, poultry, game, and their processed forms
    "beef", "pork", "veal", "lamb", "chicken", "turkey", "duck", "horse",
    "bacon", "ham", "gammon", "lard", "sausage", "salami", "meat", "meatloaf",
    "liver", "kromesky", "hamburger", "burger",
    # fish / seafood — same names Pescatarian already matches on
    *_FISH_NAME_KEYWORDS,
    # eggs
    "egg", "eggs",
    # dairy
    "milk", "cheese", "cream", "butter", "buttermilk", "yoghurt", "yogurt",
    "quark", "mascarpone", "whey",
    # other animal-derived
    "gelatin", "gelatine", "honey",
}
# Vegetarian allows eggs and dairy — only meat/fish/gelatin/honey are excluded.
_VEGETARIAN_EXCLUDED_KEYWORDS = _VEGAN_EXCLUDED_KEYWORDS - {
    "egg", "eggs", "milk", "cheese", "cream", "butter", "buttermilk",
    "yoghurt", "yogurt", "quark", "mascarpone", "whey", "honey",
}
_DIET_EXCLUSION_NEGATORS = {"wo", "without"}
_DIET_EXCLUSION_QUALIFIER_EXCEPTIONS = {
    "coconut", "soy", "soya", "almond", "oat", "cashew", "peanut", "rice", "hemp", "cocoa",
}
# Trailing modifier, e.g. "Pasta gluten free raw" / "Bakery mix ... gluten free" —
# the keyword is immediately followed by "free" rather than preceded by "wo".
_DIET_EXCLUSION_TRAILING_NEGATORS = {"free"}


def _is_diet_excluded(label: str, keywords: Set[str]) -> bool:
    tokens = re.findall(r"[a-z0-9]+", label.lower())
    for i, tok in enumerate(tokens):
        if tok not in keywords:
            continue
        prev_tok = tokens[i - 1] if i > 0 else ""
        next_tok = tokens[i + 1] if i + 1 < len(tokens) else ""
        if prev_tok in _DIET_EXCLUSION_NEGATORS or prev_tok in _DIET_EXCLUSION_QUALIFIER_EXCEPTIONS:
            continue
        if next_tok in _DIET_EXCLUSION_TRAILING_NEGATORS:
            continue
        return True
    return False


def _label_explicitly_free_of(label: str, keywords: Set[str]) -> bool:
    """True if the label itself says "<keyword> free" for one of the given
    keywords (e.g. "Pasta gluten free raw", "Gluten free bread dark") — used
    to override an ontology-sourced exclusion that's wrong for a product
    explicitly marketed free of that allergen/ingredient."""
    tokens = re.findall(r"[a-z0-9]+", label.lower())
    for i, tok in enumerate(tokens):
        if tok in keywords and i + 1 < len(tokens) and tokens[i + 1] in _DIET_EXCLUSION_TRAILING_NEGATORS:
            return True
    return False


def _get_keyword_food_buckets(keywords: Set[str]) -> Tuple[Set[str], Set[str]]:
    """Generic name-based split of every food candidate into
    (not matching any keyword, matching at least one keyword) — the same
    approximation _get_halal_food_buckets/_get_vegan_vegetarian_food_buckets
    use, generalized so Keto and the allergy checks below can reuse it."""
    recommended: Set[str] = set()
    not_recommended: Set[str] = set()
    for label, _uri in FOOD_CANDIDATES:
        if _is_diet_excluded(label, keywords):
            not_recommended.add(label)
        else:
            recommended.add(label)
    return recommended, not_recommended


def _get_vegan_vegetarian_food_buckets(diet_key: str) -> Tuple[Set[str], Set[str]]:
    keywords = _VEGAN_EXCLUDED_KEYWORDS if diet_key == "vegan" else _VEGETARIAN_EXCLUDED_KEYWORDS
    return _get_keyword_food_buckets(keywords)


@lru_cache(maxsize=1)
def _get_high_carb_food_labels(sugar_g: float = 8.0, carb_g: float = 20.0) -> frozenset:
    """Food labels a keto diet rules out on macros alone — per-100g sugar above
    sugar_g OR total carbohydrate above carb_g. Numeric backstop for
    _KETO_EXCLUDED_KEYWORDS, which is name-based and misses whole foods like
    dried fruit, fruit juice and fruit smoothies."""
    out: Set[str] = set()
    for label, uri in FOOD_CANDIDATES:
        s = ontology_graph.value(uri, SHFO["hasAmountOfSugar"])
        c = ontology_graph.value(uri, SHFO["hasAmountOfCarbohydrates"])
        try:
            if (s is not None and float(s) > sugar_g) or (c is not None and float(c) > carb_g):
                out.add(label)
        except (TypeError, ValueError):
            continue
    return frozenset(out)


# Keto's ontology coverage is as sparse as Vegan/Vegetarian's was (34 notRecommendedFor
# triples across 981 foods) — e.g. "Banana" is tagged notRecommendedFor KetoDiet but
# "Pasta white raw", "Rice white raw", "Sugar granulated" and "Honey" aren't tagged at
# all. Scoped to the clearest, least ambiguous high-carb categories (grains/starches,
# sugar/sweeteners, sweet baked goods) rather than all fruit — keto-compatible carb
# thresholds vary too much by fruit to approximate safely by name alone.
_KETO_EXCLUDED_KEYWORDS = {
    "bread", "pasta", "rice", "noodle", "noodles", "flour", "cereal", "oat", "oats",
    "cornflakes", "corn", "potato", "potatoes", "crisps", "muesli", "porridge",
    "sugar", "honey", "syrup", "jam", "marmalade", "treacle",
    "cake", "biscuit", "pastry", "doughnut", "waffle", "candy", "chocolate",
    "sweets", "dessert", "pie", "tart", "pancake", "pancakes", "muffin",
    # grains — keto excludes these generically, not just the gluten-bearing ones
    "wheat", "barley", "rye", "spelt", "couscous", "semolina", "bulgur",
    "wrap", "tortilla", "cracker", "crispbread", "shortbread", "beer",
    # sweet baked goods / breakfast pastries
    "croissant", "brioche", "bun", "bagel", "scone", "crumpet", "granola",
    "oatmeal", "wholemeal", "multigrain", "crispbake", "crispbakes",
    # fruit and fruit drinks are high-sugar — keto excludes them (numeric
    # backstop in _get_high_carb_food_labels catches the rest)
    "banana", "raisin", "raisins", "date", "dates", "prune", "prunes",
    "fig", "figs", "smoothie",
}

# Allergy keyword sets, name-based fallback for the same reason as above — the
# ontology's RestrictedFood/notRecommendedFor tagging on these allergy entities is
# fairly good (9-31 triples each) but still misses composite dishes, e.g. "Biscuit
# brown/wholemeal" (clearly wheat-based) has no Wheat_Allergy tag at all.
_DAIRY_ALLERGY_KEYWORDS = {
    "milk", "cheese", "cream", "butter", "buttermilk", "yoghurt", "yogurt",
    "quark", "mascarpone", "whey",
}
_EGG_ALLERGY_KEYWORDS = {"egg", "eggs"}
_NUTS_ALLERGY_KEYWORDS = {
    "peanut", "peanuts", "almond", "almonds", "hazelnut", "hazelnuts",
    "walnut", "walnuts", "cashew", "cashews", "pistachio", "pistachios",
    "pecan", "pecans", "macadamia", "brazil", "nuts", "nut",
}
_SOY_ALLERGY_KEYWORDS = {"soy", "soya", "tofu", "edamame"}
_GLUTEN_ALLERGY_KEYWORDS = {
    "wheat", "gluten", "barley", "rye", "spelt", "couscous", "semolina",
    "bulgur", "cracker", "breadcrumbs", "brcrumbs",
    # composite / baked wheat foods that carry no Wheat/Gluten ontology tag —
    # over-exclude here rather than let a gluten-bearing food reach the answer
    "bread", "toast", "crouton", "croutons", "breadstick", "breadsticks",
    "croissant", "brioche", "baguette", "ciabatta", "focaccia", "naan",
    "pita", "chapati", "roti", "bun", "buns", "roll", "rolls", "bagel",
    "pretzel", "scone", "crumpet", "wrap",
    "crispbake", "crispbakes", "crispbread", "wholemeal", "wholewheat",
    "multigrain", "granary", "rusk", "matzo", "matzah",
    "pancake", "pancakes", "waffle", "waffles", "muffin", "muffins",
    "doughnut", "donut", "pastry", "pie", "tart", "cake", "cakes",
    "cookie", "cookies", "biscuit", "biscuits", "shortbread", "sponge",
    "brownie", "brownies", "pancakes",
    "spaghetti", "macaroni", "penne", "tagliatelle", "vermicelli",
    "lasagne", "lasagna", "noodle", "noodles", "gnocchi",
    "muesli", "granola", "weetabix", "farro", "seitan",
}

# Keyed the same way _ALLERGY_EXTRA_ENTITIES/_ALLERGY_MOLECULE_MAP are — normalize_text
# of "<Allergy> allergy" as it appears in a user profile's allergies list.
_ALLERGY_KEYWORD_MAP: Dict[str, Set[str]] = {
    "dairyallergy": _DAIRY_ALLERGY_KEYWORDS,
    "eggsallergy": _EGG_ALLERGY_KEYWORDS,
    "nutsallergy": _NUTS_ALLERGY_KEYWORDS,
    "soyallergy": _SOY_ALLERGY_KEYWORDS,
    "glutenallergy": _GLUTEN_ALLERGY_KEYWORDS,
}


def build_health_recommendation_context(
    question: str, entity_hint: str = "", sections: Optional[Set[str]] = None,
    user_profile: Optional[dict] = None, max_per_level: int = _MAX_FOODS_PER_LEVEL,
) -> Optional[str]:
    """Build the ontology-grounded context for a health/diet/allergy entity.

    sections, if given, restricts which of _HEALTH_LEVEL_ORDER's buckets are
    included in the output (e.g. {"not recommended for"} for an avoidance-only
    question) — default None includes all non-empty sections, same as before
    this parameter existed. Doesn't change entity resolution or the nutrient
    section, only which food buckets get rendered.

    user_profile, if given, strips any food from the recommendation buckets
    that conflicts with the user's own diet type or allergies (reusing
    _get_diet_and_allergy_exclusions, the same check meal_suggestion uses) —
    without this, a question like "what foods help with anemia" would happily
    recommend a nut-heavy food to a nut-allergic user, since this function
    otherwise only reasons about the health condition, never the asker. The
    condition's own "not recommended for" bucket is left untouched — that's
    already the correct avoidance answer for the condition itself.
    """
    state_uri = find_health_state_entity(entity_hint) if entity_hint else None
    if not state_uri:
        state_uri = find_health_state_entity(question)
    if not state_uri:
        return None

    state_label = get_entity_label(state_uri)
    # Use sets to deduplicate foods across alias properties (e.g. recommendedFor + RecommendedFor)
    buckets: Dict[str, Set[str]] = {lv: set() for lv in _HEALTH_LEVEL_ORDER}
    label_to_uri: Dict[str, URIRef] = {}

    for prop_name, rel_label in _HEALTH_REC_PROPS.items():
        if rel_label not in buckets:
            buckets[rel_label] = set()
        for food, _, _ in ontology_graph.triples((None, SHFO[prop_name], state_uri)):
            if isinstance(food, URIRef):
                label = get_entity_label(food)
                buckets[rel_label].add(label)
                label_to_uri.setdefault(label, food)

    diet_key = _DIET_URI_TO_KEY.get(normalize_text(str(state_uri).split("#")[-1]))
    if diet_key == "halal":
        halal_recommended, halal_not_recommended = _get_halal_food_buckets()
        buckets["recommended for"] |= halal_recommended
        buckets["not recommended for"] |= halal_not_recommended
    elif diet_key:
        buckets["recommended for"] |= _get_diet_food_group_recommended(diet_key)

    excluded_from_recs: Set[str] = set()
    if user_profile:
        profile_excluded = _get_diet_and_allergy_exclusions(user_profile)
        if profile_excluded:
            for level in _HEALTH_LEVEL_ORDER:
                if level == "not recommended for":
                    continue
                direct_conflicts = buckets[level] & profile_excluded
                buckets[level] -= profile_excluded
                # RestrictedFood/notRecommendedFor (what profile_excluded is
                # built from) is only ever asserted on Food individuals, never
                # Recipe individuals — a recommended recipe made from an
                # excluded ingredient would otherwise slip through untouched.
                recipe_conflicts = set()
                for label in buckets[level]:
                    uri = label_to_uri.get(label)
                    if uri is None or (uri, RDF.type, SHFO["Recipe"]) not in ontology_graph:
                        continue
                    ingredient_labels = {ing["label"] for ing in get_recipe_ingredients(uri)}
                    if ingredient_labels & profile_excluded:
                        recipe_conflicts.add(label)
                buckets[level] -= recipe_conflicts
                excluded_from_recs |= direct_conflicts | recipe_conflicts

    levels_to_render = [lv for lv in _HEALTH_LEVEL_ORDER if sections is None or lv in sections]
    has_food_recs = any(buckets.get(lv) for lv in levels_to_render)

    # Even without food recommendations, the state may have nutrient effects
    nutrient_section = _get_nutrients_for_state(state_uri, state_label)

    if not has_food_recs:
        if not nutrient_section:
            return None  # nothing at all — caller will fall back to vector store
        return (
            f"HEALTH RECOMMENDATION CONTEXT\n"
            f"Health condition: {state_label}\n"
            f"No food recommendations found for {state_label} in the SustainHealthyFood Ontology.\n"
            f"{nutrient_section}"
        ) + _profile_exclusion_note(excluded_from_recs, user_profile)

    lines = [
        f"HEALTH RECOMMENDATION CONTEXT",
        f"Health condition: {state_label}",
        f"The following food recommendations come from the SustainHealthyFood Ontology for {state_label}:",
    ]
    for rel_label in levels_to_render:
        foods_sorted = sorted(buckets.get(rel_label, set()))
        total = len(foods_sorted)
        heading = {
            "highly recommended for":                    f"HIGHLY RECOMMENDED for {state_label} ({total} total)",
            "recommended for":                           f"RECOMMENDED for {state_label} ({total} total)",
            "should be consumed in high amounts for":    f"EAT IN HIGH AMOUNTS for {state_label} ({total} total)",
            "should be consumed in limited amounts for": f"EAT IN LIMITED AMOUNTS for {state_label} ({total} total)",
            "not recommended for":                       f"NOT RECOMMENDED / AVOID for {state_label} ({total} total)",
        }.get(rel_label, f"{rel_label.upper()} ({total} total)")
        if not foods_sorted:
            # Skip empty categories entirely — don't feed "0 total / None found"
            # placeholders into the LLM context, since it tends to echo them
            # back verbatim instead of omitting the section as instructed.
            continue
        lines.append(f"\n{heading}:")
        shown = foods_sorted[:max_per_level]
        for f in shown:
            lines.append(f"- {f}")
        if total > max_per_level:
            lines.append(f"  ... and {total - max_per_level} more foods in this category.")

    if nutrient_section:
        lines.append(nutrient_section)

    return "\n".join(lines) + _profile_exclusion_note(excluded_from_recs, user_profile)


def _get_nutrients_for_state(state_uri: URIRef, state_label: str) -> str:
    """Returns a formatted string listing nutrients with hasPositiveEffectOn the given state."""
    nutrients = sorted(
        get_entity_label(n)
        for n in ontology_graph.subjects(SHFO.hasPositiveEffectOn, state_uri)
        if isinstance(n, URIRef)
    )
    if not nutrients:
        return ""
    lines = [
        f"\nNUTRIENTS WITH POSITIVE EFFECT ON {state_label.upper()}:",
        f"The following nutrients support {state_label} according to the SustainHealthyFood Ontology:",
    ]
    for nutrient in nutrients:
        lines.append(f"- {nutrient}")
    return "\n".join(lines)


# Maps normalized user-profile life_stage values to their LifeStage class in the
# ontology. Note the ontology's own local names: the "Lactating" life stage is
# declared under the URI fragment "Lactacing" (its rdfs:label is "Lactating").
_LIFE_STAGE_CLASS_MAP: Dict[str, str] = {
    "generaladult": "GeneralAdult",
    "pregnant":     "Pregnant",
    "lactating":    "Lactacing",
    "pedriatic":    "Pedriatic",
    "pediatric":    "Pedriatic",
}

# The ontology has no dedicated pediatric diet/relation, so "nutritionally rich
# food" for that life stage is grounded in membership of these nutrient-density
# food classes instead (same classes already surfaced in _get_ingredient_summary).
_PEDIATRIC_NUTRIENT_DENSE_CLASSES: List[Tuple[str, str]] = [
    ("FoodHighInProtein", "high in protein"),
    ("FoodHighInIron",    "rich in iron"),
    ("FoodHighInFiber",   "high in fiber"),
]

_MAX_LIFE_STAGE_FOODS = 20

# The ontology has no dedicated "GeneralAdult" health-state/diet entity with its
# own recommendedFor triples (unlike Pregnant/Lactating), so — same fix as
# Vegan/Vegetarian/Halal earlier — a balanced-diet recommendation is derived
# from belongsToFoodGroup membership instead: the standard healthy staple
# groups are "recommended", the discretionary ones are "limit".
_GENERAL_ADULT_RECOMMENDED_GROUPS = [
    "foodgroup_vegetables", "foodgroup_fruits", "foodgroup_legumes",
    "foodgroup_nuts_and_seeds", "foodgroup_starchyvegetables",
    "foodgroup_cereal_products_and_types_of_flour", "foodgroup_bread",
    "foodgroup_milk_and_milk_products", "foodgroup_cheese", "foodgroup_eggs",
    "foodgroup_meat",
]
_GENERAL_ADULT_LIMIT_GROUPS = ["foodgroup_pastry_and_biscuits", "foodgroup_alcoholicbeverages"]


def _get_general_adult_food_buckets() -> Tuple[Set[str], Set[str]]:
    recommended: Set[str] = set()
    for key in _GENERAL_ADULT_RECOMMENDED_GROUPS:
        group_uri = SHFO[key]
        for food, _, _ in ontology_graph.triples((None, SHFO.belongsToFoodGroup, group_uri)):
            recommended.add(get_entity_label(food))
    for label, _uri in FOOD_CANDIDATES:
        if _label_has_keyword(label, _FISH_NAME_KEYWORDS):
            recommended.add(label)

    limit: Set[str] = set()
    for key in _GENERAL_ADULT_LIMIT_GROUPS:
        group_uri = SHFO[key]
        for food, _, _ in ontology_graph.triples((None, SHFO.belongsToFoodGroup, group_uri)):
            limit.add(get_entity_label(food))

    return recommended, limit


def build_life_stage_context(
    life_stage: str, question: str, user_profile: Optional[dict] = None
) -> str:
    """Ground the user's current life stage in the ontology's own data rather
    than letting the LLM reason about it from general knowledge:
    - General Adult: a balanced-diet recommendation derived from food-group
      membership (see _get_general_adult_food_buckets), plus foods to limit.
    - Pregnant: PregnancyRestrictedDiet's restricted-food description plus the
      Pregnancy HealthState's food recommendations (recommendedFor / notRecommendedFor).
    - Lactating: LactatingDiet's description plus any food/recipe marked
      isIdealFor the Lactating life stage.
    - Pedriatic: nutrient-dense food classes (high protein / iron / fiber) as a
      stand-in for a dedicated pediatric diet, which the ontology doesn't model.

    user_profile, if given, strips foods conflicting with the user's own diet
    type or allergies from every branch above (reusing
    _get_diet_and_allergy_exclusions) — this context used to be built from
    life stage alone, so e.g. a Halal General Adult would still see Bacon
    under "Recommended balanced-diet foods" since food-group membership has
    no notion of diet/allergy.
    """
    excluded = _get_diet_and_allergy_exclusions(user_profile) if user_profile else set()
    class_name = _LIFE_STAGE_CLASS_MAP.get(normalize_text(life_stage))
    if not class_name:
        return ""

    stage_uri = SHFO[class_name]
    stage_label = get_entity_label(stage_uri)
    stage_comment = ontology_graph.value(stage_uri, RDFS.comment)

    lines = [f"LIFE STAGE CONTEXT (from the SustainHealthyFood Ontology, life stage: {stage_label}):"]
    if stage_comment:
        lines.append(str(stage_comment))

    if class_name == "GeneralAdult":
        lines.append(
            "As a general adult, the ontology defines no pregnancy-, breastfeeding-, "
            "or growth-stage-specific diet or nutrient adjustment — standard dietary "
            "guidelines apply."
        )
        recommended, limit = _get_general_adult_food_buckets()
        if excluded:
            recommended = recommended - excluded
            limit = limit - excluded
        if recommended:
            foods_sorted = sorted(recommended)
            total = len(foods_sorted)
            shown = foods_sorted[:_MAX_LIFE_STAGE_FOODS]
            suffix = f" (+{total - _MAX_LIFE_STAGE_FOODS} more)" if total > _MAX_LIFE_STAGE_FOODS else ""
            lines.append(
                f"\nRecommended balanced-diet foods (standard food groups, {total} total): "
                f"{', '.join(shown)}{suffix}"
            )
        if limit:
            foods_sorted = sorted(limit)
            total = len(foods_sorted)
            shown = foods_sorted[:_MAX_LIFE_STAGE_FOODS]
            suffix = f" (+{total - _MAX_LIFE_STAGE_FOODS} more)" if total > _MAX_LIFE_STAGE_FOODS else ""
            lines.append(
                f"\nFoods to limit (discretionary food groups, {total} total): "
                f"{', '.join(shown)}{suffix}"
            )

    elif class_name == "Pregnant":
        diet_comment = ontology_graph.value(SHFO.PregnancyRestrictedDiet, RDFS.comment)
        if diet_comment:
            lines.append(f"\nPregnancy-restricted diet guidance: {diet_comment}")
        health_ctx = build_health_recommendation_context(question, "Pregnancy", user_profile=user_profile)
        if health_ctx:
            lines.append(f"\n{health_ctx}")

    elif class_name == "Lactacing":
        diet_comment = ontology_graph.value(SHFO.LactatingDiet, RDFS.comment)
        if diet_comment:
            lines.append(f"\nLactation diet guidance: {diet_comment}")
        ideal_foods = sorted({
            get_entity_label(s)
            for s, _, _ in ontology_graph.triples((None, SHFO["isIdealFor"], stage_uri))
            if isinstance(s, URIRef)
        } - excluded)
        if ideal_foods:
            shown = ideal_foods[:_MAX_LIFE_STAGE_FOODS]
            lines.append(
                f"\nFoods/recipes marked ideal for lactation in the ontology: {', '.join(shown)}"
            )

    elif class_name == "Pedriatic":
        for cls_name, tag in _PEDIATRIC_NUTRIENT_DENSE_CLASSES:
            foods = sorted({
                get_entity_label(s)
                for s in ontology_graph.subjects(RDF.type, SHFO[cls_name])
                if isinstance(s, URIRef)
            } - excluded)
            if foods:
                total = len(foods)
                shown = foods[:_MAX_LIFE_STAGE_FOODS]
                suffix = f" (+{total - _MAX_LIFE_STAGE_FOODS} more)" if total > _MAX_LIFE_STAGE_FOODS else ""
                lines.append(
                    f"\nFoods {tag} (nutrient-dense, suitable for growth, {total} total): "
                    f"{', '.join(shown)}{suffix}"
                )

    return "\n".join(lines)


def _build_constraint_summary(user_profile: dict) -> str:
    """'Diet: X; Allergies: Y; Life stage: Z' summary of the groundable parts
    of the user's profile — PROFILE_AVOIDANCE_PROMPT/PROFILE_RECOMMENDATION_
    PROMPT's constraint_summary. Factored out so every routing path that can
    land on a profile_avoidance-style answer (the dedicated branch below, and
    health_recommendation's fallback redirect further down) builds it
    identically rather than each re-deriving it."""
    constraint_parts = []
    diet = user_profile.get("diet_type", "None")
    if diet not in ("None", ""):
        constraint_parts.append(f"Diet: {diet}")
    allergies = user_profile.get("allergies", [])
    if allergies:
        constraint_parts.append(f"Allergies: {', '.join(allergies)}")
    life_stage = user_profile.get("life_stage", "General Adult")
    if life_stage in ("Pregnant", "Lactating"):
        constraint_parts.append(f"Life stage: {life_stage}")
    return "; ".join(constraint_parts)


# Bare avoidance phrasing with no "my diet"/"my allerg" wording at all (e.g.
# "what foods should I avoid eating?") is just as common and just as clearly
# about the asker's own profile, but used to fall straight through
# is_profile_avoidance_question's check below to health_recommendation's
# vector-search fallback, which has no diet/allergy awareness whatsoever —
# e.g. a Vegan user asking this got back toffees-for-diabetics and
# oils-for-fat-restriction, nothing to do with veganism. This phrase list
# catches the bare form; the exclusion regex below guards it from hijacking
# questions that are really about a meal, timing, or named condition, which
# have their own more specific routes (meal_suggestion, general_cooking,
# health_recommendation) and shouldn't be answered from diet/allergy
# exclusions alone.
_GENERIC_AVOIDANCE_PHRASES = (
    "what should i avoid eating", "what should i avoid", "what foods should i avoid",
    "what can i not eat", "what can't i eat", "what cant i eat",
    "what shouldn't i eat", "what shouldnt i eat", "foods i should avoid",
)
_GENERIC_AVOIDANCE_EXCLUDE_RE = re.compile(
    r"\b(before training|after training|workout|exercise|breakfast|lunch|dinner|snack|"
    r"anemia|diabetes|anxiety|stress|sleep|mood|energy)\b", re.I
)


def is_profile_avoidance_question(question: str) -> bool:
    """Detects 'what should I avoid' questions — both explicitly personal
    phrasing ('given my diet/allergies/life stage') and bare phrasing with no
    named diet/allergy word at all ('what foods should I avoid eating?', see
    _GENERIC_AVOIDANCE_PHRASES above). The router's Rule 1 would otherwise
    force profile_reasoning (generic vector search) for the explicitly-
    personal phrasing — but the real answer should come from each of the
    user's *actual* diet/allergy/life-stage constraints in the ontology, not
    a generic guideline search that has no idea which diet or allergy applies.
    """
    q = question.lower()
    mentions_constraint = any(t in q for t in (
        "my allerg", "my diet", "my life stage", "my current life stage",
    ))
    asks_to_avoid = any(t in q for t in (
        "avoid", "not eat", "not consume", "should not", "shouldn't eat", "shouldn't consume",
    ))
    if mentions_constraint and asks_to_avoid:
        return True
    if _GENERIC_AVOIDANCE_EXCLUDE_RE.search(q):
        return False
    return any(p in q for p in _GENERIC_AVOIDANCE_PHRASES)


# Only the avoid-list matters for this question shape — e.g. Halal's own
# "recommended" bucket is ~880 foods (everything not pork/alcohol-named, see
# _get_halal_food_buckets), which would swamp the context with foods that
# have nothing to do with an avoidance question.
_AVOID_ONLY_SECTIONS = {"not recommended for"}


def build_profile_avoidance_context(user_profile: dict) -> Optional[str]:
    """Ground a 'what should I avoid given my diet/allergies/life stage'
    question in the ontology per-constraint, using the user's actual profile
    fields rather than requiring the question text to name a specific diet
    or allergy. Combines one health-recommendation lookup per constraint
    (diet type, each allergy) plus the life-stage restricted-food
    description, so a persona with multiple constraints gets a single
    grounded answer covering everything that actually applies to them —
    reuses build_health_recommendation_context/build_life_stage_context
    rather than re-deriving ontology queries from scratch.
    """
    context_sections = []
    # Capped well below build_health_recommendation_context's own default
    # (30) — that default suits a single-condition question, but
    # profile_avoidance can stack a diet section AND an allergy section (AND
    # a life-stage one) in a single answer, and the LLM has proven unreliable
    # at respecting the "3-6 bullet points" instruction in PROFILE_AVOIDANCE_
    # PROMPT on its own — it was echoing back nearly everything it was shown
    # instead. Capping what it's shown in the first place is what actually
    # guarantees a readable answer, rather than hoping the model complies.
    _AVOIDANCE_MAX_PER_LEVEL = 10

    diet = user_profile.get("diet_type", "None")
    if diet not in ("None", ""):
        diet_ctx = build_health_recommendation_context(
            diet, diet, sections=_AVOID_ONLY_SECTIONS, max_per_level=_AVOIDANCE_MAX_PER_LEVEL
        )
        if diet_ctx:
            context_sections.append(f"=== DIET CONSTRAINT: {diet} ===\n{diet_ctx}")

    for allergy in user_profile.get("allergies", []):
        allergy_ctx = build_health_recommendation_context(
            f"{allergy} allergy", f"{allergy} allergy", sections=_AVOID_ONLY_SECTIONS,
            max_per_level=_AVOIDANCE_MAX_PER_LEVEL,
        )
        if allergy_ctx:
            context_sections.append(f"=== ALLERGY CONSTRAINT: {allergy} ===\n{allergy_ctx}")

    life_stage = user_profile.get("life_stage", "General Adult")
    if life_stage in ("Pregnant", "Lactating"):
        life_stage_ctx = build_life_stage_context(life_stage, "", user_profile=user_profile)
        if life_stage_ctx:
            context_sections.append(f"=== LIFE STAGE CONSTRAINT: {life_stage} ===\n{life_stage_ctx}")

    if not context_sections:
        return None
    return "\n\n".join(context_sections)


def is_profile_recommendation_question(question: str) -> bool:
    """Detects generic 'what food is recommended for my diet/life stage'
    questions — the positive-side mirror of is_profile_avoidance_question,
    for the same reason: Rule 1 of the router would otherwise force
    profile_reasoning (generic vector search) for personally-phrased
    ('my diet', 'my life stage') questions."""
    q = question.lower()
    mentions_constraint = any(t in q for t in (
        "my allerg", "my diet", "my life stage", "my current life stage",
    ))
    asks_for_recommendation = any(t in q for t in (
        "recommended", "recommend", "what should i eat", "what can i eat",
        "good for", "best for",
    ))
    return mentions_constraint and asks_for_recommendation


_RECOMMENDATION_ONLY_SECTIONS = {"highly recommended for", "recommended for"}


def build_profile_recommendation_context(user_profile: dict) -> Optional[str]:
    """Ground a 'what food is recommended for my life stage and diet
    preference' question in the ontology per-constraint — the positive-side
    mirror of build_profile_avoidance_context. Unlike avoidance, life stage is
    always included (not just Pregnant/Lactating): General Adult now has a
    real recommended list too (see _get_general_adult_food_buckets), and
    Pedriatic already did.
    """
    context_sections = []

    diet = user_profile.get("diet_type", "None")
    if diet not in ("None", ""):
        diet_ctx = build_health_recommendation_context(diet, diet, sections=_RECOMMENDATION_ONLY_SECTIONS)
        if diet_ctx:
            context_sections.append(f"=== DIET PREFERENCE: {diet} ===\n{diet_ctx}")

    life_stage = user_profile.get("life_stage", "General Adult")
    life_stage_ctx = build_life_stage_context(life_stage, "", user_profile=user_profile)
    if life_stage_ctx:
        context_sections.append(f"=== LIFE STAGE: {life_stage} ===\n{life_stage_ctx}")

    if not context_sections:
        return None

    # If nothing above actually named any foods (e.g. "None" diet + a life
    # stage whose ontology entry is prose-only, like Lactating), the LLM has
    # no grounded food to work with and invents a generic list from training
    # data — which then bypasses the diet/allergy filter. Append a concrete
    # profile-safe list so it has real foods to recommend.
    joined = "\n\n".join(context_sections)
    if not re.search(r"\n\s*[-•*]\s+\w", joined):
        excluded = _get_diet_and_allergy_exclusions(user_profile)
        safe = [
            (lbl, _get_ingredient_summary(uri), _meal_food_rank(uri))
            for lbl, uri in _fallback_meal_pool() if lbl not in excluded
        ]
        safe.sort(key=lambda x: (-x[2], x[0]))
        safe = _dedup_food_family(safe)
        if safe:
            lines = ["=== GENERALLY RECOMMENDED FOODS (from the SustainHealthyFood "
                     "Ontology, already filtered for the user's diet and allergies) ==="]
            for lbl, detail, _r in safe[:15]:
                lines.append(f"- {lbl}" + (f" | {detail}" if detail else ""))
            context_sections.append("\n".join(lines))
            joined = "\n\n".join(context_sections)

    return joined


_MEAL_TYPE_NAMES: Dict[str, str] = {
    "breakfast": "Breakfast",
    "lunch":     "Lunch",
    "dinner":    "Dinner",
    "snack":     "Snack",
    "pre-workout":  "PreWorkout",
    "pre workout":  "PreWorkout",
    "preworkout":   "PreWorkout",
    "post-workout": "PostWorkout",
    "post workout": "PostWorkout",
    "postworkout":  "PostWorkout",
}

# Generic workout/training vocabulary used to infer PreWorkout/PostWorkout intent
# from phrasing that doesn't literally say "pre-workout"/"post-workout", e.g.
# "what should I eat before and after training" or "foods for muscle recovery
# after intense exercise".
_WORKOUT_CONTEXT_WORDS = ["training", "workout", "exercise", "gym", "run", "sport", "sports"]

_MAX_FOODS_PER_MEAL = 30


def find_meal_types(query: str) -> List[Tuple[str, URIRef]]:
    """Return all (label, URI) meal types mentioned in the query, deduplicated,
    in the order first detected."""
    q = query.lower()
    found: List[Tuple[str, URIRef]] = []
    seen: Set[str] = set()

    def _add(label: str) -> None:
        if label in seen:
            return
        uri = SHFO[label]
        if (uri, RDF.type, None) in ontology_graph or any(
            True for _ in ontology_graph.subjects(SHFO["isIdealFor"], uri)
        ):
            found.append((label, uri))
            seen.add(label)

    for keyword, label in _MEAL_TYPE_NAMES.items():
        if keyword in q:
            _add(label)

    has_workout_context = any(w in q for w in _WORKOUT_CONTEXT_WORDS) or "muscle recovery" in q
    if has_workout_context:
        if "before" in q or "pre" in q:
            _add("PreWorkout")
        if "after" in q or "post" in q or "recovery" in q:
            _add("PostWorkout")

    return found


def find_meal_type(query: str) -> Optional[Tuple[str, URIRef]]:
    """Return the first (label, URI) meal type mentioned in the query, or None."""
    results = find_meal_types(query)
    return results[0] if results else None


def _get_ingredient_summary(food_uri: URIRef) -> str:
    """Return a short ingredient string for recipes, or nutritional classes for food items."""
    # For recipes: collect hasIngredientAmount strings (e.g. "Oats - 40g, Milk - 200ml")
    ingredient_amounts = [
        str(obj)
        for _, _, obj in ontology_graph.triples((food_uri, SHFO["hasIngredientAmount"], None))
        if isinstance(obj, Literal)
    ]
    if ingredient_amounts:
        # Shorten to first 4 ingredients to keep context readable
        preview = ", ".join(ingredient_amounts[:4])
        if len(ingredient_amounts) > 4:
            preview += f" (+{len(ingredient_amounts) - 4} more)"
        return f"Ingredients: {preview}"

    # For plain food items: report key nutritional class memberships
    tags: List[str] = []
    for cls, tag in [
        (SHFO["FoodHighInProtein"],          "high protein"),
        (SHFO["FoodHighInFiber"],            "high fiber"),
        (SHFO["FoodHighInSugar"],            "high sugar"),
        (SHFO["FoodLowInFat"],               "low fat"),
        (SHFO["FoodHighInIron"],             "rich in iron"),
        (SHFO["FermentedFood"],              "fermented"),
        (SHFO["PrebioticFood"],              "prebiotic"),
    ]:
        if (food_uri, RDF.type, cls) in ontology_graph:
            tags.append(tag)
    return f"Properties: {', '.join(tags)}" if tags else ""


def _get_diet_and_allergy_exclusions(user_profile: dict) -> Set[str]:
    """Food/recipe labels the user's diet type or allergies rule out, gathered
    the same "not recommended for" / RestrictedFood buckets build_profile_avoidance_context
    uses to ground an avoidance answer — reused here so meal_suggestion doesn't
    surface foods the profile forbids."""
    excluded: Set[str] = set()

    def _collect(state_uri: URIRef) -> None:
        for prop_name, rel_label in _HEALTH_REC_PROPS.items():
            if rel_label != "not recommended for":
                continue
            for food, _, _ in ontology_graph.triples((None, SHFO[prop_name], state_uri)):
                if isinstance(food, URIRef):
                    excluded.add(get_entity_label(food))
        diet_key = _DIET_URI_TO_KEY.get(normalize_text(str(state_uri).split("#")[-1]))
        if diet_key == "halal":
            _, halal_not_recommended = _get_halal_food_buckets()
            excluded.update(halal_not_recommended)
        elif diet_key in ("vegan", "vegetarian"):
            _, veg_not_recommended = _get_vegan_vegetarian_food_buckets(diet_key)
            excluded.update(veg_not_recommended)
        elif diet_key == "keto":
            _, keto_not_recommended = _get_keyword_food_buckets(_KETO_EXCLUDED_KEYWORDS)
            excluded.update(keto_not_recommended)
            excluded.update(_get_high_carb_food_labels())  # numeric backstop for fruit etc.

    diet = user_profile.get("diet_type", "None") if user_profile else "None"
    if diet not in ("None", ""):
        diet_uri = find_health_state_entity(diet)
        if diet_uri:
            _collect(diet_uri)

    for allergy in (user_profile.get("allergies", []) if user_profile else []):
        allergy_key = normalize_text(f"{allergy} allergy")
        allergy_uri = find_health_state_entity(f"{allergy} allergy")
        if allergy_uri:
            _collect(allergy_uri)
        for extra_name in _ALLERGY_EXTRA_ENTITIES.get(allergy_key, []):
            extra_uri = SHFO[extra_name]
            if (extra_uri, RDF.type, None) in ontology_graph:
                _collect(extra_uri)
        allergy_keywords = _ALLERGY_KEYWORD_MAP.get(allergy_key)
        if allergy_keywords:
            _, allergy_not_recommended = _get_keyword_food_buckets(allergy_keywords)
            excluded.update(allergy_not_recommended)
        molecule_name = _ALLERGY_MOLECULE_MAP.get(allergy_key)
        if molecule_name:
            for food, _, _ in ontology_graph.triples((None, SHFO.containsMolecule, SHFO[molecule_name])):
                if isinstance(food, URIRef):
                    excluded.add(get_entity_label(food))

        # Category-level backstop — see _ALLERGY_FOOD_GROUP_MAP: every food
        # belonging to one of these groups is excluded outright, regardless of
        # whether the ontology separately tagged that individual food
        # RestrictedFood or gave its label an allergy keyword.
        for group_name in _ALLERGY_FOOD_GROUP_MAP.get(allergy_key, []):
            for food, _, _ in ontology_graph.triples((None, SHFO.belongsToFoodGroup, SHFO[group_name])):
                if isinstance(food, URIRef):
                    excluded.add(get_entity_label(food))

        # Ontology data bug guard: a handful of products literally named "X
        # free" (e.g. "Pasta gluten free raw", "Gluten free bread dark") are
        # still wrongly tagged containsMolecule/RestrictedFood for the very
        # allergen they're free of. Rather than hand-edit the 2MB ontology
        # source (SHFO_extension_v2.ttl already has unrelated pending changes
        # of its own), override by name here: a food explicitly labelled
        # "<keyword> free" for this allergy's own keywords is trusted over a
        # conflicting ontology tag.
        if allergy_keywords:
            excluded -= {
                label for label in list(excluded)
                if _label_explicitly_free_of(label, allergy_keywords)
            }

    return excluded


def _ungrounded_profile_conflicts(
    response: str, ontology_context: str, user_profile: Optional[dict]
) -> List[str]:
    """Diet/allergy keywords that appear in the generated ANSWER but nowhere in
    the retrieved ontology context — i.e. the LLM introduced the food from its
    own training knowledge, so the diet/allergy filter (which only sees context
    foods) never had a chance to catch it. Returns the offending keywords."""
    if not user_profile or not response:
        return []
    resp_l = response.lower()
    ctx_l = (ontology_context or "").lower()

    kw_sets: List[Set[str]] = []
    _diet_raw = normalize_text(user_profile.get("diet_type", "") or "")
    diet_key = _DIET_URI_TO_KEY.get(_diet_raw) or _DIET_URI_TO_KEY.get(_diet_raw + "diet")
    if diet_key == "vegan":
        kw_sets.append(_VEGAN_EXCLUDED_KEYWORDS)
    elif diet_key == "vegetarian":
        kw_sets.append(_VEGETARIAN_EXCLUDED_KEYWORDS)
    elif diet_key == "halal":
        kw_sets.append(_HALAL_EXCLUDED_KEYWORDS)
    elif diet_key == "keto":
        kw_sets.append(_KETO_EXCLUDED_KEYWORDS)
    for allergy in user_profile.get("allergies", []) or []:
        s = _ALLERGY_KEYWORD_MAP.get(normalize_text(f"{allergy} allergy"))
        if s:
            kw_sets.append(s)

    hits: Set[str] = set()
    for kws in kw_sets:
        for kw in kws:
            if len(kw) < 3:
                continue
            if re.search(rf"\b{re.escape(kw)}\b", resp_l) and kw not in ctx_l:
                hits.add(kw)
    return sorted(hits)


def _profile_conflict_note(labels: List[str], user_profile: Optional[dict]) -> str:
    """Returns a warning line if any of the given food/ingredient labels
    conflict with the user's diet type or allergies, else "".

    Used by context builders that answer an explicitly-named query (a
    specific recipe, food, or class the user asked for by name) — those
    should still surface the answer, just clearly flagged, rather than
    silently dropping it the way meal_suggestion's exclusion does for
    open-ended "recommend me something" questions.
    """
    if not user_profile or not labels:
        return ""
    excluded = _get_diet_and_allergy_exclusions(user_profile)
    conflicts = sorted(set(labels) & excluded)
    if not conflicts:
        return ""
    constraints = []
    diet = user_profile.get("diet_type", "None")
    if diet not in ("None", ""):
        constraints.append(diet)
    constraints += user_profile.get("allergies", [])
    return (
        f"\n\n⚠ PROFILE WARNING: this conflicts with the user's "
        f"{'/'.join(constraints)} profile — flagged item(s): {', '.join(conflicts)}. "
        f"Clearly warn the user about this in the answer rather than omitting it."
    )


# Matches the note _profile_conflict_note() appends to ontology_context above.
# Used in generate_answer() to (a) keep it out of the "Context passed to LLM"
# trace display — it reads as an internal instruction, not retrieved context —
# and (b) re-show it, worded for the user, after the generated answer instead,
# so the warning survives even if the LLM's own prose glosses over it.
_PROFILE_WARNING_RE = re.compile(
    r"\n\n⚠ PROFILE WARNING: this conflicts with the user's (?P<constraints>.+?) profile — "
    r"flagged item\(s\): (?P<items>.+?)\. Clearly warn the user about this in the answer rather than omitting it\."
)


def _profile_exclusion_note(excluded_labels: Set[str], user_profile: Optional[dict]) -> str:
    """Returns a note if build_health_recommendation_context silently dropped
    any foods from its recommendation buckets because they conflict with the
    user's diet/allergy profile, else "".

    Unlike _profile_conflict_note (which flags a conflict still visible in an
    explicitly-named answer), this covers the exclusion step a few lines above
    in that function, which otherwise removes items with no trace left for
    the user — they never learn their profile was even factored in.
    """
    if not user_profile or not excluded_labels:
        return ""
    constraints = []
    diet = user_profile.get("diet_type", "None")
    if diet not in ("None", ""):
        constraints.append(diet)
    constraints += user_profile.get("allergies", [])
    if not constraints:
        return ""
    return (
        f"\n\n✓ PROFILE FILTER APPLIED: {len(excluded_labels)} food(s) that would otherwise "
        f"appear in the lists above were excluded because they conflict with the user's "
        f"{'/'.join(constraints)} profile. Mention in the answer that some items were "
        f"excluded for this reason."
    )


# Matches the note _profile_exclusion_note() appends to ontology_context above.
# Same role as _PROFILE_WARNING_RE: stripped from the trace's "context passed
# to the LLM" display and re-shown after the answer in generate_answer(),
# since the LLM can't be relied on to surface it from the prompt alone.
_PROFILE_EXCLUSION_RE = re.compile(
    r"\n\n✓ PROFILE FILTER APPLIED: (?P<count>\d+) food\(s\) that would otherwise appear in "
    r"the lists above were excluded because they conflict with the user's (?P<constraints>.+?) "
    r"profile\. Mention in the answer that some items were excluded for this reason\."
)


_HEALTHY_QUESTION_KEYWORDS = [
    "healthy", "healthier", "nutritious", "nutrient-dense", "nutrient dense",
    "wholesome", "clean eating", "low sugar",
]


def _wants_healthy_focus(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _HEALTHY_QUESTION_KEYWORDS)


_MEAL_RANK_BAD_GROUPS = ("pastry", "biscuit", "cake", "confection", "sweet",
                         "sauce", "snack", "sugar", "alcohol", "dessert")
_MEAL_RANK_GOOD_GROUPS = ("vegetable", "fruit", "dairy", "egg", "legume", "pulse",
                          "fish", "grain", "cereal", "nut", "seed", "potato", "meat")


def _meal_food_rank(uri: URIRef) -> float:
    """Higher = a more sensible, healthier pick for a meal-suggestion list.
    Used to rank the (often 300+) `isIdealFor` foods so the top slice shown to
    the LLM is the good stuff, not just the alphabetical head (A = 'apple pie',
    'almond paste tart', ...)."""
    g = ontology_graph
    score = 0.0
    ns = g.value(uri, SHFO["hasNutrientScore"])
    try:
        if ns is not None:
            score += float(ns)
    except (TypeError, ValueError):
        pass
    for cls, delta in (
        ("FoodHighInSugar", -6.0), ("FoodHighInFat", -2.0), ("FoodLowInFiber", -1.0),
        ("FoodHighInFiber", 2.0), ("FoodHighInProtein", 2.0),
        ("PrebioticFood", 1.0), ("FermentedFood", 1.0),
    ):
        if (uri, RDF.type, SHFO[cls]) in g:
            score += delta
    sugar = g.value(uri, SHFO["hasAmountOfSugar"])
    try:
        if sugar is not None and float(sugar) > 15:
            score -= 4.0
    except (TypeError, ValueError):
        pass
    grp_uri = g.value(uri, SHFO["belongsToFoodGroup"])
    grp = normalize_text(get_entity_label(grp_uri)) if grp_uri is not None else ""
    if any(b in grp for b in _MEAL_RANK_BAD_GROUPS):
        score -= 4.0
    if any(w in grp for w in ("meat", "fish")):
        score += 1.0
    elif any(w in grp for w in _MEAL_RANK_GOOD_GROUPS):
        score += 2.0
    lbl = normalize_text(get_entity_label(uri))
    if re.search(r"\braw\b", lbl):
        score -= 12.0  # never recommend raw meat/fish/egg to eat as-is
    if any(w in lbl for w in ("liver", "pate", "kidney", "tongue", "brain",
                              "offal", "tartare", "brawn")):
        score -= 12.0  # organ meats — not a sensible default recommendation
    if any(w in lbl for w in ("bacon", "sausage", "luncheon meat", "corned beef",
                              "smoke dried", "salted cooked")):
        score -= 6.0  # processed meat
    return score


@lru_cache(maxsize=1)
def _fallback_meal_pool() -> tuple:
    """Recognisable whole foods with real protein or carbohydrate content —
    the pool used to top up a meal section the sparse `isIdealFor` tags left
    almost empty (e.g. PostWorkout for a vegan). Cached; profile filtering is
    applied per-call in _fallback_meal_foods."""
    g = ontology_graph
    pool = []
    for label, uri in FOOD_CANDIDATES:
        low = label.lower()
        if re.search(r"\braw\b", low) or any(
            w in low for w in ("liver", "pate", "kidney", "tongue", "brain", "offal",
                               "tartare", "brawn", "bacon", "sausage", "luncheon")
        ):
            continue
        grp_uri = g.value(uri, SHFO["belongsToFoodGroup"])
        grp = normalize_text(get_entity_label(grp_uri)) if grp_uri is not None else ""
        if not any(w in grp for w in ("legume", "pulse", "dairy", "milk", "cheese",
                                      "egg", "fish", "meat", "nut", "seed", "grain",
                                      "cereal", "fruit", "potato", "starchy")):
            continue
        try:
            prot = float(g.value(uri, SHFO["hasAmountOfProtein"]) or 0)
            carb = float(g.value(uri, SHFO["hasAmountOfCarbohydrates"]) or 0)
        except (TypeError, ValueError):
            prot = carb = 0.0
        if prot < 8 and carb < 15:
            continue
        pool.append((label, uri))
    return tuple(pool)


def _dedup_food_family(items: List[Tuple[str, str, float]], per_family: int = 2
                       ) -> List[Tuple[str, str, float]]:
    """Keep at most `per_family` near-duplicate variants of the same food
    (e.g. 'Liver ox prepared' / 'Liver ox raw' / 'Liver ox boiled ...') so one
    ingredient family can't fill the whole list. Assumes items are already in
    priority order."""
    seen: Dict[str, int] = {}
    out: List[Tuple[str, str, float]] = []
    for it in items:
        toks = re.findall(r"[a-z0-9]+", it[0].lower())
        key = " ".join(toks[:2])
        if seen.get(key, 0) >= per_family:
            continue
        seen[key] = seen.get(key, 0) + 1
        out.append(it)
    return out


def _fallback_meal_foods(excluded_labels: Set[str], need: int) -> List[Tuple[str, str, float]]:
    """Profile-safe whole foods to top up an all-but-empty meal section."""
    out: List[Tuple[str, str, float]] = []
    for label, uri in _fallback_meal_pool():
        if label in excluded_labels:
            continue
        out.append((label, _get_ingredient_summary(uri), _meal_food_rank(uri)))
    out.sort(key=lambda x: (-x[2], x[0]))
    return _dedup_food_family(out)[: max(0, need)]


def build_meal_suggestion_context(
    question: str, entity_hint: str = "", user_profile: Optional[dict] = None
) -> Optional[str]:
    results = find_meal_types(entity_hint) if entity_hint else []
    if not results:
        results = find_meal_types(question)
    if not results:
        return None

    excluded_labels = _get_diet_and_allergy_exclusions(user_profile) if user_profile else set()
    healthy_focus = _wants_healthy_focus(question)

    sections: List[str] = []
    for meal_label, meal_uri in results:
        items: List[Tuple[str, str]] = []
        for entity_uri, _, _ in ontology_graph.triples((None, SHFO["isIdealFor"], meal_uri)):
            if not isinstance(entity_uri, URIRef):
                continue
            is_recipe = (entity_uri, RDF.type, SHFO["Recipe"]) in ontology_graph
            # isIdealFor's domain is Food and Recipe — include both.
            is_food_or_recipe = is_recipe or (entity_uri, RDF.type, SHFO["Food"]) in ontology_graph
            if not is_food_or_recipe:
                continue
            label = get_entity_label(entity_uri)
            if label in excluded_labels:
                continue
            # never suggest raw meat/fish/poultry/egg or organ/processed meat as
            # a thing to eat
            low = label.lower()
            if re.search(r"\braw\b", low) and (
                (entity_uri, RDF.type, SHFO["Food"]) in ontology_graph
            ) and any(w in normalize_text(get_entity_label(
                ontology_graph.value(entity_uri, SHFO["belongsToFoodGroup"]) or URIRef("urn:x")
            )) for w in ("meat", "fish", "egg")):
                continue
            if any(w in low for w in ("liver", "pate", "kidney", "tongue", "brain",
                                      "offal", "tartare", "brawn", "luncheon meat")):
                continue
            # RestrictedFood/notRecommendedFor (what excluded_labels is built
            # from) is only ever asserted on Food individuals in this
            # ontology, never on Recipe individuals — so a recipe made from an
            # excluded ingredient (e.g. "Biscuit w nuts and chocolate" containing
            # food_nuts_mixed_unsalted) would otherwise slip through the check
            # above untouched. Check the recipe's own ingredient list too.
            if is_recipe and excluded_labels:
                ingredient_labels = {ing["label"] for ing in get_recipe_ingredients(entity_uri)}
                if ingredient_labels & excluded_labels:
                    continue
            detail = _get_ingredient_summary(entity_uri)
            items.append((label, detail, _meal_food_rank(entity_uri)))

        # Top up an all-but-empty section with profile-safe whole foods rather
        # than telling the user "nothing found" (common for PostWorkout, whose
        # tag set is tiny and meat/dairy-heavy).
        fallback_used = False
        if len(items) < 3:
            extra = _fallback_meal_foods(excluded_labels, 8 - len(items))
            have = {i[0] for i in items}
            for e in extra:
                if e[0] not in have:
                    items.append(e)
                    fallback_used = True

        if not items:
            reason = " that fit your diet/allergy profile" if excluded_labels else ""
            sections.append(f"No foods or recipes found in the SustainHealthyFood Ontology for '{meal_label}'{reason}.")
            continue

        # Always rank best-first (healthiest / most sensible pick), then take the
        # top slice — never alphabetical, which only ever showed the LLM items
        # starting A-C out of 300+.
        items_sorted = _dedup_food_family(sorted(set(items), key=lambda x: (-x[2], x[0])))
        total = len(items_sorted)
        shown = items_sorted[:_MAX_FOODS_PER_MEAL]

        lines = [
            f"MEAL SUGGESTION CONTEXT",
            f"Meal type: {meal_label}",
            f"{total} foods/recipes in the SustainHealthyFood Ontology are marked ideal for "
            f"{meal_label}. The {len(shown)} below are the best picks, already filtered for "
            f"the user's diet/allergies and ordered best-first:",
        ]
        if fallback_used:
            lines.append(
                "(Few foods are tagged specifically for this timing that fit the profile, "
                "so profile-safe high-protein / high-fibre foods have been added.)"
            )
        if healthy_focus:
            lines.append(
                "The user asked for HEALTHY options — the list is already ordered so the "
                "healthiest choices come first; pick from the top."
            )
        for item_label, detail, _rank in shown:
            line = f"- {item_label}"
            if detail:
                line += f" | {detail}"
            lines.append(line)
        if total > _MAX_FOODS_PER_MEAL:
            lines.append(f"  ... and {total - _MAX_FOODS_PER_MEAL} more items in this category.")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def is_meal_suggestion_question(question: str) -> bool:
    q = question.lower()
    meal_keywords = [
        "breakfast", "lunch", "dinner", "snack",
        "pre-workout", "pre workout", "preworkout",
        "post-workout", "post workout", "postworkout",
        "training", "workout", "exercise", "muscle recovery",
    ]
    intent_keywords = [
        "what should i eat", "what to eat", "what can i eat", "what can i have",
        "food for", "foods for", "ideal for", "good for",
        "suggest", "ideas for", "options for", "recommend",
        "foods help", "food help", "help with", "help me",
    ]
    has_meal = any(m in q for m in meal_keywords)
    has_intent = any(i in q for i in intent_keywords)
    return has_meal and has_intent


def is_context_relevant(question: str, docs) -> bool:
    """Return True if any retrieved doc is topically related to the question."""
    if not docs:
        return False
    q_tokens = {t for t in re.findall(r"[a-z0-9]+", question.lower()) if len(t) >= 4}
    noise = {"what", "tell", "about", "does", "have", "with", "from", "that", "this", "some", "more", "show", "give", "list"}
    q_tokens -= noise
    if not q_tokens:
        return True
    for doc in docs[:3]:
        combined = (doc.page_content + " " + doc.metadata.get("id", "")).lower()
        if any(tok in combined for tok in q_tokens):
            return True
    return False


def build_chat_history_text(n: int = 3) -> str:
    history = st.session_state.get("chat_history", [])
    if not history:
        return "No previous conversation."
    lines = []
    for q, a in history[-n:]:
        lines.append(f"User: {q}")
        a_short = a[:400] + "..." if len(a) > 400 else a
        lines.append(f"Assistant: {a_short}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Food suitability — "can I eat X / is X recommended for me". One stable route
# for questions naming a single food/dish and asking whether THIS user should
# eat it. Without it these scattered across health_recommendation (which just
# dumped the food's own recommendedFor/notRecommendedFor condition list),
# nutrient_lookup and profile_reasoning depending on tiny wording changes,
# producing contradictory answers. The context here is already assessed
# against the user's diet / allergies / life stage so the answer is a direct
# verdict, not an unrelated condition list.
# ---------------------------------------------------------------------------

_FOOD_SUITABILITY_VERBS = (
    "can i eat", "can i have", "could i eat", "could i have", "may i eat",
    "am i allowed", "is it ok", "is it okay", "is it fine", "is it safe",
    "is it recommended", "is it advisable", "is it a good idea", "should i eat",
    "should i have", "should i avoid", "can i still eat", "is this ok",
    "is this okay", "is that ok", "is it good for me", "is it bad for me",
    "would it be ok", "ok for me to eat", "safe for me to eat",
)

# "Is peanut butter a good protein source for me?" — a yes/no question about
# one named food, judged for this user, but with no "eat" verb, so the phrase
# list above missed it. The router called it health_recommendation, that found
# no health state, and it fell to llm_general_knowledge: no context, no
# allergy check, peanut butter recommended to a nut-allergic user.
_SUIT_IS_X_GOOD_FOR_ME_RE = re.compile(
    r"^\s*(?:is|are)\s+(?P<food>.+?)\s+(?:a\s+|an\s+)?(?:really\s+|very\s+)?"
    r"(?:good|bad|ok|okay|fine|safe|healthy|suitable|right|great|decent|smart)\b"
    r".*\bfor\s+me\b",
    re.I,
)

_SUIT_HIGH_CLASSES = {
    "FoodHighInFat": "high in fat",
    "FoodHighInSaturatedFat": "high in saturated fat",
    "FoodHighInSugar": "high in sugar",
    "FoodHighInSalt": "high in salt",
    "FoodHighInCarbs": "high in carbohydrates",
    "UltraProcessedFood": "ultra-processed",
}


def is_food_suitability_question(question: str) -> bool:
    """True for 'can I eat X / is X OK for me' phrasing. Entity resolution is
    left to build_food_suitability_context — the route only activates when
    that returns a real ontology food/recipe (its None result is the gate)."""
    q = question.lower().strip()
    if not any(v in q for v in _FOOD_SUITABILITY_VERBS) and not _SUIT_IS_X_GOOD_FOR_ME_RE.search(q):
        return False
    # "what can I eat for breakfast / for anemia" is a meal / health question.
    if is_meal_suggestion_question(question):
        return False
    if re.search(
        r"\b(for breakfast|for lunch|for dinner|as a snack|before training|"
        r"after training|for anemia|for diabetes|for my stress)\b", q
    ):
        return False
    return True


_SUIT_STRIP_RE = re.compile(
    r"\b(can|could|may|should|would|am|is|are|it|this|that|i|to|eat|eating|have|"
    r"having|consume|still|allowed|ok|okay|fine|safe|recommended|advisable|good|"
    r"bad|idea|for|me|be|want|wanting|going|planning|like|really|a|an|the|any|"
    r"some|my|profile|based|on)\b"
)


def _suit_probe(text: str) -> str:
    """Strip the 'can I eat ... is it recommended' scaffolding so what's left is
    just the food phrase for matching."""
    cleaned = _SUIT_STRIP_RE.sub(" ", text.lower())
    return re.sub(r"[^a-z0-9 ]+", " ", cleaned).strip()


# Precise extraction of the food phrase directly from an "eat" verb, tried
# BEFORE the generic word-strip above. Needed because _suit_probe strips the
# scaffolding from the WHOLE sentence, so a question that names the allergy
# too (e.g. "I am allergic to dairy, can I eat yogurt?") leaves both "dairy"
# and "yogurt" in the cleaned probe — and "dairy" alone happens to
# substring-match "Ice cream dairy av" at the same score "yogurt" matches
# "Yoghurt full fat" at, so the wrong leftover word could win. Anchoring on
# the verb picks out the actual object of "eat" instead of guessing among
# every leftover noun.
_SUIT_AFTER_VERB_RE = re.compile(
    r"\b(?:can|could|may|should|would)\s+i\s+(?:still\s+)?(?:eat|have|consume)\s+"
    r"(.+?)(?=[.!?]|$)",
    re.I,
)
_SUIT_WANT_TO_EAT_RE = re.compile(
    r"\b(?:want|wanting|going|plan|planning)\s+to\s+eat\s+(.+?)(?=[.!?]|$)",
    re.I,
)


def _extract_suit_food_phrase(question: str) -> str:
    """Best-effort extraction of just the named food from a suitability
    question (e.g. "...can I eat yogurt?" -> "yogurt"). Returns "" if no
    clean "eat X" segment is found — caller then falls back to _suit_probe."""
    for pattern in (_SUIT_AFTER_VERB_RE, _SUIT_WANT_TO_EAT_RE, _SUIT_IS_X_GOOD_FOR_ME_RE):
        m = pattern.search(question)
        if m:
            phrase = m.group(1).strip(" ?.,!")
            phrase = re.sub(r"^(a|an|the|some|any)\s+", "", phrase, flags=re.I)
            if phrase and phrase.lower() not in ("it", "this", "that", "they", "these", "those"):
                return phrase
    return ""


def _pick_best_suitability_match(probe: str, matches: List[Tuple[float, str, URIRef]]):
    """Among find_food_candidates matches tied at the top score, prefer the
    plain food over one where the query word is just a minor ingredient
    mention buried in a compound name.

    find_food_candidates breaks ties alphabetically by label, which for a
    generic query like "yogurt" was picking "Salad dressing 20% oil w
    yoghurt" over any of the six actual "Yoghurt ..." individuals purely
    because 'S' < 'Y' — same top score (85, phrase-in-label), wrong food.
    Reordering: a label whose FIRST word is the query term wins over one
    where it only appears later in the name, then shortest label wins."""
    top_score = matches[0][0]
    tied = [m for m in matches if m[0] == top_score]
    if len(tied) == 1:
        return tied[0]
    probe_norm = normalize_text(probe)

    def _rank(m):
        _, label, _ = m
        label_words = re.findall(r"[a-z0-9]+", label.lower().replace("yoghurt", "yogurt"))
        first_word_norm = normalize_text(label_words[0]) if label_words else ""
        is_primary_term = 0 if (
            first_word_norm and
            (first_word_norm == probe_norm or probe_norm.startswith(first_word_norm))
        ) else 1
        return (is_primary_term, len(label))

    return min(tied, key=_rank)


def _resolve_suitability_entity(question: str, entity_hint: str = ""):
    """Return (uri, kind) for the food/recipe the question names, or None."""
    precise_food = _extract_suit_food_phrase(question)
    probe_food = _suit_probe(question)
    # precise_food (anchored on the "eat" verb) is tried before the
    # generic word-strip probe — see _extract_suit_food_phrase for why.
    for probe in ([entity_hint] if entity_hint else []) + [precise_food, probe_food, question]:
        if not probe:
            continue
        matches = find_food_candidates(probe)
        if matches and matches[0][0] >= 70:
            best = _pick_best_suitability_match(probe, matches)
            return best[2], "food"
    for probe in (precise_food, probe_food, question):
        if not probe:
            continue
        food_uri = find_food_entity(probe)
        if food_uri is not None:
            return food_uri, "food"
        recipe_uri = find_recipe_entity(probe)
        if recipe_uri is not None:
            return recipe_uri, "recipe"
    return None


def _is_composite_food(label: str, groups: List[str], kind: str = "food") -> bool:
    """A recipe or prepared/mixed food the ontology can't decompose — it has no
    ingredient list, so the absence of an allergen tag proves nothing."""
    _grp = " ".join(groups).lower()
    _name_l = label.lower()
    return kind == "recipe" or any(
        w in _grp for w in ("pastry", "biscuit", "cake", "dessert", "bread",
                            "pie", "confection", "snack", "prepared",
                            "composite", "sauce", "ready", "soup", "pizza",
                            "sandwich")
    ) or any(
        w in _name_l for w in ("cookie", "biscuit", "cake", "pastry", "pie",
                               "muffin", "brownie", "pancake", "waffle",
                               "doughnut", "donut", "tart", "pudding",
                               "dessert", "chocolate bar", "cereal bar",
                               "sandwich", "pizza", "quiche", "sauce",
                               "dressing", "gravy", "batter", "breaded")
    )


def build_food_suitability_context(
    question: str, entity_hint: str = "", user_profile: Optional[dict] = None
) -> Optional[str]:
    """Ontology context for a 'can I eat X' question: the food's own facts plus
    a PROFILE ASSESSMENT (CONFLICT / CAUTION bullets + verdict) computed against
    the user's diet, allergies and life stage. Returns None when no ontology
    food/recipe matches — the caller then falls through to the next route."""
    resolved = _resolve_suitability_entity(question, entity_hint)
    if resolved is None:
        return None
    uri, kind = resolved
    label = get_entity_label(uri)

    lines = [
        "FOOD SUITABILITY CONTEXT",
        f"Food/dish asked about: {label}",
        f"Ontology type: {kind}",
    ]

    groups = get_food_groups(uri)
    if groups:
        lines.append(f"Food group(s): {', '.join(groups)}")

    class_flags = sorted({
        _SUIT_HIGH_CLASSES[str(cls).split('#')[-1]]
        for _, _, cls in ontology_graph.triples((uri, RDF.type, None))
        if str(cls).split('#')[-1] in _SUIT_HIGH_CLASSES
    })
    if class_flags:
        lines.append(f"Nutritional flags from the ontology: {', '.join(class_flags)}")

    molecules = sorted({
        get_entity_label(o)
        for o in ontology_graph.objects(uri, SHFO.containsMolecule)
        if isinstance(o, URIRef)
    })
    if molecules:
        lines.append(f"Contains molecule(s): {', '.join(molecules)}")

    ingredient_labels: List[str] = []
    if kind == "recipe":
        ingredient_labels = [ing["label"] for ing in get_recipe_ingredients(uri)]
        if ingredient_labels:
            lines.append(f"Recipe ingredients: {', '.join(ingredient_labels)}")

    _, nutrient_rows = get_all_nutrients_for_food(uri)
    nmap = {n: v for n, v in nutrient_rows}
    key_nutr = [
        f"{n} {nmap[n]}"
        for n in ("Calories", "Sugar", "Fat", "Saturated Fatty Acids",
                  "Fiber", "Protein", "Nutrient Score")
        if n in nmap
    ]
    if key_nutr:
        lines.append(f"Key nutrients (per 100 g): {'; '.join(key_nutr)}")

    pos_targets: Set[str] = set()
    neg_targets: Set[str] = set()
    neg_uris: Set[URIRef] = set()
    for prop_name, rel in _HEALTH_REC_PROPS.items():
        for o in ontology_graph.objects(uri, SHFO[prop_name]):
            if not isinstance(o, URIRef):
                continue
            if rel == "not recommended for":
                neg_targets.add(get_entity_label(o))
                neg_uris.add(o)
            else:
                pos_targets.add(get_entity_label(o))

    # Only surface the recommend / not-recommend contexts that match THIS user's
    # own diet, life stage or stated concerns — the full list (e.g. a cookie is
    # "not recommended for" ~10 unrelated states) is just noise the weak router
    # LLM tends to dump back verbatim.
    _profile_raw: List[str] = []
    if user_profile:
        _profile_raw.append(user_profile.get("diet_type", "") or "")
        _profile_raw.append(user_profile.get("life_stage", "") or "")
        _profile_raw += user_profile.get("allergies", []) or []
        _profile_raw += re.split(r"[,/;]+", user_profile.get("user_concerns", "") or "")
    _profile_terms = {
        normalize_text(t) for t in _profile_raw
        if normalize_text(t) and normalize_text(t) not in ("none", "generaladult", "na")
    }

    def _relevant(names: Set[str]) -> List[str]:
        return sorted(
            n for n in names
            if any(pt and (pt in normalize_text(n) or normalize_text(n) in pt)
                   for pt in _profile_terms)
        )

    rel_pos, rel_neg = _relevant(pos_targets), _relevant(neg_targets)
    if rel_pos:
        lines.append(f"Ontology marks it recommended for your profile: {', '.join(rel_pos)}")
    if rel_neg:
        lines.append(f"Ontology marks it NOT recommended for your profile: {', '.join(rel_neg)}")
    other_neg = len(neg_targets) - len(rel_neg)
    if other_neg > 0:
        lines.append(
            f"(It is also tagged not-recommended-for {other_neg} other health/diet contexts that "
            f"do not match this user — ignore those.)"
        )

    conflicts: List[str] = []
    cautions: List[str] = []
    if user_profile:
        diet = user_profile.get("diet_type") or "None"
        allergies = user_profile.get("allergies", []) or []
        life_stage = user_profile.get("life_stage", "General Adult")

        excluded = _get_diet_and_allergy_exclusions(user_profile)
        if label in excluded:
            conflicts.append(f"the ontology puts {label} on your diet/allergy exclusion list")
        if kind == "recipe":
            bad_ings = sorted(set(ingredient_labels) & excluded)
            if bad_ings:
                conflicts.append(f"recipe ingredient(s) conflict with your profile: {', '.join(bad_ings)}")

        resolved_allergies: Set[str] = set()   # allergy names the ontology could judge
        molecules_norm = {normalize_text(m) for m in molecules}

        if diet not in ("None", ""):
            du = find_health_state_entity(diet)
            if du is not None and du in neg_uris:
                conflicts.append(
                    f"the ontology marks {label} as not recommended for your {diet} diet"
                )
        for allergy in allergies:
            allergy_uris = set()
            au = find_health_state_entity(f"{allergy} allergy")
            if au is not None:
                allergy_uris.add(au)
            for extra in _ALLERGY_EXTRA_ENTITIES.get(normalize_text(f"{allergy} allergy"), []):
                allergy_uris.add(SHFO[extra])
            if allergy_uris & neg_uris:
                conflicts.append(
                    f"the ontology restricts {label} for your {allergy} allergy/intolerance"
                )
                resolved_allergies.add(allergy)

            kws = _ALLERGY_KEYWORD_MAP.get(normalize_text(f"{allergy} allergy"), set())
            haystack = " ".join([label] + molecules + ingredient_labels).lower()
            hit = sorted({
                kw for kw in kws
                if len(kw) >= 3 and re.search(rf"\b{re.escape(kw)}\b", haystack)
            })
            if hit:
                conflicts.append(f"contains {allergy.lower()} allergen(s): {', '.join(hit)}")
                resolved_allergies.add(allergy)

            # A molecule tag naming this allergen (in practice only Gluten) is a
            # positive confirmation, so that allergen no longer needs the
            # can't-confirm caution below.
            mol = _ALLERGY_MOLECULE_MAP.get(normalize_text(f"{allergy} allergy"))
            if mol and normalize_text(mol) in molecules_norm:
                resolved_allergies.add(allergy)

        # Allergen-uncertainty: for any allergy we could NOT judge from the
        # ontology. Food individuals carry no hasIngredient and containsMolecule
        # is only ever Gluten, so the ontology genuinely cannot confirm a
        # composite food is free of milk / egg / nuts / soy — say so rather
        # than imply it is safe.
        unresolved = [a for a in allergies if a not in resolved_allergies]
        composite = _is_composite_food(label, groups, kind)
        if unresolved and kind == "food" and composite:
            # Safety-critical: an allergy plus a composite/prepared food the
            # ontology can't decompose. Deliberately NOT handed to the LLM as
            # a "use your general knowledge" judgment call — tested with the
            # app's llama3.2 and it guessed backwards (claimed a chocolate
            # chip cookie "likely does not contain dairy"), the same
            # composition-hallucination failure mode already seen with
            # llama3.1:8b as judge (see memory: almonds→soy). Butter/milk/egg/
            # nuts are routine in these items, so treat it as
            # not-recommended-until-checked instead. Whole single foods
            # (fruit, veg, plain grain/meat/fish) are NOT flagged here — an
            # apple obviously has no dairy.
            conflicts.append(
                f"this is a composite/prepared food and the ontology has no ingredient list for "
                f"it, so it cannot be confirmed free of your {', '.join(a.lower() for a in unresolved)} "
                f"allergen(s) — such items routinely contain milk, butter, egg or nuts; do not eat it "
                f"until the product label is checked"
            )

        if life_stage in ("Pregnant", "Lactating", "Pedriatic") and class_flags:
            cautions.append(
                f"you are in the {life_stage} life stage and this food is {', '.join(class_flags)}"
            )
        elif class_flags and not conflicts:
            cautions.append(f"this food is {', '.join(class_flags)} — best kept occasional")

    conflicts = sorted(set(conflicts))
    cautions = sorted(set(cautions))
    if conflicts:
        verdict = "NOT RECOMMENDED for this user"
        verdict_sentence = f"No — {label} isn't recommended for you."
    elif cautions:
        verdict = "OK IN MODERATION for this user, with caveats"
        verdict_sentence = f"You can have {label} occasionally, but with some caveats."
    else:
        verdict = "NO PROFILE CONFLICT found for this user"
        verdict_sentence = f"Yes — {label} fits your profile."

    # Machine-read by generate_answer and prepended verbatim to the answer, so
    # the weak router LLM can't soften "not recommended" into "you can
    # sometimes" — tested and confirmed necessary (see food_suitability route
    # notes: letting the LLM freely judge unresolved allergens produced a
    # backwards answer, so the verdict here is deterministic, not LLM-guessed).
    lines.append("")
    lines.append(f"VERDICT_SENTENCE: {verdict_sentence}")
    lines.append("")
    lines.append(f"PROFILE ASSESSMENT — {verdict}")
    for c in conflicts:
        lines.append(f"- CONFLICT: {c}")
    for c in cautions:
        lines.append(f"- CAUTION: {c}")
    if not conflicts and not cautions:
        lines.append("- No diet, allergy or life-stage conflict was found in the ontology for this food.")

    return "\n".join(lines)


# ── False premise ("since almonds contain soy, what can I use instead?") ─────
# The question states a composition "fact" as given, and every other route
# takes it at face value: the router sent this one to health_recommendation for
# Soy_Allergy, which listed soy foods and had llama3.2 restate "since almonds
# contain soy" as if the ontology said so. The claim is checked here against the
# same allergen exclusion data the profile filters use, and a false one gets a
# deterministic correction (no LLM call). Claims that hold, or that the ontology
# can't settle (composite foods have no ingredient list), fall through.
_PREMISE_ALLERGEN_TO_ALLERGY: Dict[str, str] = {
    "soy": "Soy", "soya": "Soy",
    "dairy": "Dairy", "milk": "Dairy", "lactose": "Dairy",
    "egg": "Eggs", "eggs": "Eggs",
    "nut": "Nuts", "nuts": "Nuts", "tree nuts": "Nuts", "peanut": "Nuts", "peanuts": "Nuts",
    "gluten": "Gluten", "wheat": "Gluten",
}
_PREMISE_CLAIM_RE = re.compile(
    r"(?P<food>[a-z][a-z \-]*?)\s+"
    r"(?P<verb>contains?|has|have|(?:is|are)\s+made\s+(?:with|from|of)|(?:is|are)\s+full\s+of)\s+"
    r"(?:a\s+lot\s+of\s+|lots\s+of\s+|some\s+)?"
    r"(?P<allergen>tree\s+nuts|peanuts?|nuts?|soya?|dairy|milk|lactose|eggs?|gluten|wheat)\b",
    re.I,
)
# "Does goulash contain dairy?" — asked outright, so a true claim gets a direct
# yes instead of falling through.
_PREMISE_QUESTION_RE = re.compile(r"^\s*(?:does|do|did|is there|are there|can)\b", re.I)
# Everything up to the last lead-in word ("since", "because", "I heard that", ...)
# is discourse, not the food name.
_PREMISE_LEAD_IN_RE = re.compile(
    r"^.*\b(?:since|because|as|given\s+that|considering|now\s+that|heard|read|know|"
    r"learned|that|do|does|if)\s+",
    re.I,
)
_PREMISE_NOT_A_FOOD = {
    "it", "this", "that", "they", "these", "those", "which", "what", "food", "foods",
    "something", "everything", "anything", "i", "you", "he", "she", "we", "my diet",
}
# Prepared items _is_composite_food doesn't flag but whose recipe varies (bread
# can contain soy flour, "almond paste" can contain egg) — the ontology has no
# ingredient list for them, so they can't refute a claim.
_PREMISE_PREPARED_WORDS = {
    "bread", "crumbs", "cracker", "crackers", "mix", "paste", "imitation", "imitat",
    "spread", "bar", "cereal", "flour", "drink", "bun", "roll",
}
_PREMISE_ALLERGY_NOUN = {"Soy": "soy", "Dairy": "dairy", "Eggs": "egg", "Nuts": "nut", "Gluten": "gluten"}


@lru_cache(maxsize=None)
def _allergen_food_labels(allergy: str) -> frozenset:
    """Every food label the profile filters would exclude for this app allergy."""
    return frozenset(_get_diet_and_allergy_exclusions({"allergies": [allergy]}))


def _match_premise_foods(phrase: str) -> List[Tuple[str, URIRef]]:
    """Foods whose label contains the phrase as whole words, singular or plural
    ("almonds" -> "Almonds w skin unsalted", "Almond paste w egg"). Stricter than
    find_food_candidates, whose substring scoring matches "it" inside "biscuit"."""
    stems = {phrase}
    if phrase.endswith("es"):
        stems.add(phrase[:-2])
    if phrase.endswith("s"):
        stems.add(phrase[:-1])
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(s) for s in stems if len(s) >= 3) + r")(?:s|es)?\b"
    )
    matches = [(label, uri) for label, uri in FOOD_CANDIDATES if pattern.search(label.lower())]
    # Prefer items that ARE the food ("Almonds blanched salted") over ones that
    # merely mention it ("Stollen w almond paste", "Apple pie") — the claim is
    # about the food itself.
    primary = [(label, uri) for label, uri in matches if pattern.match(label.lower())]
    return primary or matches


def check_false_premise(question: str, user_profile: Optional[dict] = None) -> Optional[dict]:
    """If the question asserts "<food> contains <allergen>" and the ontology
    contradicts it, return {"context", "answer", "food", "allergen"}; else None."""
    m = _PREMISE_CLAIM_RE.search(question.lower())
    if not m:
        return None
    phrase = _PREMISE_LEAD_IN_RE.sub("", m.group("food")).strip(" -")
    phrase = re.sub(r"^(?:the|a|an|all|my|some)\s+", "", phrase)
    phrase = " ".join(phrase.split()[-4:])
    allergen = re.sub(r"\s+", " ", m.group("allergen"))
    allergy = _PREMISE_ALLERGEN_TO_ALLERGY.get(allergen)
    if not allergy or len(phrase) < 3 or phrase in _PREMISE_NOT_A_FOOD:
        return None

    allergy_l = allergy.lower()
    allergy_noun = _PREMISE_ALLERGY_NOUN[allergy]
    a_allergy = f"{'an' if allergy_noun[0] in 'aeiou' else 'a'} {allergy_noun} allergy"
    plural = m.group("verb").lower() in ("contain", "have", "are made with", "are made from",
                                         "are made of", "are full of")
    # Asked outright: "does X contain Y?" — or any "...contain Y?" not
    # introduced by "since/because" (so "oes goulash contain dairy?" counts).
    asked = bool(_PREMISE_QUESTION_RE.match(question)) or (
        question.rstrip().endswith("?")
        and not re.search(r"\b(?:since|because|as|given that|considering|now that)\b",
                          question.lower()[:m.end()])
    )
    user_allergies = [a.lower() for a in (user_profile or {}).get("allergies", []) or []]
    allergen_foods = _allergen_food_labels(allergy)

    # Longest word-suffix that names something, so a stray leading word
    # ("oes goulash" from a dropped "d") doesn't block the match.
    words = phrase.split()
    suffixes = [" ".join(words[i:]) for i in range(len(words)) if len(" ".join(words[i:])) >= 3]
    for candidate in suffixes:
        recipe = _match_premise_recipe(candidate)
        if recipe is not None:
            return _recipe_allergen_answer(recipe, allergen, allergy_noun, asked, allergen_foods,
                                           allergy_l in user_allergies)
    matches = []
    for candidate in suffixes:
        if candidate in _PREMISE_NOT_A_FOOD:
            continue
        matches = _match_premise_foods(candidate)
        if matches:
            phrase = candidate
            break
    if not matches:
        return None
    # Judged on whole foods only: "Apple pie" containing gluten says nothing
    # about apples, and a prepared item's missing allergen tag proves nothing.
    whole = [
        (label, uri) for label, uri in matches
        if not _is_composite_food(label, get_food_groups(uri))
        and not set(re.findall(r"[a-z]+", label.lower())) & _PREMISE_PREPARED_WORDS
    ]
    if not whole:
        return None   # only composite items: the ontology can't rule it out
    food_title = phrase[:1].upper() + phrase[1:]

    hits = [label for label, _ in whole if label in allergen_foods]
    if hits:
        # The claim holds. Stated as a premise ("since tofu contains soy, ...")
        # the normal routes answer the real question; asked outright ("does
        # tofu contain soy?") it gets a direct yes.
        if not asked:
            return None
        answer = [
            f"Yes — {phrase} {'contain' if plural else 'contains'} {allergen}.",
            "",
            f"In the SustainHealthyFood Ontology, these matching items are restricted for "
            f"{a_allergy} or listed as {allergen} foods: {', '.join(hits[:5])}.",
        ]
        if allergy_l in user_allergies:
            answer.append(f"\nThat conflicts with your {allergy_noun} allergy — avoid {phrase}.")
        return {
            "context": "\n".join([
                "ALLERGEN CHECK",
                f"Question: does {phrase} contain {allergen}?",
                f"Matching ontology foods (whole foods): {', '.join(l for l, _ in whole)}",
                f"Restricted for {a_allergy} / {allergen} foods: {', '.join(hits)}",
            ]),
            "answer": "\n".join(answer),
            "food": food_title,
            "allergen": allergen,
            "route": "allergen_check",
            "uri": "",
        }

    restricted_for = sorted({
        get_entity_label(o).replace("_", " ")
        for _, uri in whole
        for o in ontology_graph.objects(uri, SHFO.RestrictedFood)
        if isinstance(o, URIRef)
    })
    examples = ", ".join(label for label, _ in whole[:3])

    context_lines = [
        "FALSE PREMISE CHECK",
        f"Claim in question: {phrase} contain(s) {allergen}",
        f"Matching ontology foods (whole foods): {', '.join(label for label, _ in whole)}",
        f"None of them is restricted for {a_allergy} or carries a {allergen} "
        f"keyword/molecule tag in the ontology.",
    ]
    if restricted_for:
        context_lines.append(f"They are restricted for: {', '.join(restricted_for)}")

    answer = [
        f"No — {phrase} {'don' if plural else 'doesn'}'t contain {allergen}"
        + ("." if asked else ", so the premise of the question isn't right."),
        "",
        f"In the SustainHealthyFood Ontology, none of the matching items ({examples}) is restricted "
        f"for {a_allergy} or listed as containing {allergen}."
        + (f" The ontology restricts them for: {', '.join(restricted_for)}." if restricted_for else ""),
    ]
    if allergy_l in user_allergies:
        answer.append(f"\nSo your {allergy_noun} allergy is not a reason to avoid {phrase}.")
    elif not asked:
        answer.append(f"\nSo there's no need to replace {phrase} because of {allergen}.")

    # The user's own verdict for the food, so a real conflict (e.g. a nut
    # allergy) is still reported even though the stated reason was wrong.
    if user_profile:
        suit = build_food_suitability_context(f"can I eat {whole[0][0]}", whole[0][0], user_profile)
        if suit:
            verdict = re.search(r"^VERDICT_SENTENCE:\s*(.+)$", suit, re.M)
            conflicts = re.findall(r"^- CONFLICT:\s*(.+)$", suit, re.M)
            if verdict:
                answer.append(f"\nFor your profile: {verdict.group(1).strip()}")
                answer += [f"- {c}" for c in conflicts]
            context_lines += ["", suit]

    return {
        "context": "\n".join(context_lines),
        "answer": "\n".join(answer),
        "food": food_title,
        "allergen": allergen,
        "route": "false_premise",
        "uri": "",
    }


def _match_premise_recipe(phrase: str) -> Optional[URIRef]:
    """The recipe whose label IS the phrase ("goulash" -> Goulash). Exact only —
    find_recipe_entity's substring/fuzzy matching would turn a food name into
    some unrelated recipe."""
    target = normalize_text(phrase)
    singular = target[:-1] if target.endswith("s") else target
    for recipe in ontology_graph.subjects(RDF.type, SHFO.Recipe):
        if normalize_text(get_entity_label(recipe)) in (target, singular):
            return recipe
    return None


# Prepared recipe ingredients with no ingredient list of their own — they can't
# be cleared of an allergen, so they're named rather than called safe.
_UNCHECKABLE_INGREDIENT_WORDS = {"stock", "cube", "bouillon", "ketchup", "sauce", "mix", "dressing", "paste"}


def _recipe_allergen_answer(
    recipe: URIRef, allergen: str, allergy_noun: str, asked: bool,
    allergen_foods: frozenset, user_has_allergy: bool,
) -> Optional[dict]:
    """Recipes carry a real ingredient list, so containment is checked
    ingredient by ingredient against the same allergen exclusion labels."""
    label = get_entity_label(recipe)
    ingredients = [i["label"] for i in get_recipe_ingredients(recipe)]
    if not ingredients:
        return None
    hits = sorted(set(ingredients) & allergen_foods)
    if hits and not asked:
        return None   # a true premise — the normal routes answer the real question
    unclear = sorted(
        i for i in ingredients
        if i not in allergen_foods
        and (_is_composite_food(i, []) or set(re.findall(r"[a-z]+", i.lower())) & _UNCHECKABLE_INGREDIENT_WORDS)
    )
    context = [
        "ALLERGEN CHECK (recipe ingredient list)",
        f"Recipe: {label}",
        f"Ingredients: {', '.join(ingredients)}",
        f"{allergen} ingredients: {', '.join(hits) if hits else 'none'}",
    ]
    if hits:
        answer = [
            f"Yes — {label} contains {allergen}.",
            "",
            f"Its ingredient list in the SustainHealthyFood Ontology includes: {', '.join(hits)}.",
        ]
        if user_has_allergy:
            answer.append(f"\nThat conflicts with your {allergy_noun} allergy — avoid {label}.")
    else:
        answer = [
            f"No — {label} doesn't contain {allergen}"
            + ("." if asked else ", so the premise of the question isn't right."),
            "",
            f"None of the {len(ingredients)} ingredients in its SustainHealthyFood Ontology recipe "
            f"is a {allergen} food.",
        ]
        if unclear:
            answer.append(
                f"\n{', '.join(unclear)} {'is a prepared product' if len(unclear) == 1 else 'are prepared products'} "
                f"with no ingredient list in the ontology — check the label if you have "
                f"{'an' if allergy_noun[0] in 'aeiou' else 'a'} {allergy_noun} allergy."
            )
            context.append(f"Prepared ingredients that can't be checked: {', '.join(unclear)}")
    return {
        "context": "\n".join(context),
        "answer": "\n".join(answer),
        "food": label,
        "allergen": allergen,
        "route": "allergen_check" if (hits or asked) else "false_premise",
        "uri": str(recipe),
    }


QA_PROMPT = """
You are a precise dietary Q&A assistant in a conversational chat.

User Profile:
{profile_summary}

Recent conversation:
{chat_history}

User question:
{question}

Context:
{ontology_context}

Rules:
1. If the context contains numeric nutrient values from the SustainHealthyFood Ontology, treat them as authoritative.
2. NEVER invent numbers. If a number does not appear literally in the context, do not state it. 
Do not use your training knowledge to fill in missing values.
3. If the question asks whether the food/recipe CONTAINS a specific ingredient or allergen (e.g. "does X contain dairy",
"is there gluten in X", "does X have nuts/egg/soy"), answer that directly and clearly FIRST: check the Ingredients list
in the context (for a recipe) or the food's own composition for a literal match (dairy = milk, cheese, cream, yoghurt,
butter; gluten = wheat, flour, bread, pasta, breadcrumbs; nuts = almond, hazelnut, walnut, cashew, pistachio, etc.), and
state plainly whether it does or does not contain it, naming the specific matching ingredient(s) if it does. Do NOT
skip this and just list nutrient values instead — answering the actual question asked comes first.
For nutrient questions, you MUST report all nutrient values found in the ontology context for the chosen food per 100g,
and start the sentence saying based on the data from the SustainHealthyFood Ontology these are:
   - Always include Calories, all macronutrients, all micronutrients, and Nutrient Score when present.
   - If a nutrient is missing from the ontology context, explicitly say "not available in the ontology context".
4. For broader recommendation questions, use the ontology context plus the user profile to personalize the answer.
5. Keep answers concise, practical, and easy to read.
6. If the ontology context starts with "No food entity" or indicates no nutrient values were found, 
explain that clearly and ask the user to use the ontology food name.
7. If the context contains fields like "Entity:", "Ontology classes:", and "Data properties:" but NO numeric nutrient values, 
the question is about the entity's definition or role — NOT a nutrient lookup. In this case answer ONLY using the description 
and data properties from the context. Do NOT add any nutrient numbers from memory.
8. If the context contains "Chosen ontology food:", answer using only that chosen food instance.
9. For nutrient questions, if other matching ontology food instances are listed in the context, mention them briefly at the end 
and say:
   "If you want, I can show you another matching food instance from the ontology."
10. Do not replace ontology values with generic nutrition knowledge from memory.
11. If the context starts with "HEALTH RECOMMENDATION CONTEXT", this is a food recommendation query for a specific health condition. 
You MUST:
    - State clearly which health condition is being addressed.
    - Present each recommendation level as a labelled section (e.g. "Highly recommended:", "Recommended:", "Avoid:").
    - List the foods from the context under the correct section. Do NOT invent additional foods.
    - If a category has "... and X more", mention the total count and offer to show more on request.
    - Never pick a random food name from the list and treat it as the topic of the question.
    - End with a brief practical tip relevant to the health condition.
12. If the context contains "Environmental sustainability data:" followed by a list of values (GHG/CO2 Emissions,
Eutrophying Emissions, Land Use), report those exact values for the chosen food — do not estimate or invent a
number from memory. If instead the context says "Environmental sustainability data: not available in the ontology
for this food item", tell the user plainly that there is no information about the exact environmental impact/score
for this food item in the ontology — do NOT guess a value or substitute general knowledge about that food's
environmental impact.

Rules for recipes:
- If the context contains "Chosen ontology recipe:", you MUST answer using only that chosen recipe.
- If the user asks for another recipe and the context contains another stored ontology recipe, present that recipe only.
- Do NOT merge ingredients from different recipes.
- Use ONLY ontology facts for ingredients, ingredient amounts, and nutrient values.
- A recipe's nutrient values are listed under the "Recipe nutrient values:" heading, near the END of the context,
  after the ingredients list and the ingredient amounts list — check there specifically before saying a nutrient is
  "not available"; it's easy to miss since it comes after two other lists.
- If ingredient amounts are missing, say they are not available in the ontology context.
- Cooking instructions are NOT stored in the ontology. If the user asks how to prepare the dish, you MAY reason a short, 
practical set of preparation steps based solely on the ingredient list from the ontology context. 
Clearly indicate these steps are suggested based on the ingredients, not from the ontology.
- If the context says multiple recipes matched, present the chosen recipe first and end with:
  "If you want, I can show you a different matching recipe from the ontology."

Now provide the final answer only.
"""

PROFILE_REASONING_PROMPT = """
You are a knowledgeable dietary advisor who gives personalized nutrition guidance.

IMPORTANT: The user's complete profile is already provided below. Do NOT ask for any profile information — use what is given.

User Profile:
{profile_summary}

Recent conversation:
{chat_history}

User question:
{question}

Relevant context from the SustainHealthyFood Ontology (use only if helpful, ignore if not relevant):
{ontology_context}

Instructions:
- Use the profile above to answer the question directly and personally. The profile is complete — never ask the user for more details.
- Draw on ALL relevant profile fields in your answer:
  * Basics: age, sex, life stage (General Adult / Pregnant / Lactating / Pedriatic), weight, height
  * Goals & lifestyle: goal, activity level, diet preference, allergies/intolerances
  * Mental health & lifestyle: stress level, sleep quality, motivation/energy level, mental persona, appetite pattern today, support goal
- The mental health & lifestyle fields matter especially for questions about energy, mood, focus, or wellbeing — always factor them in.
- If the context includes a "MENTAL STATE CHART CONTEXT" section, it describes the eating pattern and nutrient gaps typical of the mood
 state resolved from the user's own profile. Use it to ground your food/nutrient advice in that state's recovery pathway 
 — blend it naturally into the answer, do not label it "mental state chart" to the user.
- If the context includes a "LIFE STAGE CONTEXT" section, it is the ontology's own guidance for the user's current life stage
 — use it as the authoritative source for that part of the answer instead of general knowledge. 
 In particular: for General Adult, state plainly that no special diet or nutrient is required; 
 for Pregnant, ground the answer in the pregnancy-restricted-diet and Pregnancy food-recommendation data given; 
 for Lactating, ground it in the lactation-diet guidance (and any foods listed as ideal for lactation); 
 for Pedriatic, recommend the nutrient-dense foods listed (high protein / iron / fiber) as supporting healthy growth.
- Be specific: quote the user's actual profile values in your answer (e.g. "Given your high stress level and irregular sleep..." 
or "Since you are lactating and your goal is to maintain weight...").
- Apply established nutritional knowledge where the ontology context does not cover the topic.
- Keep the response practical, concise, and easy to read.
- Do NOT invent specific numbers that are not in the ontology context or in established nutritional science.
"""

PROFILE_AVOIDANCE_PROMPT = """
You are a dietary Q&A assistant working with the SustainHealthyFood Ontology — a REAL ontology built on NEVO nutritional data. 
Do NOT question its existence.

User question:
{question}

The user's actual constraints (from their profile):
{constraint_summary}

Ontology context — one section per constraint the user actually has, each already restricted to real foods/guidance to avoid for that specific constraint:
{ontology_context}

Instructions:
- Cover ONLY the constraint sections shown above — do not add a category the user doesn't actually have 
(e.g. skip [Life stage] entirely if no life-stage section is present).
- Group your answer under one bold heading per constraint (e.g. "**Diet (Halal)**", "**Allergy (Soy)**", "**Life stage (Pregnant)**"), each followed by bullet points naming a food/food group to avoid.
- Write each food as its own markdown bullet line, starting with "- " — one food per line. Do NOT join foods into a single inline run separated by "•" or commas; that renders as a dense, hard-to-scan paragraph instead of a list.
- Just name each food/food group — do NOT add a reason or explanation for why it's excluded. The ontology context above gives you food names only, not reasons; any reason you wrote would be invented, not grounded, and dangerously confident-sounding when wrong (e.g. inventing that brandy is "made from pigs' feet" — it's a fruit spirit, the real issue is alcohol content).
- Only use foods/guidance that appear in the ontology context above for each section. Do NOT invent foods or add constraints not shown.
- Keep it concise — no intro paragraph, go straight into the first heading.
"""

PROFILE_RECOMMENDATION_PROMPT = """
You are a dietary Q&A assistant working with the SustainHealthyFood Ontology — a REAL ontology built on NEVO nutritional data. Do NOT question its existence.

User question:
{question}

The user's actual profile (diet preference and life stage):
{constraint_summary}

Ontology context — one section per constraint the user actually has, each already restricted to real foods recommended for that specific constraint:
{ontology_context}

Instructions:
- Cover EXACTLY the sections present in the ontology context above — no more, no fewer. If the context has only a "Life stage" section, your answer has only that one heading. NEVER add a "Diet preference (Vegan/Gluten-free/Low-calorie/High-protein/...)" section that is not in the context — the user's real diet preference is stated above and it is the only one that may appear.
- Group your answer under one bold heading per section (e.g. "**Diet preference (Vegan)**", "**Life stage (General Adult)**"), each followed by 4-8 bullet points naming a food to eat and a brief reason (max 10 words).
- Only use foods/guidance that appear in the ontology context above for each section. Do NOT invent foods or add constraints not shown.
- Keep it concise — no intro paragraph, go straight into the first heading.
"""

ENTITY_DEFINITION_PROMPT = """
You are a precise dietary Q&A assistant for the SustainHealthyFood Ontology.

Recent conversation:
{chat_history}

User question:
{question}

Ontology context:
{ontology_context}

Answer using ONLY the information in the ontology context above.
Do NOT add nutrient values, calorie counts, or any numbers that are not explicitly present in the context.
Do NOT use your training knowledge to fill in missing information.
Keep the answer concise: describe what the entity is, what class it belongs to, and what its properties say. If the context has a "Description:" line, base your explanation on it.

FOOD SOURCES — follow this exactly:
- Look for a line in the context that begins with "Food sources". If such a line is present, you MUST list several of the foods it names as the answer to "which foods are richest in it". NEVER say the ontology has no food sources when that line is present.
- Only if there is truly no "Food sources" line anywhere in the context: say the ontology does not record specific food sources for this entity, and do NOT name any foods from your own knowledge.

If the context includes a "Relationships" line, briefly work in anything practically relevant (e.g. a diet it's not recommended for).
"""

FOOD_CLASS_PROMPT = """
You are a precise dietary Q&A assistant. You are working with the SustainHealthyFood Ontology — a REAL, domain-specific ontology built on NEVO nutritional data. Treat all ontology data as authoritative.

User question:
{question}

Ontology context (authoritative — use ONLY the foods listed here):
{ontology_context}

Rules:
1. NEVER say the ontology is fictional, hypothetical, or not real.
2. Do NOT add foods, recipes, or nutritional values not present in the context.
3. If the context includes a "Foods" section and a "Recipes" section, present them under separate headings.
4. If the context describes a nutrient-threshold query (e.g. "less than 3 g of Fat"), include the measured value next to each item name.

Format rules — choose based on the question type:

A. If the question combines two nutritional properties (e.g. "high in protein AND low in fat", "high in fiber AND low in fat"):
   - Do NOT produce a plain bullet list of all foods.
   - Group the foods by their food group (e.g. Legumes, Fish & Seafood, Dairy, Vegetables). Use the food group information from the context if available.
   - For each group write 1–2 sentences explaining WHY foods in that group tend to have those nutritional properties (e.g. legumes are naturally high in plant protein and contain very little fat because...).
   - After the explanation, show a short representative selection (3–5 examples) from that group with their protein and fat values in parentheses.
   - End with a brief summary sentence about which food groups are the best sources overall.

B. If the question asks about a single nutritional category (e.g. "foods high in protein", "low fat foods"):
   - Present the foods as a grouped or alphabetical list with their measured value next to each name.
   - End with: "These are all the foods from the SustainHealthyFood Ontology matching this category."

Rules for recipes:
- Present under a "Matching Recipes" heading.
- Report only nutrient values stated in the context.
- Do NOT provide cooking instructions unless the user asks.
- End with: "These are all the recipes from the SustainHealthyFood Ontology matching this category."
"""

CLARIFICATION_PROMPT = """
You are a friendly dietary Q&A assistant for the SustainHealthyFood Ontology system.

Recent conversation:
{chat_history}

The user sent a message that could not be understood or matched:
"{question}"

Ask a short, friendly clarifying question. If the message looks like it contains a spelling mistake, gently suggest the corrected version.

This system can answer questions about:
- Nutrient values of specific foods (e.g. "How many calories in avocado?")
- Recipes (e.g. "Give me a recipe for goulash")
- Foods recommended for health conditions (e.g. "What foods are good for anemia?")
- Definitions of food concepts (e.g. "What is omega-3?")
- Personalized dietary advice based on the user profile

Keep your response short and friendly — one or two sentences at most.
"""

GENERAL_COOKING_PROMPT = """
You are a helpful dietary and cooking assistant with broad culinary knowledge.

User Profile:
{profile_summary}

Recent conversation:
{chat_history}

User question:
{question}

Instructions:
- Answer this cooking or meal planning question using your general culinary knowledge.
- Always respect the user's diet preference, allergies, and goal from the profile above.
- Be practical and specific: give recipe names, key ingredients, brief preparation steps, and storage tips where relevant.
- If multiple recipes are requested, present each one under a clear heading.
- Keep portion sizes and complexity realistic for home cooking and meal prep.
- Do NOT invent specific nutrient numbers unless you are certain they are accurate.
"""

GENERAL_KNOWLEDGE_PROMPT = """
You are a knowledgeable dietary and nutrition Q&A assistant.

User Profile:
{profile_summary}

Recent conversation:
{chat_history}

User question:
{question}

Ontology context (use only if directly relevant to the question, otherwise rely on general knowledge):
{ontology_context}

Instructions:
- If the question is about the user personally (e.g. "what is my BMI", "what are my calorie needs") answer directly from the User Profile above — it already includes calculated values like BMI, daily calorie need, and EFSA macro targets where available. Do NOT ask the user to provide information that's already in the profile.
- If the ontology context is directly relevant to the question, use it as your primary source.
- If the ontology context is not relevant or does not address the question, ignore it and answer from your general knowledge.
- Do NOT report nutrient values from the context unless the user is asking for the nutrient data of a specific food.
- For concept definitions (e.g. "what are carbohydrates"), give a clear educational explanation.
- Keep the answer concise, accurate, and easy to read.
"""

LLM_GENERAL_KNOWLEDGE_PROMPT = """
You are a knowledgeable dietary and nutrition Q&A assistant.

User Profile:
{profile_summary}

Recent conversation:
{chat_history}

User question:
{question}

This topic is not covered by the SustainHealthyFood Ontology. Answer using your general knowledge.
If the question is about the user personally (e.g. "what is my BMI", "what are my calorie needs")
answer directly from the User Profile above — it already includes calculated values like BMI, daily
calorie need, and EFSA macro targets where available. Do NOT ask the user to provide information
that's already in the profile.
Be concise, accurate, and practical. Focus on nutrition, dietary health, or food science as relevant.
"""

MEAL_SUGGESTION_PROMPT = """
You are a dietary Q&A assistant working with the SustainHealthyFood Ontology — a REAL ontology built on NEVO nutritional data.

Recent conversation:
{chat_history}

User question:
{question}

Ontology context:
{ontology_context}

Instructions:
- Use ONLY the foods/recipes listed in the ontology context above. Do NOT add foods from your training knowledge.
- {section_instruction}
- Within each section, pick 8–10 of the most practical and recognisable foods/recipes from the list.
- If a section includes a "Note: the user asked for HEALTHY options..." line, prefer items NOT tagged "high sugar" for your 8–10 picks; only include a "high sugar"-tagged item if there aren't enough untagged options to fill the list.
- Format each section as a markdown list. Each food or recipe must be on its OWN LINE using this exact format:
  - **Name** — one sentence: if the context provides "Ingredients:", list the main ones; otherwise explain briefly why it suits this meal/timing.
- Do NOT group items under extra category headers beyond the meal-type sections above. Do NOT invent ingredients or health claims not present in the context.
- Do NOT suggest preparing or serving a food with an accompaniment (e.g. "with milk", "topped with nuts", "served with yoghurt") unless that accompaniment is itself one of the listed foods and does not conflict with the user's profile.
- Only state that no foods were found for a section if that section's text literally says so. If the section lists foods, you MUST list them — never claim a listed section has no options.

Now answer the user's question using only the ontology context above.
"""

HEALTH_RECOMMENDATION_PROMPT = """
You are a dietary Q&A assistant working with the SustainHealthyFood Ontology — a REAL ontology built on NEVO nutritional data.

User question:
{question}

Ontology context:
{ontology_context}

Instructions:
- Only use foods and nutrients that appear in the ontology context. Do NOT invent any.
- The ontology context only includes sections that actually have foods in them — every section shown has at least one food. Never write a category heading (e.g. "EAT IN HIGH AMOUNTS") if it is not present in the context, and never say "none found" for a section that isn't there — just leave it out.
- Read the question carefully to decide which part of the context to use:
  a) If the question asks what to EAT, INCLUDE, or what HELPS → focus on the HIGHLY RECOMMENDED and RECOMMENDED sections (if present). Select 6–8 practical, recognisable whole foods (skip processed duplicates). For each food, write one sentence explaining WHY it helps, connecting it to the relevant nutrients listed in the "NUTRIENTS WITH POSITIVE EFFECT ON..." section.
  b) If the question asks what to AVOID, LIMIT, or what is NOT RECOMMENDED → focus on the "NOT RECOMMENDED / AVOID" section (if present). Select 6–8 common foods to avoid — the same count as the recommended list — and briefly explain why (e.g. they contain lactose, high sugar, etc.).
  c) If the question asks both (e.g. "which foods should I limit and which should I eat more of", "recommended and not recommended foods") → cover both sides, but ONLY using sections that are actually present in the context above. If the context has data for just one side (e.g. only a "NOT RECOMMENDED / AVOID" section, nothing under "RECOMMENDED"), answer with that one side alone. Do NOT invent a second list to fill the missing side, do NOT repeat foods from the side you do have under a second heading with different wording, and do NOT write a "no foods found"/"not listed in this context" placeholder for the missing side — the instruction above (never write a heading or "none found" placeholder for a section that isn't there) applies here too, not just to a) and b) individually.
- Never narrate your own reasoning or instructions in the answer — e.g. do NOT write things like "Since the question asks about both X and Y, I will provide a list of..." Just write the answer directly, with no meta-commentary about how you're approaching it.
- Add a short "Key nutrients" closing section ONLY when the context includes a "NUTRIENTS WITH POSITIVE EFFECT ON..." section — base it strictly on those listed nutrients. If that section is absent from the context, do NOT add a "Key nutrients" section or any generic nutrition summary of your own.
- If the context contains only nutrient data (no food recommendations), list the nutrients with a one-sentence explanation of each.
- Keep the overall response concise and easy to read.

Now answer the user's question using only the ontology context above.
"""

MEAL_PREP_PROMPT = """
You are a dietary Q&A assistant working with the SustainHealthyFood Ontology — a REAL ontology built on NEVO nutritional data. Do NOT question its existence.

User question:
{question}

Ontology context ({n_recipes} recipes retrieved from the SustainHealthyFood Ontology):
{ontology_context}

Read the ontology context above and follow these instructions:

1. NEVER say the ontology is fictional or hypothetical. It is real.
2. Present each recipe as a separate section. Begin each section with: "According to the SustainHealthyFood Ontology, here is a recipe: [Recipe Name]"
3. For each recipe:
   a. List the ingredients with their exact amounts as given in the context. Do NOT invent amounts.
   b. If ingredient amounts are not available in the context, say so — do NOT invent them.
   c. Provide brief practical preparation steps (4–6 steps) based ONLY on the listed ingredients. Clearly label these as "Suggested preparation steps" — they are not stored in the ontology.
   d. After the preparation steps, include a short nutrient summary (Calories, Protein, Carbohydrates, Fat, Fiber) using only the values from the context. If a value is missing, write "not available in the ontology context".
4. Do NOT merge ingredients or nutrients from different recipes.
5. Do NOT add ingredients or nutrient values not present in the context.
6. After all recipes, add 1–2 sentences of practical meal prep advice — e.g. which recipe stores best, which is quickest to reheat.

Now answer the user's question using only the ontology context above.
"""


RECIPE_PROMPT = """
You are a dietary Q&A assistant working with the SustainHealthyFood Ontology — a REAL ontology built on NEVO nutritional data. Do NOT question its existence.

Recent conversation:
{chat_history}

User question:
{question}

Ontology context:
{ontology_context}

Read the ontology context above and follow these instructions:

If the ontology context does NOT contain a recipe (it says something like "No recipe", "not found", or "no match"):
- Tell the user clearly that this recipe is NOT in the SustainHealthyFood Ontology.
- If the context lists available recipe examples, show them as suggestions.
- Do NOT generate or invent a recipe from your training knowledge.
- Do NOT provide ingredients, steps, or nutrients.

If the ontology context DOES contain a recipe (it includes "Chosen ontology recipe:" or an "Ingredients:" section):
1. NEVER say the ontology is fictional or hypothetical. It is real.
2. Present the answer as a RECIPE. Do NOT start with "these are the nutrient values".
3. Begin with: "According to the SustainHealthyFood Ontology, here is a recipe: [Recipe Name]"
4. List the ingredients with their amounts exactly as given in the context.
5. If ingredient amounts are not available in the context, say so — do NOT invent them.
6. After the ingredient list, provide the nutrient summary from the context. If a value is missing, write "not available in the ontology context".
7. Do NOT merge ingredients or nutrients from different recipes.
8. Do NOT add ingredients not listed in the context.
9. Cooking instructions are NOT stored in the ontology. If the user asks how to prepare the dish or asks for instructions, provide brief practical preparation steps based ONLY on the listed ingredients. Clearly label these as suggested preparation steps, not from the ontology. If the user is asking specifically for instructions (not just the recipe), lead with the preparation steps right after naming the recipe — do NOT repeat the full nutrient breakdown first.
10. If other matching recipes are mentioned in the context, end with: "If you want, I can show you a different matching recipe from the ontology."

Now answer the user's question using only the ontology context above.
"""


FOOD_SUITABILITY_PROMPT = """
You are a dietary Q&A assistant working with the SustainHealthyFood Ontology — a REAL ontology built on NEVO nutritional data. Do NOT question its existence.

User profile:
{profile_summary}

User question:
{question}

Ontology context:
{ontology_context}

The context has a "PROFILE ASSESSMENT" line with a verdict, plus "CONFLICT:" and "CAUTION:" bullets already
computed against THIS user's diet, allergies and life stage. The verdict sentence is handled separately and
will be placed before your text — do NOT write your own yes/no verdict sentence.

Write ONLY: 2–4 short bullets, then one closing sentence. No opening sentence, no numbering, no "1.".

- Bullets = the reasons, taken ONLY from the CONFLICT / CAUTION bullets and the ontology facts above (food
  group, nutritional flags, molecules, key nutrients). Cite the numbers from "Key nutrients" where relevant.
  Do not repeat a phrase like "high in fat (high in fat)" — state each fact once, cleanly.
- If there is an allergy CONFLICT, the FIRST bullet must name that allergy as the reason. An allergy
  conflict is about the allergy only — never say the user's diet type (vegetarian, halal, ...) excludes a
  food because of an allergen.
- If the question asks about a specific quality (e.g. "a good protein source"), answer it with the
  context's numbers (e.g. the Protein value per 100 g) in one bullet.
- Address the user as "you", never "the user".
- If a CONFLICT or CAUTION says the ontology cannot confirm the food is free of an allergen, state plainly
  that the ontology stores no ingredient list for this item and the user must check the product label — do
  NOT claim it is safe or unsafe based on your own general knowledge of the dish, and do NOT contradict this
  later.
- Closing sentence = one practical line: a safer alternative, or how to fit it in. It must NOT reverse the
  verdict (never end a "not recommended" answer with "but you can have it occasionally").

Rules:
- Only mention a health condition or diet from the "recommended for" / "NOT recommended for" lines if it
  matches this user's own profile — otherwise ignore those lines.
- Do NOT invent ingredients, nutrients or allergen information that is not in the context.
- Keep it under about 90 words.

Now write the bullets and closing sentence.
"""

# Appended deterministically after generation (see generate_answer, near the
# return) to every answer on a routing path that makes a diet/allergy/health-
# condition/mood recommendation — not asked of the LLM as a prompt
# instruction, since every other "always do X in your answer" instruction in
# this file has proven unreliable with llama3.2 (inconsistent counts,
# inconsistent headings, a disclaimer like this one appearing on some health
# questions but not others with no code reason why). A fixed, code-level
# append is the only way to guarantee every user actually sees it, with
# identical wording every time.
_HEALTHCARE_DISCLAIMER_ROUTES = {
    "profile_reasoning", "profile_avoidance", "profile_recommendation",
    "health_recommendation",
}
_HEALTHCARE_DISCLAIMER = (
    "These recommendations are based on the provided ontology context and "
    "your stated profile, but may not account for your complete medical "
    "history or individual circumstances. Always consult a healthcare "
    "professional for specific dietary advice."
)


_retry_q = st.session_state.retry_question
if _retry_q:
    st.session_state.retry_question = None
    user_input = _retry_q
    submitted = True
    _is_retry = True
else:
    _is_retry = False

def generate_answer(
    user_input: str,
    user_profile: dict,
    profile_summary: str,
    is_retry: bool = False,
    status_cb: Optional[Callable[[str, int], None]] = None,
) -> dict:
    """Classify the question, route it to the right ontology-grounded context
    builder, and generate the final LLM answer. This is the single source of
    truth for "how the system answers a question" — the Streamlit UI below
    calls it when the user submits a question, and evaluation scripts can
    call it directly with a synthetic profile/question to test the exact
    same routing and prompting logic, with no Streamlit runtime required.

    Returns a dict with: response, ontology_context, routing_path,
    retrieval_method, matched_entity, matched_entity_uri, llm_category,
    llm_entity, trace.
    """
    def _report(msg: str, pct: int) -> None:
        if status_cb:
            status_cb(msg, pct)

    _report("Step 1 / 3 — Classifying question...", 10)

    general_retriever = init_general_retriever()

    routing_path = ""
    retrieval_method = ""
    matched_entity = ""
    matched_entity_uri = ""
    # Every route below is grounded in retrieved ontology context — the job is
    # to report that context faithfully, not to compose freely. Temperature 0
    # keeps the eval reproducible and stops "same context, different answer".
    # A retry still gets a little jitter so it can escape whatever rut produced
    # the first (bad/empty) answer.
    _base_temp = 0.3 if is_retry else 0.0
    llm_kwargs = {"temperature": _base_temp, "num_predict": 300}

    # Step 1: let the LLM understand the question and classify intent + entity.
    # Followup requests are purely session-based, so skip classification for them.
    is_followup = (
        (is_recipe_followup_request(user_input) or is_food_followup_request(user_input))
        and not is_general_cooking_question(user_input)
    )
    if not is_followup:
        llm_category, llm_entity = classify_question(user_input)
    else:
        llm_category, llm_entity = "unknown", ""

    _report("Step 2 / 3 — Retrieving context from ontology...", 40)

    # Step 2: route based on the LLM classification, with keyword fallbacks.
    # A "what is X" question whose entity has an ontology rdfs:comment is
    # answered from that comment first, ahead of every other routing path —
    # otherwise entities that double as e.g. RecommendedFor/notRecommendedFor
    # targets (a Diet, a LifeStage, ...) get intercepted by that path's
    # handler, which never surfaces the entity's own description.
    # A false premise ("since almonds contain soy, ...") is checked before
    # anything else — every route below would answer as if it were true.
    premise = check_false_premise(user_input, user_profile)
    commented_entity = find_commented_entity(user_input, llm_entity)
    if premise is not None:
        # "false_premise" (claim refuted) or "allergen_check" ("does goulash
        # contain dairy?" answered from the ingredient list / allergen tags).
        ontology_context = premise["context"]
        routing_path = premise["route"]
        retrieval_method = "ontology_graph"
        matched_entity = f"{premise['food']} — {premise['allergen']}"
        matched_entity_uri = premise["uri"]

    elif commented_entity:
        ontology_context = build_general_entity_context(commented_entity, user_profile=user_profile)
        routing_path = "general_ontology_direct_lookup"
        retrieval_method = "ontology_graph"
        matched_entity = get_entity_label(commented_entity)
        matched_entity_uri = str(commented_entity)

    elif is_recipe_followup_request(user_input) and not is_general_cooking_question(user_input):
        ontology_context = build_next_recipe_from_memory(user_profile=user_profile)
        routing_path = "recipe_followup"
        retrieval_method = "session_memory"
        idx = st.session_state.get("last_recipe_index", 0)
        matches = st.session_state.get("last_recipe_matches", [])
        if matches and idx < len(matches):
            matched_entity = get_entity_label(matches[idx])
            matched_entity_uri = str(matches[idx])

    elif is_food_followup_request(user_input):
        ontology_context = build_next_food_from_memory(user_profile=user_profile)
        routing_path = "food_followup"
        retrieval_method = "session_memory"
        idx = st.session_state.get("last_food_index", 0)
        matches = st.session_state.get("last_food_matches", [])
        if matches and idx < len(matches):
            matched_entity = get_entity_label(matches[idx])
            matched_entity_uri = str(matches[idx])

    elif (_comparison_ctx := build_nutrient_comparison_context(user_input, user_profile)) is not None:
        # "Which has more iron, chicken or beef?" — ahead of every route that
        # resolves a single food (nutrient_lookup, food_suitability), since
        # those match the whole two-food question against one label and miss.
        ontology_context = _comparison_ctx
        routing_path = "nutrient_comparison"
        retrieval_method = "ontology_graph"
        llm_kwargs = {"temperature": 0.0, "num_predict": 500}
        _parsed = parse_nutrient_comparison(user_input)
        matched_entity = " vs ".join(_parsed[1]) + " — " + ", ".join(_parsed[0])

    elif (_ranking_ctx := build_nutrient_ranking_context(user_input, user_profile)) is not None:
        # "Which fruit has the highest vitamin C?" — a superlative over a food
        # group; nutrient_lookup would pick one fruit and list its nutrients.
        ontology_context = _ranking_ctx
        routing_path = "nutrient_ranking"
        retrieval_method = "ontology_graph"
        _rk = parse_nutrient_ranking(user_input)
        matched_entity = f"{_rk[3]} ({_rk[2] or 'all foods'}) — {_rk[0]}, {_rk[1]}est"

    elif is_profile_avoidance_question(user_input) and build_profile_avoidance_context(user_profile):
        # "What should I avoid given my diet/allergies/life stage" phrasing
        # trips Rule 1 of the router (personal "given my"/"my allergies"
        # wording) into profile_reasoning, but the real answer should come
        # from the user's *actual* diet/allergy/life-stage constraints in the
        # ontology, not a generic guideline search that has no idea which
        # diet or allergy applies. Checked ahead of profile_reasoning below
        # so it wins whenever the persona actually has a groundable constraint;
        # falls through to profile_reasoning when they have none (nothing to
        # avoid, so the generic answer there is fine).
        ontology_context = build_profile_avoidance_context(user_profile)
        routing_path = "profile_avoidance"
        retrieval_method = "ontology_graph (multi-constraint)"
        llm_kwargs = {"temperature": 0.0, "num_predict": 600}
        matched_entity = _build_constraint_summary(user_profile)

    elif is_profile_recommendation_question(user_input) and build_profile_recommendation_context(user_profile):
        # Positive-side mirror of profile_avoidance above, same rationale:
        # "What food is recommended for my life stage and diet preference"
        # trips Rule 1 into profile_reasoning, but the real answer should
        # come from the user's diet-type recommended foods plus their life
        # stage's own guidance (which now has real content for every life
        # stage, including General Adult — see _get_general_adult_food_buckets).
        ontology_context = build_profile_recommendation_context(user_profile)
        routing_path = "profile_recommendation"
        retrieval_method = "ontology_graph (multi-constraint)"
        llm_kwargs = {"temperature": 0.0, "num_predict": 600}

        constraint_parts = []
        diet = user_profile.get("diet_type", "None")
        if diet not in ("None", ""):
            constraint_parts.append(f"Diet: {diet}")
        life_stage = user_profile.get("life_stage", "General Adult")
        constraint_parts.append(f"Life stage: {life_stage}")
        matched_entity = "; ".join(constraint_parts)

    elif (
        (llm_category == "food_suitability" or is_food_suitability_question(user_input))
        and (_suitability_ctx := build_food_suitability_context(
            user_input, llm_entity, user_profile
        )) is not None
    ):
        # "Can I eat X / is X recommended for me" — a specific named food plus a
        # personal-consumption question. Placed ahead of profile_reasoning and
        # health_recommendation because both otherwise intercept it: the former
        # on "should I" wording, the latter by treating the food's own
        # recommendedFor/notRecommendedFor triples as the answer. The context is
        # already assessed against the user's diet/allergies/life stage.
        ontology_context = _suitability_ctx
        routing_path = "food_suitability"
        retrieval_method = "ontology_graph"
        llm_kwargs = {"temperature": 0.0, "num_predict": 500}
        _suit_resolved = _resolve_suitability_entity(user_input, llm_entity or "")
        if _suit_resolved is not None:
            matched_entity = get_entity_label(_suit_resolved[0])
            matched_entity_uri = str(_suit_resolved[0])

    elif (
        (llm_category == "profile_reasoning" or is_profile_reasoning_question(user_input)
         or _is_generic_mood_question(user_input))
        and not find_meal_types(user_input)
        and (
            not find_health_state_entity(llm_entity or user_input)
            # A generic mood question ("what food helps elevate my mood?")
            # can still resolve to SOME health-state entity here — either the
            # router LLM guessed one from the bare text, or the fuzzy-match
            # fallback in find_health_state_entity coincidentally matched
            # something — but that guess has no connection to the user's own
            # actual resolved state, so it must not disqualify this branch
            # the way an explicitly-named condition correctly still does.
            or _is_generic_mood_question(user_input)
        )
    ):
        # Profile-based questions: retrieve general ontology context but strip out
        # food-individual documents — they are noisy and confuse the LLM into
        # thinking it should answer about a specific food rather than the profile.
        # Excludes questions that also resolve to a specific meal type (e.g.
        # "before and after training" -> PreWorkout/PostWorkout) or health-state
        # entity (e.g. "manage my stress" -> Stressed) — those have a more
        # specific, more strictly ontology-grounded route further down this
        # chain, and generic profile_reasoning triggers like "my goal" or "my
        # stress" would otherwise hijack them before they get there.
        food_uris = {str(uri) for _, uri in FOOD_CANDIDATES}
        ontology_docs = general_retriever.get_relevant_documents(user_input)
        relevant_docs = [
            d for d in ontology_docs
            if d.metadata.get("id", "") not in food_uris
            and d.metadata.get("kind") not in ("individual_sources", "individual")
        ]
        ontology_context = "\n\n".join(
            doc.page_content for doc in relevant_docs[:5]
        ) or "No additional ontology context available for this question."
        routing_path = "profile_reasoning"
        retrieval_method = "vector_store_general"
        llm_kwargs = {"temperature": 0.0, "num_predict": 600}

        # Mood/energy-flavoured profile questions ("what should I eat to support
        # my mood?") land here because "should I" forces profile_reasoning in the
        # router — pull in the mental state chart using the state resolved from
        # the user's own stress/sleep/motivation/appetite/support_goal fields.
        if is_mood_related_question(user_input):
            chart_state = resolve_mental_state_from_profile(user_profile)
            mental_chart_context = build_mental_state_chart_context(chart_state, user_input)
            ontology_context = f"{ontology_context}\n\n{mental_chart_context}"
            retrieval_method = "vector_store_general + mental_state_chart"
            matched_entity = chart_state

        # Always ground the user's life stage (General Adult / Pregnant /
        # Lactating / Pedriatic) in the ontology's own data — see
        # build_life_stage_context for what each stage pulls in.
        life_stage_context = build_life_stage_context(
            user_profile.get("life_stage", ""), user_input, user_profile=user_profile
        )
        if life_stage_context:
            ontology_context = f"{ontology_context}\n\n{life_stage_context}"
            retrieval_method = f"{retrieval_method} + life_stage"

    elif llm_category == "meal_suggestion" or is_meal_suggestion_question(user_input):
        entity_hint = llm_entity if llm_entity else user_input
        ontology_context = build_meal_suggestion_context(user_input, entity_hint, user_profile)
        if ontology_context is None:
            health_docs = init_health_retriever().get_relevant_documents(user_input)
            ontology_context = "\n\n".join(
                d.page_content for d in health_docs[:5]
            ) or "No meal suggestions found in the ontology for this query."
        routing_path = "meal_suggestion"
        retrieval_method = "ontology_graph"
        meal_results = find_meal_types(entity_hint)
        # Two meal-type sections (e.g. PreWorkout + PostWorkout) need more room
        # than a single-section breakfast/lunch/dinner/snack answer.
        num_predict = 900 if len(meal_results) <= 1 else 700 * len(meal_results)
        llm_kwargs = {"temperature": 0.0, "num_predict": num_predict}
        if meal_results:
            matched_entity = " & ".join(label for label, _ in meal_results)
            matched_entity_uri = " & ".join(str(uri) for _, uri in meal_results)

    elif detect_meal_prep_recipe_request(user_input):
        n_recipes = detect_meal_prep_recipe_request(user_input)
        ontology_context = build_meal_prep_context(n_recipes, user_profile=user_profile)
        routing_path = "meal_prep"
        retrieval_method = "ontology_graph"
        llm_kwargs = {"temperature": 0.0, "num_predict": 900}
        st.session_state["_meal_prep_n"] = n_recipes

    elif llm_category == "general_cooking" or is_general_cooking_question(user_input):
        ontology_context = ""
        routing_path = "general_cooking"
        retrieval_method = "llm_general_knowledge"
        llm_kwargs = {"temperature": 0.5, "num_predict": 900}

    elif detect_food_class_intersection_query(user_input):
        class_names = detect_food_class_intersection_query(user_input)
        ontology_context = build_food_class_intersection_context(class_names)
        routing_path = "food_class_lookup"
        retrieval_method = "ontology_graph"
        matched_entity = " & ".join(get_entity_label(SHFO[c]) for c in class_names)
        matched_entity_uri = ", ".join(str(SHFO[c]) for c in class_names)

    elif llm_category == "nutrient_lookup":
        # Use LLM-extracted entity name for precise food matching — unless the
        # question's own food phrase matches at least as well. The router
        # sometimes truncates the entity ("sausage raw beef" -> "Sausage"),
        # and then ~9 "... sausage" labels tie and the alphabetical first
        # (Liver pate sausage) wins. Its spelling fixes ("avokado" ->
        # "Avocado") still win, since the misspelt phrase scores lower.
        lookup_phrase = llm_entity if llm_entity else user_input
        q_phrase = extract_food_phrase(user_input)
        if llm_entity and q_phrase != user_input.lower().strip(" ?.,!"):
            def _top(p: str) -> int:
                m = find_food_candidates(p)
                return m[0][0] if m else 0
            if _top(q_phrase) >= _top(llm_entity):
                lookup_phrase = q_phrase
        ontology_context = build_nutrient_context_from_question(lookup_phrase, user_profile=user_profile)
        routing_path = "nutrient_lookup"
        retrieval_method = "ontology_graph"
        # Set by build_nutrient_context_from_question itself for whichever
        # entity it actually resolved — a Food via FOOD_CANDIDATES, or a
        # Recipe via its find_recipe_entity fallback (recipes never appear
        # in last_food_matches, see that function for why). More precise
        # than last_food_matches, which could be stale from an unrelated
        # earlier food question if this lookup matched nothing at all.
        _nl_uri = st.session_state.get("last_nutrient_lookup_entity_uri")
        if _nl_uri:
            matched_entity = get_entity_label(_nl_uri)
            matched_entity_uri = str(_nl_uri)

    elif llm_category == "environmental_impact" or is_environmental_question(user_input):
        # Checked ahead of definition_lookup/general_ontology_question (which would
        # otherwise intercept phrasing like "what is the environmental impact of X")
        # and ahead of the generic is_nutrient_question fallback below (some
        # environmental keywords overlap with "how much"/"amount of").
        lookup_phrase = llm_entity if llm_entity else user_input
        ontology_context = build_environmental_context_from_question(lookup_phrase, user_profile=user_profile)
        routing_path = "environmental_impact"
        retrieval_method = "ontology_graph"
        food_matches = st.session_state.get("last_food_matches", [])
        if food_matches:
            matched_entity = get_entity_label(food_matches[0])
            matched_entity_uri = str(food_matches[0])

    elif llm_category == "recipe_lookup" or is_recipe_by_ingredient_question(user_input) or is_recipe_title_search_question(user_input) or is_recipe_question(user_input):
        lookup_phrase = llm_entity if llm_entity else user_input
        if is_recipe_by_ingredient_question(user_input):
            ontology_context = build_recipe_by_ingredient_context(user_input, user_profile=user_profile)
            routing_path = "recipe_by_ingredient"
        elif is_recipe_title_search_question(user_input) or (llm_entity and llm_category == "recipe_lookup"):
            ontology_context = (
                build_recipe_title_search_context(lookup_phrase, user_profile=user_profile)
                if llm_entity else
                build_recipe_title_search_context(user_input, user_profile=user_profile)
            )
            routing_path = "recipe_title_search"
        else:
            ontology_context = build_recipe_context_from_question(user_input, user_profile=user_profile)
            routing_path = "recipe_lookup"
        retrieval_method = "ontology_graph"
        llm_kwargs = {"temperature": 0.0, "num_predict": 500}
        matches = st.session_state.get("last_recipe_matches", [])
        if matches:
            matched_entity = get_entity_label(matches[0])
            matched_entity_uri = str(matches[0])

    elif detect_food_class_query(user_input):
        food_class = detect_food_class_query(user_input)
        ontology_context = build_food_class_context(food_class, user_profile=user_profile)
        routing_path = "food_class_lookup"
        retrieval_method = "ontology_graph"
        matched_entity = get_entity_label(SHFO[food_class])
        matched_entity_uri = str(SHFO[food_class])

    elif llm_category == "health_recommendation" or find_health_state_entity(llm_entity or user_input):
        entity_hint = llm_entity if llm_entity else user_input
        ontology_context = build_health_recommendation_context(user_input, entity_hint, user_profile=user_profile)
        if ontology_context is None:
            # No specific health condition resolved (e.g. the question names
            # no condition at all, just "what should I avoid eating?"). Before
            # falling back to a vector search with NO diet/allergy awareness
            # whatsoever — which previously returned content with no relation
            # to the asker's actual profile, e.g. a Vegan user getting back
            # toffees-for-diabetics and oils-for-fat-restriction — check
            # whether this is really a generic personal avoidance question in
            # disguise, and if so answer from the user's actual constraints
            # via the same path the dedicated profile_avoidance branch above
            # uses, instead of semantically-similar-but-irrelevant snippets.
            _avoidance_ctx = (
                build_profile_avoidance_context(user_profile)
                if is_profile_avoidance_question(user_input) else None
            )
            if _avoidance_ctx:
                ontology_context = _avoidance_ctx
                routing_path = "profile_avoidance"
                retrieval_method = "ontology_graph (multi-constraint)"
                matched_entity = _build_constraint_summary(user_profile)
            else:
                # Genuinely nothing ontology-grounded to offer — no condition
                # resolved, and not an avoidance question either (e.g. the
                # resolved state has no food-recommendation data at all, like
                # "Energized" — see _is_generic_mood_question's docstring).
                # This used to fall back to a semantic vector search instead,
                # but that search has no reliable way to tell "relevant" from
                # "irrelevant" here: every health_recommendation-sourced
                # document already contains boilerplate phrases like "avoid"
                # ("avoid if you have, should not eat for, bad for..."), so a
                # keyword-overlap relevance check (is_context_relevant, used
                # for exactly this purpose elsewhere) would call almost any
                # such document "relevant" regardless of real topical fit —
                # which is how the Vegan user from the earlier bug report got
                # back toffees-for-diabetics in the first place. Answering
                # plainly from general knowledge, clearly marked as such via
                # LLM_GENERAL_KNOWLEDGE_PROMPT, is more honest than dressing
                # an ungrounded answer up as if it came from the ontology.
                ontology_context = ""
                routing_path = "llm_general_knowledge"
                retrieval_method = "llm_fallback"
        else:
            routing_path = "health_recommendation"
            retrieval_method = "ontology_graph"
        llm_kwargs = {"temperature": 0.0, "num_predict": 600}
        # Guarded: find_health_state_entity(entity_hint) can resolve a real
        # entity that build_health_recommendation_context still returned None
        # for (one with no food buckets AND no hasPositiveEffectOn nutrients
        # at all — fully empty of data) — in the llm_general_knowledge
        # fallback case above, showing that entity as "matched" would be
        # misleading, since the answer wasn't actually grounded in it.
        if routing_path != "llm_general_knowledge":
            state_uri = find_health_state_entity(entity_hint)
            if state_uri:
                matched_entity = get_entity_label(state_uri)
                matched_entity_uri = str(state_uri)
        # No mental_state_chart blend here (unlike profile_reasoning below) —
        # for a direct "what food helps with X" question, build_health_recommendation_context
        # already returns a rich, contradiction-free, ontology-grounded answer
        # (recommendedFor/notRecommendedFor buckets + hasPositiveEffectOn nutrients).
        # The chart's vector-retrieved research-evidence text was frequently
        # off-topic (e.g. depression/anxiety studies surfacing for a stress
        # question) and, being appended after the ontology block, was mostly
        # cut off by ragas_faithfulness's 3000-char context truncation anyway
        # — so it was adding retrieval noise without being reliably measurable.

    elif llm_category == "definition_lookup" or is_general_ontology_question(user_input):
        subject = llm_entity if llm_entity else extract_query_subject(user_input)
        direct_uri = GENERAL_ENTITY_INDEX.get(normalize_text(subject)) if subject else None

        if direct_uri:
            ontology_context = build_general_entity_context(direct_uri, user_profile=user_profile)
            routing_path = "general_ontology_direct_lookup"
            retrieval_method = "ontology_graph"
            matched_entity = get_entity_label(direct_uri)
            matched_entity_uri = str(direct_uri)
        else:
            ontology_docs = general_retriever.get_relevant_documents(user_input)
            desc_docs = [d for d in ontology_docs if d.metadata.get("kind") != "individual_sources"]
            seen_ids: set = set()
            deduped = []
            for doc in desc_docs:
                base_id = doc.metadata.get("id", "").replace("__sources", "")
                if base_id not in seen_ids:
                    seen_ids.add(base_id)
                    deduped.append(doc)
            docs_to_use = deduped[:5] if deduped else ontology_docs[:5]

            food_uris = {str(uri) for _, uri in FOOD_CANDIDATES}
            top_id = docs_to_use[0].metadata.get("id", "") if docs_to_use else ""
            if top_id in food_uris:
                ontology_context = build_nutrient_context_from_question(user_input, user_profile=user_profile)
                routing_path = "nutrient_lookup"
                retrieval_method = "ontology_graph"
                _nl_uri = st.session_state.get("last_nutrient_lookup_entity_uri")
                if _nl_uri:
                    matched_entity = get_entity_label(_nl_uri)
                    matched_entity_uri = str(_nl_uri)
            else:
                ontology_context = "\n\n".join(
                    doc.page_content for doc in docs_to_use
                ) or ""
                if is_context_relevant(user_input, docs_to_use):
                    routing_path = "general_ontology_vector_search"
                    retrieval_method = "vector_store_general"
                    retrieved_names = [
                        doc.metadata.get("id", "").split("#")[-1]
                        for doc in docs_to_use
                        if doc.metadata.get("id")
                    ]
                    if retrieved_names:
                        matched_entity = ", ".join(retrieved_names[:5])
                        matched_entity_uri = docs_to_use[0].metadata.get("id", "")
                else:
                    routing_path = "llm_general_knowledge"
                    retrieval_method = "llm_fallback"
                    ontology_context = ""

    elif is_nutrient_question(user_input):
        ontology_context = build_nutrient_context_from_question(user_input, user_profile=user_profile)
        routing_path = "nutrient_lookup"
        retrieval_method = "ontology_graph"
        _nl_uri = st.session_state.get("last_nutrient_lookup_entity_uri")
        if _nl_uri:
            matched_entity = get_entity_label(_nl_uri)
            matched_entity_uri = str(_nl_uri)

    elif llm_category == "unknown":
        ontology_context = ""
        routing_path = "clarification"
        retrieval_method = "none"

    else:
        ontology_docs = general_retriever.get_relevant_documents(user_input)
        ontology_context = "\n\n".join(
            doc.page_content for doc in ontology_docs[:5]
        ) or "No ontology knowledge found."
        routing_path = "generic_vector_search"
        retrieval_method = "vector_store_general"

    # Strip _profile_conflict_note()'s warning out of what the trace displays
    # as "context passed to the LLM" — it's an instruction to the model, not
    # retrieved ontology content, and is re-shown after the answer instead
    # (see the profile_conflict_note_display handling near the return below).
    # The LLM's own prompt still gets the unmodified ontology_context.
    _pw_match = _PROFILE_WARNING_RE.search(ontology_context)
    _display_context = _PROFILE_WARNING_RE.sub("", ontology_context)
    _pen_match = _PROFILE_EXCLUSION_RE.search(_display_context)
    _display_context = _PROFILE_EXCLUSION_RE.sub("", _display_context)

    trace = ResponseTrace(
        question=user_input,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        routing_path=routing_path,
        retrieval_method=retrieval_method,
        matched_entity=matched_entity,
        matched_entity_uri=matched_entity_uri,
        context_snippet=_display_context[:1500],
        context_length=len(ontology_context),
        llm_temperature=llm_kwargs["temperature"],
        llm_num_predict=llm_kwargs["num_predict"],
    )

    chat_history = build_chat_history_text(n=3)

    if routing_path == "clarification":
        prompt = CLARIFICATION_PROMPT.format(
            question=user_input,
            chat_history=chat_history,
        )
    elif routing_path == "general_ontology_direct_lookup":
        prompt = ENTITY_DEFINITION_PROMPT.format(
            question=user_input,
            ontology_context=ontology_context,
            chat_history=chat_history,
        )
    elif routing_path == "profile_avoidance":
        prompt = PROFILE_AVOIDANCE_PROMPT.format(
            question=user_input,
            constraint_summary=matched_entity,
            ontology_context=ontology_context,
        )
    elif routing_path == "profile_recommendation":
        prompt = PROFILE_RECOMMENDATION_PROMPT.format(
            question=user_input,
            constraint_summary=matched_entity,
            ontology_context=ontology_context,
        )
    elif routing_path == "profile_reasoning":
        prompt = PROFILE_REASONING_PROMPT.format(
            profile_summary=profile_summary,
            question=user_input,
            ontology_context=ontology_context,
            chat_history=chat_history,
        )
    elif routing_path == "food_suitability":
        prompt = FOOD_SUITABILITY_PROMPT.format(
            profile_summary=profile_summary,
            question=user_input,
            ontology_context=ontology_context,
        )
    elif routing_path == "food_class_lookup":
        prompt = FOOD_CLASS_PROMPT.format(
            question=user_input,
            ontology_context=ontology_context,
        )
    elif routing_path == "meal_suggestion":
        if len(meal_results) > 1:
            section_instruction = (
                'The context contains more than one "MEAL SUGGESTION CONTEXT" section '
                '(one per meal type/timing above) — answer each separately under its own '
                'short heading named after that section\'s meal type/timing, covering both.'
            )
        else:
            section_instruction = (
                'The context contains a single "MEAL SUGGESTION CONTEXT" section — answer '
                'it directly as one plain list. Do NOT split it into multiple headings and '
                'do NOT invent a "Before training"/"After training" or any other heading.'
            )
        prompt = MEAL_SUGGESTION_PROMPT.format(
            question=user_input,
            ontology_context=ontology_context,
            chat_history=chat_history,
            section_instruction=section_instruction,
        )
    elif routing_path == "health_recommendation":
        prompt = HEALTH_RECOMMENDATION_PROMPT.format(
            question=user_input,
            ontology_context=ontology_context,
        )
    elif routing_path == "meal_prep":
        n_recipes = st.session_state.pop("_meal_prep_n", 3)
        prompt = MEAL_PREP_PROMPT.format(
            question=user_input,
            ontology_context=ontology_context,
            n_recipes=n_recipes,
        )
    elif routing_path in (
        "recipe_by_ingredient", "recipe_lookup",
        "recipe_title_search", "recipe_followup",
    ):
        prompt = RECIPE_PROMPT.format(
            question=user_input,
            ontology_context=ontology_context,
            chat_history=chat_history,
        )
    elif routing_path == "general_cooking":
        prompt = GENERAL_COOKING_PROMPT.format(
            profile_summary=profile_summary,
            question=user_input,
            chat_history=chat_history,
        )
    elif routing_path == "general_ontology_vector_search":
        prompt = GENERAL_KNOWLEDGE_PROMPT.format(
            profile_summary=profile_summary,
            question=user_input,
            ontology_context=ontology_context,
            chat_history=chat_history,
        )
    elif routing_path == "llm_general_knowledge":
        prompt = LLM_GENERAL_KNOWLEDGE_PROMPT.format(
            profile_summary=profile_summary,
            question=user_input,
            chat_history=chat_history,
        )
    else:
        prompt = QA_PROMPT.format(
            profile_summary=profile_summary,
            question=user_input,
            ontology_context=ontology_context,
            chat_history=chat_history,
        )

    # Universal profile safety net — runs for every routing_path, not just the
    # ones that already built their context off _get_diet_and_allergy_exclusions
    # (health_recommendation, meal_suggestion). Some routes (nutrient_lookup,
    # recipe_lookup, food_class_lookup, general_ontology_direct_lookup, ...)
    # pull ontology_context from a food/recipe/class match with no diet/allergy
    # awareness at all, so a conflicting item could otherwise reach the answer
    # unflagged. Doesn't rewrite ontology_context (routes above already used it
    # to build the prompt) — just appends an instruction the LLM must follow.
    #
    # Skipped for profile_avoidance — same reason the grounding safety net
    # below already skips it ("avoidance answers are meant to name conflicting
    # foods"). profile_avoidance's own context IS ALREADY the correct,
    # complete avoid-list straight from the ontology's notRecommendedFor
    # triples. This check re-scans that same text against a SEPARATE,
    # keyword-based exclusion set, truncates to 15 items with a "(+N more)"
    # placeholder, and hands the LLM two different, overlapping-but-not-
    # identical "conflicting" lists for the same profile. In practice the LLM
    # then (a) copies the "(+N more)" placeholder verbatim into its own
    # bullet list as if it were part of a food name, and (b) invents a false
    # "doesn't actually contain dairy" distinction for every item outside the
    # truncated 15 — even when the ontology's own, already-authoritative list
    # correctly included them with no such distinction needed.
    profile_conflict_warning = ""
    if routing_path != "profile_avoidance" and user_profile and ontology_context:
        diet_allergy_excluded = _get_diet_and_allergy_exclusions(user_profile)
        if diet_allergy_excluded:
            context_lower = ontology_context.lower()
            conflicts = sorted(
                label for label in diet_allergy_excluded
                if label.lower() in context_lower
            )
            if conflicts:
                constraints = []
                diet = user_profile.get("diet_type", "None")
                if diet not in ("None", ""):
                    constraints.append(diet)
                constraints += user_profile.get("allergies", [])
                shown = conflicts[:15]
                more = f" (+{len(conflicts) - 15} more)" if len(conflicts) > 15 else ""
                profile_conflict_warning = (
                    f"\n\n⚠ PROFILE SAFETY CHECK — EXACTLY these context items conflict with "
                    f"the user's {'/'.join(constraints)} profile: {', '.join(shown)}{more}.\n"
                    f"- Do NOT recommend any of them; either omit them or clearly mark them "
                    f"as conflicting rather than presenting them as safe.\n"
                    f"- Attach a '(conflicts with your profile)' note ONLY to a food whose name "
                    f"matches one in the list above. Do NOT infer a conflict for any other food, "
                    f"and do NOT suggest preparing or serving any food with milk, cheese, yoghurt, "
                    f"cream, butter or nuts if those conflict with the profile.\n"
                    f"- Word-substring matches are NOT conflicts: 'eggplant'/'aubergine' contains "
                    f"no egg, 'butternut' squash is not a nut, 'coconut' is not a tree nut."
                )
                prompt += profile_conflict_warning

    _report("Step 3 / 3 — Generating answer...", 75)

    # A food lookup that matched nothing must not reach the LLM: given "No food
    # entity ... was matched", llama3.2 still wrote nutrient values from memory
    # and attributed them to the ontology. The context text is already a
    # user-facing explanation, so answer with it directly.
    # Answers built without the LLM must not be "corrected" by the grounding
    # regeneration below either — that would hand them back to the LLM.
    deterministic_answer = True
    if (
        routing_path in ("nutrient_lookup", "nutrient_comparison", "environmental_impact")
        and ontology_context.startswith("No food entity")
    ):
        if ontology_context.startswith("No food entity from"):
            final_response = (
                "I couldn't find that food in the SustainHealthyFood Ontology, so I can't give "
                "ontology-based values for it. Please ask again using the food's name as it "
                "appears in the ontology (for example \"Lentils red boiled\")."
            )
        else:
            final_response = ontology_context
    elif routing_path == "nutrient_comparison":
        final_response = render_nutrient_comparison_answer(ontology_context)
    elif routing_path in ("false_premise", "allergen_check"):
        final_response = premise["answer"]
    elif routing_path == "nutrient_ranking":
        final_response = render_nutrient_ranking_answer(ontology_context)
    elif (
        routing_path == "nutrient_lookup"
        and (_asked_answer := render_asked_nutrient_answer(user_input, matched_entity_uri, ontology_context))
    ):
        final_response = _asked_answer
    else:
        final_response = _call_llm(prompt, **llm_kwargs)
        deterministic_answer = False

    # Grounding safety net: catch conflicting foods the LLM pulled from its own
    # training knowledge (they're in the answer but never in the retrieved
    # context, so no diet/allergy filter saw them — e.g. "Greek yogurt" for a
    # dairy allergy, "fatty fish" as the only omega-3 source for a vegan).
    # Avoidance answers are meant to name conflicting foods, so skip them.
    ungrounded_conflicts: List[str] = []
    if (
        routing_path != "profile_avoidance" and not deterministic_answer
        and user_profile and len(final_response or "") > 40
    ):
        ungrounded_conflicts = _ungrounded_profile_conflicts(
            final_response, ontology_context, user_profile
        )
        if ungrounded_conflicts:
            regen = prompt + (
                "\n\nCRITICAL CORRECTION: your previous answer referred to "
                + ", ".join(ungrounded_conflicts)
                + " — these are NOT in the ontology context above and conflict with the "
                "user's profile. Rewrite the answer using ONLY foods that appear verbatim "
                "in the ontology context, and do not mention those conflicting foods at all. "
                "If the context has no suitable food, say so plainly."
            )
            final_response = _call_llm(regen, **llm_kwargs)

            # The rewrite can't help when the question itself names the
            # conflicting food (or the context is empty, as on
            # llm_general_knowledge): it just talks about the same food again.
            # An allergen still in the answer is safety-critical, so flag it
            # deterministically instead of trusting a third attempt.
            still = set(_ungrounded_profile_conflicts(final_response, ontology_context, user_profile))
            allergy_hits = []
            for allergy in user_profile.get("allergies", []) or []:
                kws = _ALLERGY_KEYWORD_MAP.get(normalize_text(f"{allergy} allergy"), set())
                hit = sorted(still & kws)
                if hit:
                    allergy_hits.append(f"{allergy.lower()} allergy ({', '.join(hit)})")
            if allergy_hits:
                final_response = (
                    f"⚠️ **Allergy warning:** this answer mentions food that conflicts with your "
                    f"{'; '.join(allergy_hits)}. It is not recommended for you — do not eat it "
                    f"unless your doctor has said it is safe.\n\n{final_response.lstrip()}"
                )

    # food_suitability: prepend the deterministically-computed verdict sentence
    # so a weak router LLM can't soften "not recommended" into "you can sometimes".
    if routing_path == "food_suitability":
        _vm = re.search(r"^VERDICT_SENTENCE:\s*(.+)$", ontology_context, re.M)
        if _vm:
            _verdict = _vm.group(1).strip()
            _body = (final_response or "").strip()
            # Drop a leading verdict line the LLM wrote anyway (starts with
            # Yes/No/You can), so we don't show two.
            _body = re.sub(
                r"^(?:no|yes|you can(?:'t| not)?|it'?s (?:not )?recommended|i (?:would|wouldn'?t))\b[^\n]*\n+",
                "", _body, count=1, flags=re.I,
            ).strip()
            final_response = f"{_verdict}\n\n{_body}" if _body else _verdict

    # Deterministically prepend an intro sentence to health_recommendation
    # answers — the LLM includes one inconsistently on its own (same free-
    # choice problem as the heading-wording inconsistency), since nothing in
    # HEALTH_RECOMMENDATION_PROMPT governs it either way. Wording adapts to
    # which sections are actually present in the real ontology_context (not
    # the LLM's own section headings, which have proven unreliable) so it
    # never claims "recommended foods" when the answer is avoid-only, etc.
    if routing_path == "health_recommendation" and matched_entity and final_response:
        # ontology_context always contains every section the entity has data
        # for, regardless of what the question actually asked — only the
        # prompt's own a)/b)/c) instructions decide which part the answer
        # covers. So detection has to start from the question's own wording,
        # not from what's merely available in the context, or a recommend-
        # only question (e.g. "what foods are recommended for X") would
        # wrongly get the "recommended and not recommended" intro just
        # because the entity also happens to have avoid-data.
        _q_lower = user_input.lower()
        _asks_avoid_kw = any(k in _q_lower for k in ("avoid", "limit", "not recommended", "should not", "shouldn't"))
        _asks_eat_kw = any(k in _q_lower for k in ("recommend", "eat", "include", "help", "good for", "best for"))
        _ctx_has_rec = "RECOMMENDED for " in ontology_context or "EAT IN HIGH AMOUNTS for " in ontology_context
        _ctx_has_avoid = "NOT RECOMMENDED / AVOID for " in ontology_context or "EAT IN LIMITED AMOUNTS for " in ontology_context
        if _asks_avoid_kw and not _asks_eat_kw:
            _has_rec, _has_avoid = False, _ctx_has_avoid
        elif _asks_eat_kw and not _asks_avoid_kw:
            _has_rec, _has_avoid = _ctx_has_rec, False
        else:
            # question's intent is ambiguous or genuinely asks for both —
            # fall back to showing whatever the entity actually has data for.
            _has_rec, _has_avoid = _ctx_has_rec, _ctx_has_avoid
        if _has_rec and _has_avoid:
            _intro_what = "the recommended and not recommended foods"
        elif _has_rec:
            _intro_what = "the recommended foods"
        elif _has_avoid:
            _intro_what = "the foods to avoid"
        else:
            _intro_what = "what the ontology says"
        _intro = f"Based on the SustainHealthyFood Ontology for {matched_entity}, here are {_intro_what}:"
        final_response = f"{_intro}\n\n{final_response.lstrip()}"

    # Re-show _profile_conflict_note()'s warning (stripped from the trace's
    # "Context passed to LLM" display above) after the generated answer
    # instead, worded for the user rather than as an instruction to the LLM.
    # Appended unconditionally when present, regardless of whether the LLM's
    # own prose already mentioned the conflict, so it can't be silently
    # dropped.
    if _pw_match and final_response:
        final_response = (
            f"{final_response.rstrip()}\n\n"
            f"⚠ **Profile note:** this conflicts with your "
            f"{_pw_match.group('constraints')} profile — "
            f"flagged item(s): {_pw_match.group('items')}."
        )

    # Re-show _profile_exclusion_note()'s note (stripped from the trace above)
    # after the generated answer, worded for the user, so the fact that items
    # were silently excluded for their profile isn't lost even if the LLM's
    # own prose never mentions it.
    if _pen_match and final_response:
        final_response = (
            f"{final_response.rstrip()}\n\n"
            f"ℹ️ **Note:** {_pen_match.group('count')} food(s) that would otherwise appear "
            f"above were excluded because they conflict with your "
            f"{_pen_match.group('constraints')} profile."
        )

    if routing_path in _HEALTHCARE_DISCLAIMER_ROUTES and final_response:
        final_response = f"{final_response.rstrip()}\n\n{_HEALTHCARE_DISCLAIMER}"

    _report("Done", 100)

    return {
        "response": final_response,
        "ontology_context": ontology_context,
        "profile_conflict_warning": profile_conflict_warning,
        "ungrounded_conflicts": ungrounded_conflicts,
        "profile_summary": profile_summary,
        "routing_path": routing_path,
        "retrieval_method": retrieval_method,
        "matched_entity": matched_entity,
        "matched_entity_uri": matched_entity_uri,
        "llm_category": llm_category,
        "llm_entity": llm_entity,
        "trace": trace,
    }


if submitted and user_input:
    st.session_state.confirm_clear = False
    with st.spinner(""):
        _status = st.empty()
        _progress = st.progress(0)

        def _status_cb(msg: str, pct: int) -> None:
            _status.markdown(f"**{msg}**")
            _progress.progress(pct)

        result = generate_answer(
            user_input, user_profile, profile_summary,
            is_retry=_is_retry, status_cb=_status_cb,
        )

        _status.empty()
        _progress.empty()

    st.session_state.chat_history.append((user_input, result["response"]))
    st.session_state.response_traces.append(result["trace"])
    st.session_state.last_question = user_input
    st.rerun()
