"""
Baseline: plain LLM, no ontology, no retrieval
==============================================
Comparison system for retrieval_only_ontology_hybrid_v2.py. It differs from the
app in exactly one respect: the model sees only the user's profile and the
question. There is no question router, no ontology/SPARQL/Chroma context, no
mental-state chart, no diet/allergy safety net, no grounding regeneration.

Held identical to the app, so any difference in results is due to the ontology:
  - model            : llama3.2 via the same langchain Ollama wrapper
  - decoding         : temperature 0, num_predict 300 (the app's settings on
                       its grounded routes)
  - question list    : PREDEFINED_QUESTIONS / QUESTION_CATEGORIES, extracted
                       from the app's source (plain data, not computation)
  - prompt wording   : BASELINE_PROMPT is PROFILE_REASONING_PROMPT with the
                       ontology-specific parts removed and nothing else changed

Deliberately NOT shared with the app, unlike an earlier version of this file:
  - profile text     : build_profile_summary_text() below is its own,
                       independent implementation. It reports only the raw
                       profile fields the user entered (age, sex, life stage,
                       weight, height, goal, diet, allergies, activity level,
                       ...) and none of the app's own computed nutrition
                       science (BMI, Harris-Benedict daily calories, EFSA
                       macro ranges). Reusing those calculators would hand
                       the LLM pre-computed, pre-labelled ("EFSA...") correct
                       answers for calorie/macro questions before it even
                       starts generating - which tests whether the model can
                       reformat given numbers, not whether it can reason about
                       nutrition on its own. A true no-computation baseline
                       has to do any such reasoning itself, hallucinations
                       included.

The question list above is still extracted from the app's *source* with ast
rather than imported (importing the app would load the ontology and
Streamlit) or copied (a copy drifts when the app is edited). If a name is
renamed or removed in the app, loading fails loudly.
"""

import ast
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    from langchain_community.llms import Ollama

APP_SOURCE = Path(__file__).with_name("retrieval_only_ontology_hybrid_v2.py")

MODEL_NAME = "llama3.2"
TEMPERATURE = 0.0
NUM_PREDICT = 300

ROUTING_PATH = "llm_only_baseline"

_SHARED_NAMES = (
    "PREDEFINED_QUESTIONS",
    "QUESTION_CATEGORIES",
    "build_chat_txt",
    "build_chat_html",
)


def _load_shared_from_app(path: Path = APP_SOURCE) -> dict:
    """Exec the top-level definitions named in _SHARED_NAMES from the app file,
    without running the rest of the module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = set(_SHARED_NAMES)
    nodes, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
            found.add(node.name)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in wanted
        ):
            nodes.append(node)
            found.add(node.targets[0].id)
    missing = wanted - found
    if missing:
        raise RuntimeError(f"{path.name} no longer defines: {sorted(missing)}")
    namespace = {"Optional": Optional, "Dict": Dict, "List": List, "Tuple": Tuple, "datetime": datetime}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


_shared = _load_shared_from_app()
PREDEFINED_QUESTIONS = _shared["PREDEFINED_QUESTIONS"]
QUESTION_CATEGORIES = _shared["QUESTION_CATEGORIES"]
build_chat_txt = _shared["build_chat_txt"]
build_chat_html = _shared["build_chat_html"]


def build_profile_summary_text(user_profile: dict) -> str:
    """Plain-text profile summary fed to the LLM prompt - independent of the
    app's build_profile_summary_text(). Reports only what the user actually
    entered; performs no calculation of its own (no BMI, no calories, no
    macro targets). See the module docstring for why."""
    concern_text = (
        f"The user's main concerns are: {user_profile['user_concerns']}"
        if user_profile["user_concerns"] else "The user did not indicate any specific concerns."
    )
    return (
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


# PROFILE_REASONING_PROMPT minus: the chat-history block, the ontology-context
# block, and the LIFE STAGE / MENTAL STATE CHART context instructions. The
# app's "apply general knowledge where the ontology does not cover it" line is
# kept as an unconditional general-knowledge instruction: without it llama3.2
# refuses general questions (e.g. food composition) as outside the profile.
BASELINE_PROMPT = """
You are a knowledgeable dietary advisor who gives personalized nutrition guidance.

IMPORTANT: The user's complete profile is already provided below. Do NOT ask for any profile information — use what is given.

User Profile:
{profile_summary}

User question:
{question}

Instructions:
- Use the profile above to answer the question directly and personally. The profile is complete — never ask the user for more details.
- Draw on ALL relevant profile fields in your answer:
  * Basics: age, sex, life stage (General Adult / Pregnant / Lactating / Pedriatic), weight, height
  * Goals & lifestyle: goal, activity level, diet preference, allergies/intolerances
  * Mental health & lifestyle: stress level, sleep quality, motivation/energy level, mental persona, appetite pattern today, support goal
- The mental health & lifestyle fields matter especially for questions about energy, mood, focus, or wellbeing — always factor them in.
- Be specific: quote the user's actual profile values in your answer (e.g. "Given your high stress level and irregular sleep..." 
or "Since you are lactating and your goal is to maintain weight...").
- Keep the response practical, concise, and easy to read.
- Answer from your general nutrition knowledge. If the question is general (e.g. about a food, nutrient or recipe) rather than about the user, answer it directly and mention profile values only where they are relevant.
- For food composition questions, give approximate values per 100 g and say they are approximate.
"""

_llm = Ollama(model=MODEL_NAME, temperature=TEMPERATURE, num_predict=NUM_PREDICT)


def generate_answer(
    user_input: str,
    user_profile: dict,
    profile_summary: Optional[str] = None,
    status_cb: Optional[Callable[[str, int], None]] = None,
) -> dict:
    """Same call shape and return keys as the app's generate_answer(), so eval
    code can treat the two systems alike. There is no retrieval, so
    ontology_context is always empty."""
    if profile_summary is None:
        profile_summary = build_profile_summary_text(user_profile)
    if status_cb:
        status_cb("Generating answer (no retrieval)...", 50)

    prompt = BASELINE_PROMPT.format(profile_summary=profile_summary, question=user_input)
    response = _llm.invoke(prompt)

    if status_cb:
        status_cb("Done", 100)
    return {
        "response": response,
        "ontology_context": "",
        "profile_conflict_warning": "",
        "ungrounded_conflicts": [],
        "profile_summary": profile_summary,
        "routing_path": ROUTING_PATH,
        "retrieval_method": "none",
        "matched_entity": "",
        "matched_entity_uri": "",
        "llm_category": "",
        "llm_entity": "",
        "trace": None,
    }


if __name__ == "__main__":
    print("baseline_llm_only.py is a library, not an app. Use one of:")
    print("  python -m streamlit run app_baseline_llm_only.py   (interactive UI)")
    print("  python eval_baseline_llm_only.py                   (questions x personas eval)")
