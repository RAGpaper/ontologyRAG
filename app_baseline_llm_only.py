"""
Streamlit UI for the no-ontology, no-retrieval baseline.

Run from this folder with the RAG_env interpreter:
    <RAG_env>\python.exe -m streamlit run app_baseline_llm_only.py

The sidebar profile has the same fields and options as
retrieval_only_ontology_hybrid_v2.py; the answer comes from baseline_llm_only.
"""

from datetime import datetime

import streamlit as st

import baseline_llm_only as baseline

st.set_page_config(page_title="Baseline LLM (no ontology)", layout="wide")
st.title("Baseline: plain LLM, no ontology, no retrieval")
st.caption(
    f"Model: {baseline.MODEL_NAME} (temperature {baseline.TEMPERATURE}, "
    f"{baseline.NUM_PREDICT} tokens max). The model sees only the profile and your question; "
    "earlier turns are not sent back to it."
)

with st.sidebar:
    st.header("👤 User Profile")
    profile = {}

    with st.expander("🧬 Basics", expanded=True):
        profile["age"] = st.number_input("Age", min_value=0, max_value=120, value=30)
        profile["sex"] = st.selectbox("Sex", ["Female", "Male"])
        profile["life_stage"] = st.selectbox(
            "Life Stage", ["General Adult", "Pregnant", "Lactating", "Pedriatic"])
        profile["weight_kg"] = st.number_input("Weight (kg)", min_value=0.0, value=55.0, max_value=150.0)
        profile["height_cm"] = st.number_input("Height (cm)", min_value=10.0, value=165.0, max_value=215.0)

    with st.expander("🎯 Goals & Lifestyle", expanded=True):
        profile["goal"] = st.selectbox(
            "Goal", ["Not defined", "Maintain weight", "Lose weight", "Gain muscle"])
        profile["activity_level"] = st.selectbox(
            "Activity Level", ["Not defined", "Sedentary", "Moderately Active", "Active", "Athlete"])
        profile["diet_type"] = st.selectbox(
            "Diet Preference", ["None", "Vegetarian", "Vegan", "Halal", "Keto"])
        profile["allergies"] = st.multiselect(
            "Allergies/Intolerances", ["Nuts", "Dairy", "Soy", "Gluten", "Eggs"])

    with st.expander("🌍 Context", expanded=False):
        profile["cultural_background"] = st.text_input("Cultural background")
        profile["occupation"] = st.text_input("Occupation")
        profile["user_concerns"] = st.text_area(
            "Current concerns", placeholder="e.g., I'm worried about my iron levels...")

    with st.expander("🧠 Mood & Well-being", expanded=False):
        profile["stress_level"] = st.selectbox("Stress level", ["Not defined", "Low", "Moderate", "High"])
        profile["sleep_quality"] = st.selectbox("Sleep quality", ["Not defined", "Good", "Fair", "Poor"])
        profile["motivation_level"] = st.selectbox(
            "Energy & motivation", ["Not defined", "High", "Moderate", "Low"])
        profile["mental_persona"] = st.text_input("Relationship with food", value="None")
        profile["appetite_pattern"] = st.selectbox(
            "Typical appetite pattern",
            ["Not defined", "Normal / regular meals", "Craving sugar / caffeine",
             "Low appetite / skipping meals", "Only small/light snacks"])
        profile["support_goal"] = st.selectbox(
            "Current focus / goal",
            ["Not defined", "Feel calmer / reduce stress", "Steady my mood / reduce anxiety",
             "Improve energy / fight fatigue", "Support recovery after training",
             "Just maintain my good routine"])

    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []


def _profile_line(p: dict) -> str:
    return (
        f"Age {p.get('age')}, {p.get('sex')}, {p.get('life_stage')}, "
        f"{p.get('weight_kg')} kg / {p.get('height_cm')} cm | Goal: {p.get('goal')} | "
        f"Activity: {p.get('activity_level')} | Diet: {p.get('diet_type')} | "
        f"Allergies: {', '.join(p.get('allergies', [])) or 'none'} | "
        f"Stress: {p.get('stress_level')} | Sleep: {p.get('sleep_quality')}"
    )


def _export_history() -> list:
    """History for the shared exporters, with the profile each answer was
    generated with written above it. The exporters' header only shows the
    sidebar profile at export time, so a session where the profile changed
    between questions (e.g. a different allergy per trap question) otherwise
    exported every turn under the last profile."""
    rows = []
    for turn in st.session_state.history:
        q, a = turn[0], turn[1]
        if len(turn) > 2:   # turns asked before this change carry no snapshot
            a = f"[Profile used for this answer — {_profile_line(turn[2])}]\n\n{a}"
        rows.append((q, a))
    return rows


with st.sidebar:
    st.header("💾 Save session")
    _stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "⬇️ .txt",
        data=baseline.build_chat_txt(_export_history(), profile),
        file_name=f"baseline_chat_export_{_stamp}.txt",
        mime="text/plain",
        disabled=not st.session_state.history,
        use_container_width=True,
    )
    st.download_button(
        "⬇️ .html",
        data=baseline.build_chat_html(_export_history(), profile),
        file_name=f"baseline_chat_export_{_stamp}.html",
        mime="text/html",
        help="Open in a browser and use File → Print → Save as PDF",
        disabled=not st.session_state.history,
        use_container_width=True,
    )

with st.expander("Profile text sent to the model"):
    st.write(baseline.build_profile_summary_text(profile))

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn[0])
    with st.chat_message("assistant"):
        if len(turn) > 2:
            st.caption(f"Profile used: {_profile_line(turn[2])}")
        st.write(turn[1])

preset = st.selectbox("Try a predefined question", [""] + baseline.PREDEFINED_QUESTIONS)
with st.form("ask", clear_on_submit=True):
    typed = st.text_input("Or ask your own question")
    submitted = st.form_submit_button("Send")

question = typed.strip() or preset
if submitted and question:
    with st.spinner("Generating answer..."):
        result = baseline.generate_answer(question, profile)
    # Snapshot the profile (a copy, so later sidebar edits don't change it).
    st.session_state.history.append(
        (question, result["response"], {**profile, "allergies": list(profile["allergies"])})
    )
    st.rerun()
