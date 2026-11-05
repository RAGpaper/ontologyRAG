# Ontology-Grounded RAG for Personalised Nutrition Advice

This repository contains the code and comparison data for an ontology-grounded
retrieval-augmented generation (RAG) system that answers nutrition questions
based on a user's profile. Answers are grounded in the **SustainHealthyFood
Ontology (SHFO)** [1].
The repository also contains an **LLM-only baseline**, which has no ontology or
retrieval. It is used to compare against the grounded system.

## How the system works

1. **User profile.** The user enters their profile in the sidebar: basic
   attributes, goals and lifestyle, context, and mood and well-being. A rule-based
   resolver maps the mood fields onto a mental-state chart.
2. **Question classification.** An LLM router assigns one of nine categories to
   the question and extracts the food, recipe or concept it mentions.
3. **Routing.** Rule-based checks refine or override that category and select one
   of 25 routing paths.
4. **Retrieval.** Most paths query the ontology graph directly with `rdflib`.
   Vector search over three Chroma collections is used as a fallback and for
   additional context.
5. **Answer generation.** Each routing path has its own prompt. Some answers are
   computed in Python without the LLM: nutrient comparisons and rankings,
   allergen checks and false-premise checks. Generated answers are checked
   against the user's diet and allergies.
6. **Traceability.** Each answer shows its routing path, retrieval method,
   matched ontology entity and the context passed to the LLM.

## Repository contents

| File | Description |
|---|---|
| `retrieval_only_ontology_hybrid_v2.py` | The ontology-grounded system (Streamlit app) |
| `SHFO_extension_v2.ttl` | The SustainHealthyFood Ontology, extended version used by the system |
| `mental_state_model.jsonl` | Mood and well-being state chart, embedded into a vector store |
| `embeddingsStorage_ontology_general.py` | Builds the general ontology vector store |
| `embeddingsStorage_ontology_health.py` | Builds the health ontology vector store |
| `embeddingsStorage_with_mentalstates_v2.py` | Builds the mental-state vector store |
| `baseline_llm_only.py` | LLM-only baseline: same model and profile, no ontology or retrieval |
| `app_baseline_llm_only.py` | Streamlit interface for the baseline |
| `requirements.txt` | Python dependencies |
| `ontology rag_ *.jsonl` | Answers from the grounded system for the comparison question sets |
| `baseline_chat_export_*.txt` | Answers from the baseline for the same question sets |
| `Evaluation of Ontology-Grounded Nutrition Recommendations - Google Forms.pdf` | Questionnaire distributed to experts for the evaluation|

## Requirements

- Python 3.11
- [Ollama](https://ollama.com), with these two models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Installation

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

The system was developed and tested with Python 3.11.13, streamlit 1.46.1,
rdflib 7.1.4, langchain 0.3.26, langchain-community 0.3.26, langchain-ollama
0.3.3, langchain-chroma 0.2.4 and chromadb 1.0.13.

## Building the vector stores

The vector stores are not included in the repository. Build them once before
running the system. Make sure Ollama is running, then run these commands from
the repository folder:

```bash
python embeddingsStorage_ontology_general.py
python embeddingsStorage_ontology_health.py
python embeddingsStorage_with_mentalstates_v2.py
```

This creates three folders: `ontology_chroma_db_general`,
`ontology_chroma_db_health` and `mental_state_chroma_db`.

## Running the system

Ontology-grounded system:

```bash
streamlit run retrieval_only_ontology_hybrid_v2.py
```

LLM-only baseline:

```bash
streamlit run app_baseline_llm_only.py
```

Both open in the browser at `http://localhost:8501`. The baseline reads the
predefined questions and profile text from `retrieval_only_ontology_hybrid_v2.py`,
so both files must be in the same folder.

## Comparison data

The grounded system and the baseline were asked the same three question sets:

- **Factual questions:** nutrient values and general nutrition knowledge
- **Food suitability questions:** whether a food suits the user's profile
- **Trap questions:** questions with a false premise or an allergy conflict

`ontology rag_ *.jsonl` contains one JSON object per answer from the grounded
system. Each object includes the user profile, the question, the answer, the
routing path, the matched ontology entity and the ontology context.

`baseline_chat_export_*.txt` contains the baseline's answers. Where the profile
changed between questions, each answer begins with the profile used for it.

## Settings

- Model: `llama3.2` through Ollama, at `http://localhost:11434`
- Temperature: 0 on almost all routes. General cooking questions use 0.5, and a
  retry after an empty answer uses 0.3.
- Embedding model: `nomic-embed-text`
- Baseline: `llama3.2`, temperature 0, at most 300 tokens per answer


## License

The original Python source code in this repository is released under the MIT License (see [LICENSE](LICENSE)).

The SustainHealthyFood Ontology (SHFO) and its extensions are separate from the software implementation. Their licensing is subject to the rights of the ontology authors and any incorporated third-party data.

The ontology incorporates nutritional information from NEVO-online version 2023/8.0, maintained by RIVM. NEVO-derived data remain subject to RIVM's applicable copyright and conditions of use and are not licensed under MIT by this repository.

The inclusion of third-party data does not imply permission for unrestricted redistribution or reuse. See the Data Sources and Acknowledgments section below.

## Data Sources and Acknowledgments

### SustainHealthyFood Ontology (SHFO)

The ontology-grounded RAG system presented in this repository builds upon the SustainHealthyFood Ontology (SHFO), originally described by Xhani and Guizzardi (2025) [1].

The repository includes `SHFO_extension_v2.ttl`, an extended version of SHFO used for ontology-based retrieval and personalised nutrition recommendations.

### Nutritional Data — NEVO

The ontology incorporates food composition data from the Dutch Food Composition Database (NEVO), developed and maintained by the Rijksinstituut voor Volksgezondheid en Milieu (RIVM), the Dutch National Institute for Public Health and the Environment.

The nutritional information used in this ontology originates from **NEVO-online version 2023/8.0**.

**Reference:**

Rijksinstituut voor Volksgezondheid en Milieu (RIVM). (2023). *NEVO online, version 2023/8.0*. RIVM, Bilthoven. https://www.rivm.nl/en/dutch-food-composition-database

**Required NEVO source attribution:**

NEVO online, version 2023/8.0. RIVM, Bilthoven.

The NEVO-derived data remain subject to RIVM's copyright and applicable conditions of use. The software license of this repository does not extend to NEVO-derived data.

For further information, see the [NEVO 2023/8.0 conditions of use](https://www.rivm.nl/sites/default/files/2023-11/Conditions%20of%20use%20NEVO-online%202023%20dataset.pdf).

## Citation

[1] Xhani, D., & Guizzardi, R. (2025, October). Ontology for Personalized Recipe Recommendations Supporting Sustainability, Personal Preferences, and Health Restrictions. In International Conference on Conceptual Modeling (pp. 376-392). Cham: Springer Nature Switzerland.
