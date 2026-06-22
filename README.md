# Paper-Insight

A transformer-based pipeline for semantic search and structured analysis of scientific papers. Given a corpus of arXiv abstracts, it retrieves relevant papers, extracts structured information, answers questions grounded in the retrieved text, and evaluates its own extraction quality.

## Architecture

The system is a four-stage pipeline that separates **retrieval** (finding relevant papers) from **generation** (analyzing them):

| Stage | What it does | How |
|-------|-------------|-----|
| **1. Retrieval** | Find papers relevant to a query | Pretrained sentence-transformer encoder (`all-MiniLM-L6-v2`) embeds each abstract into a 384-dim vector; ranked by cosine similarity |
| **2. Extraction** | Pull structured fields (method / results / limitations) from a paper | LLM (Gemini) with a JSON-constrained prompt |
| **3. Grounded Q&A** | Answer a question using the retrieved papers, with citations | Retrieval-Augmented Generation: retrieved abstracts are placed in the prompt as context; the model answers only from them |
| **4. Evaluation** | Measure extraction quality and compare prompt variants | LLM-as-judge scoring + structural validation over a fixed test set |

### Why two different models?
Retrieval and generation are different jobs. The encoder turns text into vectors for fast similarity search over the whole corpus; the generative LLM reads retrieved text and produces written analysis. Embeddings are used to *find* papers; the papers' *text* is what grounds the generated answers.

## Key design choices

- **Grounding to prevent hallucination.** In Stage 3, the model is instructed to answer only from the retrieved papers and to explicitly decline ("the retrieved papers do not address this question") when they lack the answer — so it doesn't fabricate from training memory.
- **LLM-as-judge evaluation.** Stage 4 uses a structural gate (valid JSON, all fields present) followed by an LLM judge (1–5 quality score) to compare prompt variants — the technique used in practice for evaluating LLM outputs.
- **Model-agnostic.** The pipeline was migrated across LLM providers by changing only the generation function; the architecture is unchanged.

## Results

Evaluated two extraction prompt variants on a held-out test set using the Stage 4 harness:

- Both variants produced **100% structurally-valid output** (parseable JSON, all fields populated).
- Judge-rated extraction quality: **~4.5–4.8 / 5** for both.
- The simpler prompt performed at least as well as the more detailed one — indicating that prompt complexity does not automatically improve quality, and that evaluation (not assumption) should guide prompt design.

## Known limitations & future work

- **Corpus coverage bounds retrieval quality.** The demo corpus is fetched from arXiv by keyword and sorted by recency, so it is a loose topical sample; a specific query may have no closely relevant paper. A curated, stable corpus would improve grounding and reproducibility.
- **Small evaluation set.** The eval set is small (constrained by free-tier API rate limits); a larger set with higher API quota would give statistically firmer comparisons.
- **LLM-as-judge is approximate.** Using an LLM to grade another LLM is a scalable but imperfect signal; validating the judge against human labels would increase confidence.

## Stack
Python · sentence-transformers · scikit-learn · Gemini API · arXiv API

## Running
The notebook (`paper_insight_stage1.ipynb`) runs in Google Colab. A Gemini API key must be set in Colab Secrets as `GEMINI_API_KEY`. Run cells top to bottom.
