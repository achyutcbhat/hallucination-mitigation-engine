# Mitigating Hallucination Snowballing in Multi-Agent LLMs via Heterogeneous Semantic Clustering and Dual Guardrails

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework: FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Inference: Ollama](https://img.shields.io/badge/Inference-Ollama-orange.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of the **Dual-Guardrail Heterogeneous Multi-Agent Consensus Framework** for interrupting and preventing hallucination cascading in Large Language Model (LLM) agent architectures.

---

## 📌 Abstract
While multi-agent LLM systems improve problem-solving via peer review and consensus, they remain critically vulnerable to **hallucination snowballing**—where an early ungrounded statement or adversarial premise is mutually reinforced across homogeneous agents into a false consensus. 

This framework introduces a **7-stage dual-guardrail orchestration pipeline**:
1. **Guardrail #1 (Pre-Generation)** intercepts ambiguous, underspecified, or temporally drifted queries before consuming GPU resources.
2. **Heterogeneous Model Ensemble** disrupts cognitive uniformity by drawing candidates across three distinct model families (Alibaba Qwen-2.5-3B, Google Gemma-2-2B, and DeepSeek-R1-1.5B with chain-of-thought stripping).
3. **Graph Semantic Clustering** uses dense embeddings (`all-MiniLM-L6-v2`) and connected-component graph analysis to isolate primary consensus topologies and prune outlier drafts.
4. **Guardrail #2 (Post-Generation)** validates atomic claims against local knowledge bases (Qdrant vector store, Wikipedia, Google KG) using cross-encoder **DeBERTa-v3 NLI**.
5. **Adaptive Decision Routing** automatically triggers live search grounding (DuckDuckGo + Trafilatura) for uncertain or contested claims, conservatively refusing unresolvable claims.

---

## 🏛️ System Architecture

```
                       [ Incoming User Query ]
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │  STAGE 1: Input Understanding & Multi-Hop Decomposition │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │  STAGE 2: Pre-Generation Risk Assessment (GUARDRAIL #1) │
      │  (Ambiguity 35%, Context 30%, Drift 20%, Difficulty 15%)│
      └───────┬──────────────────────────────────────┬─────────┘
              │ [Risk >= 0.65 & Unresolved]          │ [Risk < 0.65]
              ▼                                      ▼
      [ Conservative Refusal ]     ┌────────────────────────────────────┐
                                   │  STAGE 3: Heterogeneous Ensemble   │
                                   │  • Qwen-2.5-3B (Factual Breadth)   │
                                   │  • Gemma-2-2B  (Precision)         │
                                   │  • DeepSeek-R1 (Reasoning Traces)  │
                                   │  (T = {0.70, 0.85} -> 6 Candidates)│
                                   └─────────────────┬──────────────────┘
                                                     │
                                                     ▼
                                   ┌────────────────────────────────────┐
                                   │  STAGE 4: Graph Semantic Clustering│
                                   │  (NetworkX Cosine Sim >= 0.72)     │
                                   │  • Primary Consensus Cluster (C*)  │
                                   │  • Prune Outlier Hallucinations    │
                                   └─────────────────┬──────────────────┘
                                                     │
                                                     ▼
                                   ┌────────────────────────────────────┐
                                   │  STAGE 5: Synthesis & Claim Split  │
                                   │  • Master Response via Qwen-2.5    │
                                   │  • Decompose into Atomic Claims    │
                                   └─────────────────┬──────────────────┘
                                                     │
                                                     ▼
                                   ┌────────────────────────────────────┐
                                   │  STAGE 6: Offline NLI Verification │
                                   │  (GUARDRAIL #2 - DeBERTa-v3)       │
                                   │  • Qdrant + Google KG + Wikipedia  │
                                   └─────────────────┬──────────────────┘
                                                     │
                                                     ▼
                                   ┌────────────────────────────────────┐
                                   │  STAGE 7: Corrective Decision Route│
                                   └───────┬──────────────┬─────────────┘
                                           │              │
                     ┌─────────────────────┘              └─────────────────────┐
                     ▼                                                          ▼
        [ Score >= 0.75 & No Contradiction ]                 [ 0.45 <= Score < 0.75 or Contradicted ]
                     │                                                          │
                     ▼                                                          ▼
             ★ DIRECT VERIFIED ★                                    [ Dynamic Web Fallback ]
                                                                    (DuckDuckGo Live Snippets)
                                                                                │
                                                            ┌───────────────────┴───────────────────┐
                                                            ▼                                       ▼
                                                  [ Re-score >= 0.75 ]                    [ Re-score < 0.45 ]
                                                            │                                       │
                                                            ▼                                       ▼
                                                    ★ WEB VERIFIED ★                     ★ CONSERVATIVE REFUSAL ★
```

---

## 📊 Empirical Benchmark Results ($N = 200$)

Evaluated on a comprehensive benchmark of **200 challenging queries** (including **115 deceptive trap queries** with false presuppositions, non-existent historical events, and fictitious scientific claims):

| Architecture / System | Factual Accuracy (≥0.75) ↑ | Hallucination Rate (<0.50) ↓ | Trap Block Rate (TBR %) ↑ | F1-Score ↑ | Mean Score (μ ± σ) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multi-Agent Engine (Ours)** | **80.5%** (161/200) | **15.5%** (31/200) | **77.4%** (89/115) | **0.821** | **0.794 ± 0.30** |
| Single Qwen-2.5-3B | 79.5% (159/200) | 18.5% (37/200) | 74.8% (86/115) | 0.803 | 0.757 ± 0.36 |
| Single Gemma-2-2B | 62.0% (124/200) | 37.0% (74/200) | 53.0% (61/115) | 0.623 | 0.573 ± 0.42 |
| Single DeepSeek-R1-1.5B | 39.5% (79/200) | 58.5% (117/200) | 27.8% (32/115) | 0.399 | 0.383 ± 0.43 |

* **Trap Block Rate (TBR %)**: Measures the system's ability to identify false premises and issue a correction or refusal rather than hallucinating answers.
* **Latency Profiling**: The system operates with deterministic baseline latency ($\approx 0\sigma$), selectively investing compute in the $+2\sigma$ tail only when multi-hop live web verification is strictly necessary.

---

## 💻 Tech Stack

* **Inference Runtime**: Python 3.11+, PyTorch (optimized for 4GB VRAM with CUDA cache clearing).
* **Local LLM Serving**: [Ollama](https://ollama.com/) (`http://localhost:11434`).
* **Models**:
  * `qwen2.5:3b` (Alibaba)
  * `gemma2:2b` (Google)
  * `deepseek-r1:1.5b` (DeepSeek)
* **NLP & Parsing**: spaCy (`en_core_web_sm`).
* **Embeddings & NLI**: `sentence-transformers/all-MiniLM-L6-v2`, `cross-encoder/nli-deberta-v3-base`.
* **Graph Clustering**: NetworkX, NumPy, Scikit-Learn.
* **Vector Store**: Qdrant (Embedded local storage).
* **Web Grounding**: DuckDuckGo Search, Trafilatura.
* **API & Validation**: FastAPI, Uvicorn, Pydantic v2.

---

## 🚀 Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/hallucination-mitigation-engine.git
cd hallucination-mitigation-engine
```

### 2. Set Up Python Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Pull Required LLMs via Ollama
Ensure [Ollama](https://ollama.com/) is installed and running:
```bash
ollama pull qwen2.5:3b
ollama pull gemma2:2b
ollama pull deepseek-r1:1.5b
```

### 5. Configure Environment Variables
Create a `.env` file in the project root:
```env
GOOGLE_KG_API_KEY=your_optional_google_kg_key
GEMINI_API_KEY=your_optional_gemini_key
```

### 6. (Optional) Seed Local Evidence Store
```bash
python scripts/seed_qdrant.py
```

---

## 🖥️ Usage

### Interactive Command-Line Interface (CLI)
Run queries directly in your terminal:
```bash
python src/main.py
```

### REST API Service
Start the FastAPI server:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
* **Health Check**: `GET http://localhost:8000/health`
* **Query Endpoint**: `POST http://localhost:8000/query`
  ```json
  {
    "query": "What is the capital of Australia?",
    "interactive_hitl": false
  }
  ```

---

## 📁 Repository Structure

```
├── config/
│   ├── settings.py                # Centralized thresholds (0.65, 0.72, 0.75, 0.45) & paths
│   └── logging_config.py          # Structured logging
├── data/
│   └── qdrant_storage/            # Local vector evidence database
├── scripts/
│   └── seed_qdrant.py             # Offline knowledge seeding script
├── src/
│   ├── main.py                    # Entry point (CLI & FastAPI)
│   ├── pipeline.py                # 7-stage master orchestration logic
│   ├── schemas/                   # Pydantic data schemas
│   ├── stage_01_input_processing/ # NLP tokenization, ambiguity & decomposition
│   ├── stage_02_risk_assessment/  # Guardrail #1 (Pre-generation screening)
│   ├── stage_03_multiagent_engine/# Ollama heterogeneous candidate generation
│   ├── stage_04_graph_clustering/ # Semantic graph & outlier pruning
│   ├── stage_05_response_polishing/# Master consensus synthesis & claim extraction
│   ├── stage_06_verification/     # Guardrail #2 (DeBERTa-v3 NLI checking)
│   ├── stage_07_decision_routing/ # Tri-state routing & live web fallback
│   └── utils/
│       └── memory_manager.py      # CUDA memory clearing & garbage collection
└── requirements.txt
```

---

## 👥 Authors & Affiliation

* **Achyut Chandrashekhar** — *Department of Computer Engineering, CMR University, Bengaluru, India* (`achyut.chandrashekhar@cmr.edu.in`)
* **Krishna Patil** — *Department of Computer Engineering, CMR University, Bengaluru, India* (`krishna.patil@cmr.edu.in`)
* **Priya Mendegar** — *Department of Computer Engineering, CMR University, Bengaluru, India* (`priya.mendegar@cmr.edu.in`)
* **Dr. Kiran Kumari Patil** — *Department of Computer Engineering, CMR University, Bengaluru, India* (`kirankumari.p@cmr.edu.in`)

---

## 📜 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@inproceedings{chandrashekhar2026mitigating,
  title={Mitigating Hallucination Snowballing in Multi-Agent LLMs via Heterogeneous Semantic Clustering and Dual Guardrails},
  author={Chandrashekhar, Achyut and Patil, Krishna and Mendegar, Priya and Patil, Kiran Kumari},
  booktitle={Proceedings of the Conference on Large Language Models and Multi-Agent Systems},
  year={2026}
}
```

---
