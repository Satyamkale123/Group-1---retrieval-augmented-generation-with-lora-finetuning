# DATA 612 — Final Project: Code Submission

**Project Title:** A Study of Retrieval-Augmented Generation with LoRA Fine-Tuning on Open-Domain Question Answering

**Team:** Shruti Gajipara · Mayur Sangle · Satyam Kale · Parth Maheshwari · Adwait Gaur

**Submission Date:** May 7, 2026

---

## 1. What this submission contains

This archive contains all code developed for the project, organized by team member contribution. The five Jupyter notebooks together implement the complete experimental pipeline described in the final report.

```
DATA612_FinalCode/
├── README.md                          (this file)
├── DATA612_corpus_scaling.ipynb       Build the 525k-passage Wikipedia FAISS index
├── DATA612_Mayur.ipynb                Baselines 1 (zero-shot) and 2 (few-shot)
├── DATA612_Satyam.ipynb               RAG pipeline (retriever + Mistral-7B)
├── DATA612_Parth.ipynb                LoRA fine-tuning + RAG + LoRA evaluation
├── DATA612_Adwait.ipynb               Evaluation harness (EM / Substring / F1)
└── results/                           Final prediction JSON files (optional)
    ├── baseline1_results.json
    ├── baseline2_results.json
    ├── rag_full_results.json
    └── rag_lora_full_results.json
```

---

## 2. Quick summary of results

All four configurations were evaluated on the same fixed 500-question slice of TriviaQA validation (random seed = 42). The evaluation harness in `DATA612_Adwait.ipynb` produces the numbers below from the prediction files in `results/`.

| Configuration | Strict EM | Substring EM | F1 |
|---|---|---|---|
| Baseline 1 (zero-shot)   | 0.358 | 0.746 | 0.492 |
| Baseline 2 (few-shot)    | 0.292 | 0.728 | 0.426 |
| RAG (no fine-tune)       | 0.014 | 0.588 | 0.116 |
| RAG + LoRA               | 0.256 | 0.726 | 0.418 |

The final report (submitted separately) discusses these results in full.

---

## 3. How to run the code

The notebooks were developed and tested on **Google Colab Pro**. Running them from scratch requires GPU access and takes approximately 2-3 hours of compute time end to end. We strongly recommend Colab Pro with an A100 GPU.

### 3.1 Required environment

- **Platform:** Google Colab Pro (Python 3.12)
- **GPU:** NVIDIA A100 (for LoRA training and final evaluations) or T4 (for corpus building and retrieval testing)
- **Disk:** ~5 GB Google Drive storage for the corpus index and LoRA adapter
- **HuggingFace account:** Needed to download Mistral-7B-Instruct-v0.2 (no gating, but the token is required). Create a free account at huggingface.co and store your token in Colab Secrets as `HF_TOKEN`.

### 3.2 Required Python packages

The first cell of every notebook installs these. Versions matter — older versions of `bitsandbytes` are incompatible with the current `transformers` release.

```python
!pip install -q transformers==4.44.2 accelerate==0.33.0 datasets==2.21.0
!pip install -q peft==0.12.0 trl==0.9.6
!pip install -q sentence-transformers faiss-cpu==1.8.0
!pip install -q "bitsandbytes>=0.46.1"
```

⚠️ **After installing `bitsandbytes`, restart the Colab runtime** (Runtime → Restart session) before continuing. This is required because the bitsandbytes shared library must reload.

### 3.3 Drive folder structure

The notebooks expect this folder structure in your Google Drive:

```
/content/drive/MyDrive/
├── wiki_corpus_50k/                            (~1.9 GB — built by notebook 1)
│   ├── wiki.faiss                              FAISS IndexFlatIP
│   ├── passages.json                           Passage metadata (525,731 entries)
│   └── embeddings.npy                          384-dim vectors
├── lora_checkpoints/                           (~30 MB — built by notebook 4)
│   └── mistral-lora-triviaqa-final/
│       └── checkpoint-625/                     Final LoRA adapter
├── baseline1_results.json                      Output of notebook 2
├── baseline2_results.json                      Output of notebook 2
├── rag_full_results.json                       Output of notebook 3
└── rag_lora_full_results.json                  Output of notebook 4
```

### 3.4 Step-by-step reproduction

Run the notebooks **in this order**. Each builds on artifacts from the previous notebooks.

#### Step 1 — Build the retrieval corpus (~45 min on T4)

Open `DATA612_corpus_scaling.ipynb`, set runtime to T4, and run all cells.

This notebook streams 50,000 articles from the wikimedia/wikipedia 2023 English dump, chunks them into 525,731 passages of ~100 words each, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and saves the resulting FAISS index to Drive.

**Output:** `MyDrive/wiki_corpus_50k/` (~1.9 GB)

#### Step 2 — Run the two baseline configurations (~30 min on A100)

Open `DATA612_Mayur.ipynb`, set runtime to A100, and run all cells.

This notebook loads Mistral-7B-Instruct in 4-bit quantization, then runs Baseline 1 (zero-shot prompting) and Baseline 2 (few-shot prompting with 5 in-context examples) on the same fixed 500-question evaluation slice.

**Output:** `MyDrive/baseline1_results.json` and `MyDrive/baseline2_results.json`

#### Step 3 — Run RAG without fine-tuning (~30 min on A100)

Open `DATA612_Satyam.ipynb`, set runtime to A100, and run all cells.

This notebook loads the FAISS index from Step 1 and the Mistral-7B model, then runs the RAG pipeline (retrieve top-5 passages, prepend to prompt, generate answer) on the 500-question evaluation slice.

**Output:** `MyDrive/rag_full_results.json`

#### Step 4 — Train LoRA and run RAG + LoRA (~60 min on A100)

Open `DATA612_Parth.ipynb`, set runtime to A100, and run all cells in order.

This notebook does two things:

1. **Fine-tunes** a LoRA adapter (rank 8, alpha 16, q_proj + v_proj) for 625 steps on 10,000 randomly sampled TriviaQA training examples. Loss decreases from 3.23 → 0.87 over ~30 minutes.
2. **Evaluates** the resulting RAG + LoRA configuration on the same 500-question slice used for the other configurations.

**Output:** `MyDrive/lora_checkpoints/mistral-lora-triviaqa-final/checkpoint-625/` and `MyDrive/rag_lora_full_results.json`

#### Step 5 — Compute final metrics (<1 min, CPU is fine)

Open `DATA612_Adwait.ipynb` and run all cells.

This notebook reads all four prediction JSON files and computes Strict Exact Match, Substring Match, and F1 using a canonical SQuAD-style evaluation harness with multi-alias matching. It produces the comparison table reported in the paper.

**Output:** Console-printed comparison table.

### 3.5 Reproducibility notes

- **Random seed:** 42, used for the evaluation slice selection. Identical across all configurations.
- **Decoding:** greedy (`do_sample=False`) throughout, ensuring deterministic outputs.
- **Quantization:** Mistral-7B is loaded with NF4 4-bit quantization. Different quantization schemes will produce slightly different numerical results.
- **Hardware:** A100 vs T4 should not affect numerical results, only wall-clock time.

---

## 4. Verifying results without re-running

If you want to verify our numbers without running the entire pipeline (which requires GPU access and several hours), the `results/` folder contains the prediction files for all four configurations. These can be evaluated directly:

```text
Open DATA612_Adwait.ipynb, point CONFIGS to results/*.json files,
and run all cells. Takes < 1 minute on CPU.
```

The harness will read the JSON files and print the same numerical results reported in our final report.

---

## 5. Implementation details and design choices

A few decisions worth flagging for the grader:

- **Mistral-7B-Instruct-v0.2** is used as the generator throughout. It was chosen because it is fully open-source (no gating), fits in 4-bit on a single Colab GPU, and is a reasonable proxy for the kinds of mid-size open LLMs that practitioners actually deploy.
- **The 50k-article corpus subset** is a deliberate scope choice driven by Colab compute and Drive storage budget. The final report acknowledges this as a primary limitation that bottlenecks RAG performance.
- **The LoRA hyperparameters** (rank 8, alpha 16, target modules q_proj + v_proj, learning rate 2e-4 with cosine decay, paged 8-bit AdamW) follow Hu et al. (2021) conventions for 7B-parameter models. We did not perform a hyperparameter sweep.
- **Greedy decoding** was used throughout for reproducibility. The final report discusses how stop-token decoding might further improve Strict EM but was outside scope.

---

## 6. Acknowledgment of tool use

Per course policy: the team used Anthropic's Claude as an AI assistant for project planning, environment debugging (NumPy/FAISS/bitsandbytes version conflicts in Colab), report drafting, and presentation iteration. All experimental code, training runs, and numerical results were produced by team members. No AI-generated numerical results appear in any submission. The example outputs shown in the report figures are representative of behavior we observed in our notebooks.

Open-source components used: HuggingFace Transformers, PEFT, TRL, sentence-transformers, FAISS (Meta AI), bitsandbytes, and the Mistral-7B-Instruct-v0.2 model from Mistral AI. All are cited in the final report's References section.

---

## 7. Contact

For questions about the code, please contact any member of the team via the email addresses on the course roster.
