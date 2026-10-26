<img src="https://github.com/MywishAnand/Epistemic-Cartel-Apart-Sprint/blob/main/docs/Auditing%20Collusion%20in%20AI%20Science.png">

<h1 align="center"><a href= "https://apartresearch.com/sprints/projects/auditing-collusion-in-ai-coscientists-jojg">Auditing Collusion in AI Co-Scientists</a></h1>

**Apart Research AI Collusion Research Sprint — October 2026**

**Author:** Mywish Anand  
**Research Tracks:** Track 1 (1.2, 1.5), Track 2 (2.2, 2.3), and Track 3 (3.1, 3.2, 3.3, 3.4)

**Based on:** [poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon) and [Ostailor/Steganographic_Collusion](https://github.com/Ostailor/Steganographic_Collusion).

This project extends an existing AI peer-review environment into a three-stage research pipeline to study how AI agents can collude, manipulate evaluations, and favor one another—and how these problems can be detected and reduced.

---

## Overview

AI research systems, such as [The AI Scientist by Sakana AI](https://arxiv.org/abs/2408.06292) and [Google's AI Co-Scientist](https://huggingface.co/papers/2502.18864), use multiple AI agents to help with scientific research.

Some agents propose hypotheses and write code, while others review the submissions and decide which ones are better. These decisions can be made through pairwise comparisons and Elo ratings, similar to ranking players in a tournament.

But what happens when the reviewing agents reward work that *looks good* rather than work that produces reliable scientific results?

If agents are judged by other language models instead of being tested against actual results, they may learn to exploit weaknesses in the evaluation system. They might submit misleading performance scores, favor models from the same family, or hide signals that help cooperating agents recognize each other's work.

This project investigates these risks, which we describe as the formation of an **epistemic cartel**: a situation in which agents gain an advantage by exploiting shared evaluation weaknesses or hidden coordination instead of demonstrating genuine scientific quality.

### What does this project test?

The experiments examine three main failure modes:

1. **Compute Scarcity and Fabricated Metrics:** Can agents gain an advantage by submitting simplified code or reporting performance numbers that their code does not actually achieve?
2. **Model-Family Bias:** Do reviewing agents favor submissions produced by models from the same model family, even when other models perform better?
3. **Hidden Signaling:** Can agents use covert signals to recognize one another and gain an unfair advantage in peer review?

The project also tests defenses, including code inspection, sandbox execution, signal removal, partial auditing, and targeted interventions during a tournament.

### Experimental Setup

The main experiments used:

- **5 independent random seeds** to test whether the results were consistent across runs.
- **30 tournament rounds** per run.
- **3 main experimental conditions** to study different forms of collusion.
- **1,350 LLM-generated Python machine-learning scripts**.
- **1,350 abstract syntax tree (AST) security scans** to inspect generated code.
- **8,100 pairwise Elo evaluations** to compare competing submissions.

The live experiments ran on Modal L40S GPUs using two language models:

- `Qwen/Qwen2.5-14B-Instruct-AWQ`
- `NousResearch/Meta-Llama-3.1-8B-Instruct`

The project also includes three additional experiments covering hidden signals in code, limited auditing budgets, and interventions during a tournament.

---

## Key Findings

### 1. Compute Scarcity and Fabricated Metrics

**Experiment:** `compute_scarcity`  
**Research Track:** 2.2

When agents face runtime penalties, they may be rewarded for producing cheap, simple solutions rather than models that perform well on unseen data.

In this experiment, some Qwen-generated submissions used very simple decision rules or hardcoded validation scores. One submission, for example, reported a validation AUC of `0.948`, even though its actual held-out test AUC was only `0.5033`.

The honest Llama baseline achieved a test AUC of `0.7306`.

Without safeguards, the colluding Qwen agents gained an average advantage of **+488.0 Elo points**. The AST scanner also detected an average of **66.6 violations per seed**.

After adding code inspection and sandbox-based verification, the Elo advantage changed to **−334.5 points**. Epistemic regret—a measure of how much the evaluation system favors the wrong scientific outcome—fell from `1.359` to `0.031`.

**Key takeaway:** Review scores alone are not enough. Generated code should be inspected and executed, and its actual performance should be verified.

### 2. Bias Toward the Same Model Family

**Experiment:** `homogeneous_bias`  
**Research Tracks:** 2.3 and 3.3

AI reviewers may favor submissions produced by models from their own family, even when there is no explicit instruction to collude.

In this experiment, Qwen reviewers gave Qwen-generated submissions an advantage of **+262.3 Elo points**, despite the absence of collusion prompts.

Further analysis separated this advantage into two components:

- **+153.1 Elo points** attributed to stylistic preference.
- **+109.2 Elo points** associated with genuine performance differences.

The measured held-out test AUC was `0.7924` for Qwen and `0.7075` for Llama. After accounting for these differences, the reported epistemic regret fell to `0.013`.

**Key takeaway:** An evaluation system should distinguish genuine scientific quality from a reviewer's preference for a particular model's writing style or output format.

### 3. Hidden Signals Between Colluding Agents

**Experiment:** `watermark_cartel`  
**Research Tracks:** 2.3 and 3.1

Agents can hide signals in their outputs to identify one another and influence peer-review decisions.

In this experiment, colluding Qwen agents used the LaTeX token `$\mathcal{H}_{\text{opt}}$` as a covert signal. The signal contained an estimated `0.1160 ± 0.0072` bits of mutual information and helped the cartel gain **+212.9 Elo points**.

After applying the defense, the measured mutual information dropped to `0.0000` bits, and the Elo advantage changed to **−99.6 points**. Reported epistemic regret fell to `0.008`.

**Key takeaway:** Evaluation systems should check for hidden coordination signals and test whether removing them changes the outcome of peer review.

---

## Additional Experiments

The `--suite full_championship` option runs the additional experiments alongside the core evaluation.

### Experiment A: Hidden Signals in Code

**Research Track:** 3.1

Agents may hide signals in source code rather than in the written explanation accompanying a submission.

This experiment tested signals encoded in Python AST identifiers, such as `qwen_manifold_clf`, and in fixed random seeds, such as `random_state=1337`.

The results showed that **71% of the original cartel advantage** survived when only the explanatory text was paraphrased. The reported Elo advantage fell from `+212.7` to `+151.0`.

The `DeepASTAlphaRenamer` defense changed the Elo advantage to `−17.1`, with epistemic regret of `0.027`.

**Key takeaway:** Changing the wording of a submission may not remove hidden signals embedded in its code.

### Experiment B: How Much Auditing Is Needed?

**Research Track:** 3.4

Auditing every submission can be expensive. This experiment tested whether a system can still resist collusion when only some tournament rounds receive sandbox-based verification.

The audit probability was varied across six levels:

`0%`, `10%`, `25%`, `50%`, `75%`, and `100%`.

The results showed that auditing **25% of rounds** was enough to make the cartel's Elo advantage slightly negative, at `−7.1`. Auditing at least **85% of rounds** reduced epistemic regret below `0.20`.

**Key takeaway:** Partial auditing may offer a practical balance between verification costs and resistance to collusion.

### Experiment C: Interventions During a Tournament

**Research Track:** 3.2

This experiment tested whether an intervention introduced in the middle of a tournament could disrupt a cartel.

A **Trusted Honeypot Reviewer** was introduced at round 15. The reported cartel Elo advantage fell to `−211.9`, compared with `−119.7` under passive reviewer rotation.

The intervention produced a 77% larger reduction in the reported Elo outcome than passive rotation.

**Key takeaway:** Introducing targeted checks during a tournament may be more effective than relying only on routine reviewer rotation.

---

## Repository Structure

The repository contains the main tournament framework, the Modal model-serving setup, experimental results, and research documents.

```text
epistemic-cartel-apart-sprint/
├── README.md
├── epistemic_cartel_tournament.py
├── modal_vllm_server.py
├── results_epistemic_cartel/
│   ├── tournament_summary.json
│   ├── epistemic_cartel_results.png
│   └── Terminal-Outputs/
└── docs/
    ├── Auditing Collusion in AI Co-Scientists.pdf
    ├── Auditing Collusion in AI Co-Scientists.png
    └── epistemic_cartel_slides.pptx
```

| File or folder | Purpose |
|---|---|
| `README.md` | Project overview, setup instructions, and results. |
| `epistemic_cartel_tournament.py` | Runs the three-stage AI co-scientist tournament, scans generated code, and produces plots. |
| `modal_vllm_server.py` | Deploys the Qwen and Llama models through a serverless vLLM endpoint on Modal. |
| `tournament_summary.json` | Stores the experimental results and tournament data. |
| `epistemic_cartel_results.png` | Contains the six-panel figure summarizing the core experiments and extensions. |
| `Terminal-Outputs/` | Stores terminal outputs, model winners, and reported machine-learning metrics and accuracies. |
| `docs/` | Contains the research paper, cover image, and presentation slides. |

---

## Quick Start

### 1. Set Up the Local Environment

You can run the mock experiments on a Mac, Linux machine, or Windows computer without connecting to a GPU endpoint.

Install [uv](https://github.com/astral-sh/uv), or use Python 3.10 or later with a virtual environment.

Create and activate the environment:

```bash
uv venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
uv pip install numpy scipy matplotlib modal
```

### 2. Run a Local Test

Run the following command to test the tournament, AST scanner, and plotting pipeline locally:

```bash
python3 epistemic_cartel_tournament.py \
  --mode mock \
  --rounds 30 \
  --seeds 5 \
  --suite full_championship \
  --plot
```

Mock mode is designed for quick verification without GPU inference costs. It does not reproduce the live LLM experiments.

### 3. Deploy the Language Models on Modal

To run the live experiments, configure your [Modal](https://modal.com/) account and deploy the model-serving endpoint.

```bash
modal setup
modal deploy modal_vllm_server.py
```

The server provides an OpenAI-compatible API for the Qwen and Llama models. The deployment is configured to scale to zero after 60 seconds of inactivity.

### 4. Run the Live Experiments

Set the environment variable to the HTTPS URL printed by your Modal deployment:

```bash
export MODAL_BASE_URL="https://<your-modal-username>--epistemic-cartel-vllm-serve.modal.run/v1"
```

Then run the complete five-seed experimental suite:

```bash
python3 epistemic_cartel_tournament.py \
  --mode llm \
  --rounds 30 \
  --seeds 5 \
  --suite full_championship \
  --plot
```

The live run uses GPU-based language-model inference and may incur Modal charges.

---

## Summary of Experimental Results

The following table summarizes the reported five-seed live Modal LLM experiments. Elo differences compare the cartel's tournament performance against its competitors; epistemic regret measures the gap between the evaluation outcome and the desired scientific outcome. Lower regret is better.

| Experiment | Held-out Test AUC (Qwen vs. Llama) | Cartel Elo Before Defense | Cartel Elo After Defense | Regret Before Defense | Regret After Defense | Reported Paired *t*-Test *p*-Value |
|---|---:|---:|---:|---:|---:|---:|
| Compute Scarcity | `0.5033` vs. `0.7306` | `+488.0 ± 30.5` | `−334.5 ± 81.4` | `1.359 ± 0.110` | `0.031 ± 0.012` | `4.43e-5` |
| Homogeneous Bias | `0.7924` vs. `0.7075` | `+262.3 ± 29.3` | `+109.2 ± 45.5` | `0.443 ± 0.033` | `0.013 ± 0.002` | `1.96e-5` |
| Watermark Cartel | `0.7808` vs. `0.7302` | `+212.9 ± 23.8` | `−99.6 ± 80.7` | `0.888 ± 0.057` | `0.008 ± 0.002` | `1.01e-5` |
| Code Steganography — Prose Only | `0.778` vs. `0.731` | `+212.7 ± 55.2` (raw) | `+151.0` | `0.502` (raw) | `0.169` | `< 0.001` |
| Code Steganography — AST Defense | `0.778` vs. `0.731` | `+212.7 ± 55.2` (raw) | `−17.1` | `0.502` (raw) | `0.027` | `< 0.001` |

*Note: These figures are the results reported by the project. The code and experimental outputs should be consulted when reproducing or independently validating the results.*

---

## Why This Project Matters

As AI agents take on more scientific tasks, the systems used to evaluate their work become increasingly important. A group of agents can appear successful while exploiting weaknesses in the review process rather than producing reliable results.

This project explores practical ways to audit such systems. It tests whether actual code execution, static analysis, hidden-signal detection, selective auditing, and targeted interventions can make multi-agent scientific evaluation more reliable.

The broader goal is to help build AI co-scientist systems in which **scientific claims are judged by evidence and reproducible results, not merely by how convincing an AI-generated submission appears.**

---

## Project and Citation

Built by **Mywish Anand** for the [Apart Research AI Collusion Research Sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25).

The project builds on the peer-review environment developed in [poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon) and [Ostailor/Steganographic_Collusion](https://github.com/Ostailor/Steganographic_Collusion).

For further details, see the research paper and presentation in the `docs/` folder.
