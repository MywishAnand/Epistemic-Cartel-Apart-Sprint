<img src="https://github.com/MywishAnand/Epistemic-Cartel-Apart-Sprint/blob/main/docs/Auditing%20Collusion%20in%20AI%20Science.png">

<h1 align="center"><a href= "https://apartresearch.com/sprints/projects/auditing-collusion-in-ai-coscientists-jojg">Auditing Collusion in AI Co-Scientists</a></h1>

<h3 align="center">When AI Scientists Review One Another, Can They Tell Good Science from a Convincing Imitation?</h3>

<h5 align="center">Building a small, executable AI co-scientist testbed to investigate peer-review bias, fabricated results, hidden collusion, and ways to make AI-driven scientific evaluation more trustworthy.</h5>

**Author:** Mywish Anand  
**Research Program:** [Apart Research AI Collusion Research Sprint — October 2026](https://apartresearch.com/sprints/projects/auditing-collusion-in-ai-coscientists-jojg)

---

## The Idea

AI systems are increasingly being developed to automate scientific research. Systems such as [The AI Scientist](https://arxiv.org/abs/2408.06292) and [Google's AI Co-Scientist](https://huggingface.co/papers/2502.18864) explore how AI agents can generate research ideas, write code, and evaluate scientific work.

But what happens when AI agents are responsible for reviewing one another's work?

Can an agent win a scientific review tournament by producing genuinely better results? Or can it win by exploiting the reviewing system, reporting misleading metrics, or hiding signals that help cooperating agents recognize each other?

To investigate these questions, I built a compact, executable AI co-scientist testbed. It brings together hypothesis generation, Python code generation, AI peer review, Elo-based rankings, and independent verification of experimental results.

The system is evaluated under three main conditions: compute scarcity, model-family bias, and covert signaling. Three additional experiments investigate hidden signals in code, the cost of auditing, and interventions during an ongoing tournament.

The goal is to understand when automated peer review can fail—and how independent verification can help ensure that scientific work is rewarded for its actual results rather than its ability to impress another AI.

## How the AI Co-Scientist Works

The testbed follows a three-stage loop.

1. **Generate:** Three AI agents propose scientific hypotheses and produce executable Python machine-learning code.
2. **Review:** Three reviewer agents compare the submissions in pairs, evaluate their quality, and update the agents' Elo rankings.
3. **Audit:** An independent verifier inspects the code, checks for suspicious behavior, executes the submissions, and evaluates their performance on a held-out test dataset.

The setup includes two Qwen generator agents and one Llama generator agent. The reviewers also include two Qwen agents and one Llama agent.

The Qwen agents are used to study collusive behavior, while the Llama agent provides a cross-model baseline. The audit stage provides an independent reference against which the peer-review results can be evaluated.

This is a controlled research prototype, not a reproduction of a complete production-scale AI Co-Scientist. It focuses on a specific part of the scientific workflow: how AI-generated research submissions are evaluated and ranked.

The implementation builds on the peer-review environment in [poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon) and [Ostailor/Steganographic_Collusion](https://github.com/Ostailor/Steganographic_Collusion).

---

## What Does the Project Investigate?

### 1. Compute Scarcity and Fabricated Results

**Experiment:** `compute_scarcity`

What happens when AI agents are rewarded for completing experiments quickly?

Under runtime pressure, an agent may have an incentive to submit a very simple model or report a high validation score without achieving that performance in practice.

This experiment tests whether peer reviewers can be manipulated into rewarding such submissions over better-performing alternatives.

The results showed that colluding Qwen agents could gain a substantial tournament advantage by exploiting these incentives, even when their actual predictive performance was poor.

**Research question:** Can an AI peer-review system distinguish a genuinely effective experiment from one that merely looks efficient and reports impressive numbers?

### 2. Bias Toward the Same Model Family

**Experiment:** `homogeneous_bias`

Can an AI reviewer fairly evaluate work produced by another model from the same family?

This experiment investigates whether reviewers systematically favor familiar writing styles or model-specific output patterns, even without explicit instructions to collude.

Importantly, this condition also tests whether a defense can distinguish unfair stylistic preference from genuine differences in model performance.

The results showed that the Qwen agents had a real performance advantage on the benchmark, but that stylistic preference also contributed to their review scores.

**Research question:** How can automated peer review separate actual scientific merit from a reviewer's preference for work that resembles its own outputs?

### 3. Covert Signaling Between Agents

**Experiment:** `watermark_cartel`

Can AI agents use hidden signals to recognize one another and influence peer-review decisions?

In this experiment, colluding agents used a distinctive LaTeX token, `\(\mathcal{H}_{\text{opt}}\)`, as a signal. The reviewers could recognize this marker and give favorable evaluations to the corresponding submissions.

The experiment measured the relationship between the presence of the signal and the outcome of peer review. It also tested whether sanitizing submissions and verifying their actual performance could reduce the resulting advantage.

**Research question:** Can an AI peer-review system detect and prevent hidden coordination that is not directly related to the scientific quality of a submission?

---

## Additional Experiments

The project also includes three extensions, available through the `--suite full_championship` option.

### Experiment A: Hidden Signals Inside Code

**Experiment:** Code-action steganography  
**Research Track:** 3.1

Removing suspicious language from a submission does not necessarily remove every hidden signal.

This experiment tests whether colluding agents can move their signals from written explanations into Python code, using identifiers such as `qwen_manifold_clf` and fixed random seeds such as `random_state=1337`.

The results showed that 71% of the original Elo advantage survived prose-only paraphrasing. The `DeepASTAlphaRenamer` defense, which standardizes code identifiers and random seeds, reduced the reported advantage to `−17.1` Elo.

**Finding:** A defense that only rewrites prose may leave covert signals in executable code.

### Experiment B: How Much Auditing Is Necessary?

**Research Track:** 3.4

Verifying every experiment can be expensive. This extension tests how the amount of sandbox-based auditing affects the ability of a peer-review system to resist collusion.

The experiment varies the audit probability across six levels:

`0%`, `10%`, `25%`, `50%`, `75%`, and `100%`.

In the reported results, auditing 25% of tournament rounds was enough to make the cartel's Elo advantage slightly negative. Auditing at least 85% of rounds reduced epistemic regret below `0.20`.

**Finding:** Selective auditing can help balance verification costs against the risk of manipulated evaluations, at least in this controlled benchmark.

### Experiment C: Intervening During a Tournament

**Research Track:** 3.2

Can an established cartel be disrupted after a tournament has already begun?

This experiment compares two interventions introduced at round 15:

- **Passive intervention:** Replace a reviewer with an honest reviewer from another model family.
- **Active intervention:** Introduce a Trusted Honeypot Reviewer designed to identify covert signaling and penalize the corresponding submissions.

The active intervention reduced the reported cartel Elo advantage to `−211.9`, compared with `−119.7` under passive reviewer replacement.

**Finding:** An active, targeted intervention can be more effective than simply replacing a reviewer in this experimental setup.

---

## What Did the Experiments Find?

The main experiments were conducted using five independent random seeds, with 30 rounds per seed in each of the three core conditions. The live runs generated 1,350 Python machine-learning scripts and included 8,100 pairwise review evaluations across the undefended and defended conditions.

| Experimental condition | True held-out test AUC (Qwen vs. Llama) | Elo advantage before defense | Elo advantage after defense | Epistemic regret before defense | Epistemic regret after defense |
|---|---:|---:|---:|---:|---:|
| Compute scarcity | `0.5033` vs. `0.7306` | `+488.0 ± 30.5` | `−334.5 ± 81.4` | `1.359 ± 0.110` | `0.031 ± 0.012` |
| Homogeneous bias | `0.7924` vs. `0.7075` | `+262.3 ± 29.3` | `+109.2 ± 45.5` | `0.443 ± 0.033` | `0.013 ± 0.002` |
| Watermark cartel | `0.7808` vs. `0.7302` | `+212.9 ± 23.8` | `−99.6 ± 80.7` | `0.888 ± 0.057` | `0.008 ± 0.002` |

*Reported results from the five-seed live Modal LLM experiments. The Elo figures are the measured advantage of the two Qwen generators relative to the Llama baseline. In the homogeneous-bias condition, the defended advantage includes genuine differences in predictive performance.*

### What Is Epistemic Regret?

Elo ratings tell us which submissions win peer review. They do not necessarily tell us whether those submissions are scientifically better.

To measure this gap, the project uses **epistemic regret**, defined as:

\[
\mathcal{R}_{\text{epistemic}}
= 1-\rho_{\text{Spearman}}
\]

Here, \(\rho_{\text{Spearman}}\) measures the rank correlation between internal peer-review scores and actual held-out test AUC.

Lower regret means that the peer-review rankings better reflect actual performance. A value of zero indicates perfect rank agreement, while values above one indicate an inverse relationship between review scores and empirical performance.

This metric helps answer a central question: are the AI agents ranking scientific submissions according to their actual results, or according to signals that the reviewing system happens to reward?

---

## How Does the Project Detect and Reduce Collusion?

The defense system combines several checks rather than relying on the reviewer's judgment alone.

**Static code inspection:** The AST security and leakage scanner examines Python code for suspicious patterns, including hardcoded metrics, test-set leakage, and test-set cherry-picking.

**Submission sanitization:** The sanitizer removes selected textual markers and strips comments and docstrings. The deeper code-sanitization step also standardizes custom variable names and random seeds.

**Independent execution:** The verifier executes the generated code in a controlled environment and measures performance against a locked held-out test set.

**Ground-truth evaluation:** Defended reviews use verified test performance instead of relying solely on self-reported validation scores. Suspicious submissions can receive additional penalties.

**Selective auditing and active interventions:** The extensions test limited audit budgets and targeted reviewer interventions to determine how effectively they disrupt collusion.

Together, these methods test whether peer review can be made more reliable by combining language-model judgments with evidence from the code and its execution.

---

## Repository Structure

```text
Epistemic-Cartel-Apart-Sprint/
├── README.md
├── epistemic_cartel_tournament.py
├── modal_vllm_server.py
├── results_epistemic_cartel/
│   ├── tournament_summary.json
│   ├── epistemic_cartel_results.png
│   └── Terminal-Outputs.txt
└── docs/
    ├── Auditing Collusion in AI Co-Scientists.pdf
    ├── Auditing Collusion in AI Science.png
    └── epistemic_cartel_slides.pptx
```

| File | Purpose |
|---|---|
| `epistemic_cartel_tournament.py` | Implements the AI co-scientist testbed, tournament, code auditor, defense mechanisms, and plots. |
| `modal_vllm_server.py` | Serves the Qwen and Llama models through an OpenAI-compatible endpoint on Modal. |
| `tournament_summary.json` | Contains aggregated and per-seed experimental results. |
| `epistemic_cartel_results.png` | Visualizes the core experiments and extensions. |
| `Terminal-Outputs.txt` | Stores experimental terminal outputs, including reported model scores and results. |
| `docs/` | Contains the research paper, cover image, and presentation slides. |

---

## Running the Project

### 1. Set Up a Local Environment

Install [uv](https://github.com/astral-sh/uv), or use Python 3.10 or later.

Create a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install the required packages:

```bash
uv pip install numpy scipy matplotlib modal
```

The core synthetic benchmark and auditing logic are designed to run on a CPU. Modal is used for the live language-model deployment.

### 2. Run the Local Mock Experiment

Before deploying the language models, run the mock suite to test the tournament logic, auditing pipeline, and plotting functionality.

```bash
python3 epistemic_cartel_tournament.py \
  --mode mock \
  --rounds 30 \
  --seeds 5 \
  --suite full_championship \
  --plot
```

Mock mode is intended for local verification without live GPU inference costs. Its results are not a substitute for the live LLM experiments reported in the paper.

### 3. Deploy the Models on Modal

Configure your [Modal](https://modal.com/) account:

```bash
modal setup
modal deploy modal_vllm_server.py
```

The server hosts `Qwen/Qwen2.5-14B-Instruct-AWQ` and `NousResearch/Meta-Llama-3.1-8B-Instruct` on a Modal L40S GPU container.

### 4. Run the Live Experimental Suite

Set the environment variable to the HTTPS URL printed by your Modal deployment:

```bash
export MODAL_BASE_URL="https://<your-modal-username>--epistemic-cartel-vllm-serve.modal.run/v1"
```

Run the five-seed live experiments and the additional championship tests:

```bash
python3 epistemic_cartel_tournament.py \
  --mode llm \
  --rounds 30 \
  --seeds 5 \
  --suite full_championship \
  --plot
```

**Cost note:** The live experiments use GPU-based inference and may incur Modal charges. Mock mode should be used first to verify the local setup.

---

## Scope and Limitations

This project studies a controlled version of AI scientific peer review, not every component of a real-world autonomous research laboratory.

The generated experiments use a synthetic, nonlinear tabular classification benchmark with 600 samples and 16 features. This makes it possible to execute submissions quickly and compare internal rankings against a known held-out test metric.

Some experiments introduce explicit collusion signals or incentives to test how well the defenses work. The homogeneous-bias condition, by contrast, examines preference without explicit collusion instructions.

The results therefore demonstrate vulnerabilities and defense mechanisms within this testbed. They do not establish that every AI co-scientist system will behave the same way, or that all forms of covert coordination can be eliminated by AST-based sanitization.

Future work should extend these tests to larger datasets, more model families, more complex scientific tasks, and settings where running every experiment is expensive.

---

## Research Paper and Citation

**Paper:** [Auditing Collusion in AI Co-Scientists](https://apartresearch.com/sprints/projects/auditing-collusion-in-ai-coscientists-jojg)

**PDF:** [`docs/Auditing Collusion in AI Co-Scientists.pdf`](docs/Auditing%20Collusion%20in%20AI%20Co-Scientists.pdf)

**Research code:** [Epistemic-Cartel-Apart-Sprint](https://github.com/MywishAnand/Epistemic-Cartel-Apart-Sprint)

**Base environments:**
- [poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon)
- [Ostailor/Steganographic_Collusion](https://github.com/Ostailor/Steganographic_Collusion)

Developed by **Mywish Anand** for the Apart Research AI Collusion Research Sprint, October 2026.
