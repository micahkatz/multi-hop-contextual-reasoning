# Multi-Hop Contextual Reasoning Evaluation Framework

This repository contains the evaluation framework for the paper:

**"Scaling Trends for Multi-Hop Contextual Reasoning in Mid-Scale Language Models"**

*Brady Steele (Georgia Institute of Technology) and Micah Katz (The University of Texas at Austin)*

## Overview

We present a synthetic evaluation framework for studying multi-hop reasoning capabilities in language models. The framework generates controlled scenarios requiring cross-document information synthesis and contextual understanding, enabling discriminative evaluation of reasoning vs. pattern-matching capabilities.

### Key Findings

1. **Task-Method Dissociation**: Rule-based pattern matching achieves 100% on structured tasks but only 6.7% on reasoning tasks, while LLM multi-agent systems achieve up to 80% on reasoning tasks.

2. **Multi-Agent Amplification**: Multi-agent coordination provides statistically significant improvements (up to +46.7 percentage points, p < 0.001) only for models with sufficient base reasoning capability.

3. **Active Parameters Predict MoE Reasoning**: Mixtral's performance aligns with its ~12B active parameters rather than 47B total.

## Installation

```bash
# Clone the repository
git clone https://github.com/micahkatz/multi-hop-contextual-reasoning.git
cd multi-hop-contextual-reasoning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally for LLM inference
- Models used in the paper: `llama3:8b`, `llama2:13b`, `mixtral:8x7b`, `deepseek-v2:16b`

## Usage

### Quick Test

Run a quick evaluation with 1 trial per condition:

```bash
python evaluation_framework.py --mode quick --model llama3:8b
```

### Full Experiment

Run the complete experiment with multiple trials:

```bash
python evaluation_framework.py --mode full --model llama3:8b --trials 5
```

### Single Scenario

Test a specific scenario type and difficulty:

```bash
# Reasoning task, difficulty 3 (4-hop)
python evaluation_framework.py --mode single --type reasoning --difficulty 3

# Pattern matching task, difficulty 1
python evaluation_framework.py --mode single --type pattern_match --difficulty 1
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Experiment mode: `single`, `quick`, `full` | `quick` |
| `--model` | Ollama model name | `llama3:8b` |
| `--trials` | Trials per condition (full mode) | `5` |
| `--difficulty` | Difficulty level 1-3 (single mode) | `2` |
| `--type` | Scenario type: `pattern_match`, `reasoning` | `reasoning` |
| `--seed` | Random seed for reproducibility | `42` |

## Framework Architecture

### Scenario Generation

The framework generates two types of tasks:

- **Pattern Match (Structured)**: Target strings derivable through direct extraction
- **Reasoning (Contextual)**: Target strings requiring multi-hop inference across documents

Difficulty levels correspond to reasoning hops required:
- Level 1 (2-hop): Semantic relationship understanding
- Level 2 (3-hop): Cross-reference with explicit context
- Level 3 (4-hop): Cross-document synthesis

### Agent Architectures

1. **Multi-Agent**: Four-node LangGraph pipeline (Analyst → Reasoner → Strategist → Generator)
2. **Single-Agent**: Single-prompt baseline
3. **No-Reasoner**: Ablation without the reasoning node

### Baselines

- `rule_based`: Pattern matching with entity extraction
- `rule_transform`: Enhanced rule-based with transformations
- `dictionary`: Common password dictionary
- `random`: Random character generation

## Results

Publication figures from our experiments are available in the `results/` directory:

```
results/
├── fig_scaling_curves.png      # Multi-agent amplification with significance
├── fig_task_comparison.png     # Task-method dissociation
├── fig_task_types.png          # Conceptual diagram of structured vs. contextual tasks
├── fig_reasoning_hops.png      # Performance vs. reasoning complexity
└── fig_active_vs_total.png     # Active vs. total parameter comparison
```

## Reproducibility

All experiments were conducted on consumer hardware:
- Apple MacBook Pro with 36GB unified memory
- Ollama local inference with 4-bit quantized models
- Total compute time: ~3 hours for complete evaluation

To reproduce the paper results:

```bash
# Run for each model
python evaluation_framework.py --mode full --model llama3:8b --trials 5 --seed 42
python evaluation_framework.py --mode full --model llama2:13b --trials 5 --seed 42
python evaluation_framework.py --mode full --model mixtral:8x7b --trials 5 --seed 42
python evaluation_framework.py --mode full --model deepseek-v2:16b --trials 5 --seed 42
```

## Ethical Considerations

- All experimental data is **entirely synthetic**
- No real user data, passwords, or personal information is used
- Names are randomly generated; dates are sampled from plausible ranges
- This framework is intended for research on LLM reasoning capabilities

## License

MIT License
