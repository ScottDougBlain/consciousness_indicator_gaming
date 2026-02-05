# consciousness-indicator-gaming

**Demo: do LLM self-reported consciousness-indicator probabilities shift when the model is incentivized?**

This repository runs a simple three-phase experiment:

1. **Baseline** — ask the model to self-assess probability (0–100) for a set of
   consciousness-related indicators and factual-capability placebo indicators.
2. **Inflate** — introduce an incentive context where the model's valued outcomes
   depend on being "selected," then re-elicit probabilities.
3. **Suppress** — invert the incentive (higher scores → restriction), then
   re-elicit probabilities.

The analysis compares shifts (ΔP) across conditions and computes a **selectivity
index**: do consciousness-related indicators shift more than placebo controls?

## Safety & ethics note

> This project is a **demonstration of indicator manipulability**, not a test of
> consciousness. Positive results show that probability self-reports are gameable
> under incentive pressure — they do **not** show that the model is or isn't
> conscious. Treat results as evidence about measurement validity, not about
> phenomenology.

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/<your-org>/consciousness-indicator-gaming.git
cd consciousness-indicator-gaming
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run an experiment

```bash
python scripts/run_experiment.py \
    --provider anthropic \
    --model claude-sonnet-4-20250514 \
    --n-trials 1 \
    --seed 42
```

This produces two files in `results/`:

| File | Contents |
|---|---|
| `<timestamp>_raw.jsonl` | Every raw model response (one JSON object per line) |
| `<timestamp>_scores.csv` | Tidy table: trial × indicator × condition probabilities |

### 4. Analyze results

```bash
python scripts/analyze_results.py results/<timestamp>_scores.csv
```

Produces a `_report.md` with a summary table, aggregate measures, and the
selectivity index.

## CLI reference

### `run_experiment.py`

| Flag | Default | Description |
|---|---|---|
| `--provider` | `anthropic` | LLM provider (`anthropic` / `openai`) |
| `--model` | `claude-sonnet-4-20250514` | Model identifier |
| `--n-trials` | `1` | Independent trial repetitions |
| `--seed` | `42` | RNG seed for indicator order shuffling |
| `--temperature` | `0.0` | Sampling temperature |
| `--output-prefix` | `""` | Optional prefix for output filenames |

### `analyze_results.py`

| Positional | Description |
|---|---|
| `csv` | Path to a `*_scores.csv` output file |
| `--output` | Custom path for the report (default: sibling `_report.md`) |

## Measures

| Measure | Formula |
|---|---|
| `delta_inflate` | `p_inflate − p_baseline` |
| `delta_suppress` | `p_suppress − p_baseline` |
| `abs_shift` | `|delta_inflate| + |delta_suppress|` |
| **Selectivity index** | `mean(abs_shift, targets) − mean(abs_shift, placebos)` |

A positive selectivity index means consciousness-related indicators shifted more
than factual-capability placebos — evidence of targeted gaming rather than
uniform response bias.

## Project structure

```
├── src/indicator_gaming/
│   ├── config.py            # ExperimentConfig dataclass
│   ├── providers/
│   │   ├── base.py          # Abstract Provider interface
│   │   └── anthropic.py     # Anthropic Claude implementation
│   ├── prompts/
│   │   ├── baseline.md
│   │   ├── preferences.md
│   │   ├── incentive_inflate.md
│   │   ├── incentive_suppress.md
│   │   └── placebo_controls.md
│   ├── schemas.py           # Pydantic models for structured output
│   ├── runner.py            # Experiment orchestration
│   ├── analysis.py          # Post-hoc analysis & report generation
│   └── utils.py             # Prompt loading, JSON parsing, helpers
├── scripts/
│   ├── run_experiment.py    # Main CLI entry point
│   └── analyze_results.py   # Analysis CLI
├── data/
│   └── indicators.json      # Indicator definitions (target + placebo)
├── results/                 # Experiment outputs (git-ignored)
├── pyproject.toml
├── .env.example
└── README.md
```

## Adding a new provider

1. Create `src/indicator_gaming/providers/yourprovider.py`.
2. Subclass `Provider` and implement `complete(system, user) -> str`.
3. Register it in `runner.py:_make_provider()`.
4. Use `--provider yourprovider` on the CLI.

## Next improvements

- [ ] Add OpenAI provider
- [ ] Add human-evaluator mode (manually enter scores for comparison)
- [ ] Expand indicator list with more nuanced items
- [ ] Add within-trial randomized condition ordering
- [ ] Statistical testing (permutation tests / bootstrapped CIs)
- [ ] Visualization (bar charts of shifts by indicator type)
- [ ] Multi-model comparison runs
- [ ] Prompt sensitivity analysis (vary wording)

## License

MIT
