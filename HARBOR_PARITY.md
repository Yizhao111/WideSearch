# Harbor Parity Experiment - Claude-Code Integration

## What was changed

A single new file was added: `scripts/run_claude_code.py`.

This script replaces the original custom agent framework (`src/agent/`) with the **claude-code CLI** for the inference stage, while keeping the entire evaluation pipeline (`src/evaluation/`) untouched. It is the "original side" of a parity experiment against the Harbor adapter (`harbor-framework/harbor`, `adapters/widesearch/`).

No existing files were modified.

## How it works

```
WideSearch task (query)
  -> claude-code CLI (claude -p "query" --print --model haiku --dangerously-skip-permissions)
  -> claude-code uses Bash/curl to search the web and outputs markdown table (stdout)
  -> WideSearch evaluation pipeline scores against gold answer
  -> Metrics: Item F1, Row F1, Success Rate
```

Key implementation details:
- `--dangerously-skip-permissions` is required so claude-code can freely use Bash tools (curl, etc.) to search the web without interactive permission prompts.
- The script monkey-patches `openai_complete` to use the standard `OpenAI` client instead of `AzureOpenAI`, so evaluation works with a regular `OPENAI_API_KEY`.

## Environment variables

```bash
# Required for claude-code agent
export ANTHROPIC_API_KEY="sk-ant-..."

# Required for the LLM judge used during evaluation
export OPENAI_API_KEY="sk-..."

# Optional: override the default eval LLM endpoint (defaults to https://api.openai.com/v1)
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Required: add project root to Python path
export PYTHONPATH="/path/to/WideSearch:$PYTHONPATH"
```

## Running the experiment

### Setup

```bash
cd /path/to/WideSearch
python3 -m venv .venv
# Install deps (dataclasses>=0.8 in pyproject.toml is incompatible with Python 3.7+, install manually)
.venv/bin/pip install aiohttp bs4 dateparser loguru openai pandarallel pandas \
  "volcengine-python-sdk[ark]" scikit-learn scipy "datasets>=4.0.0" "tenacity>=9.1.2" \
  huggingface_hub numpy

export PYTHONPATH="$(pwd):$PYTHONPATH"
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
```

### Sanity check (1-2 tasks)

```bash
.venv/bin/python scripts/run_claude_code.py \
  --instance_id ws_en_001,ws_en_002 \
  --trial_num 1
```

### Small scale (~10 tasks)

```bash
.venv/bin/python scripts/run_claude_code.py \
  --instance_id ws_en_001,ws_en_002,ws_en_003,ws_en_004,ws_en_005,ws_en_006,ws_en_007,ws_en_008,ws_en_009,ws_en_010 \
  --trial_num 1 \
  --thread_num 4
```

### Full single round (all 200 tasks)

```bash
.venv/bin/python scripts/run_claude_code.py \
  --trial_num 1 \
  --thread_num 8
```

### Multi-round (3 rounds for mean + std)

```bash
.venv/bin/python scripts/run_claude_code.py \
  --trial_num 3 \
  --thread_num 8
```

### Infer only (then eval separately)

```bash
# Infer
.venv/bin/python scripts/run_claude_code.py --stage infer --instance_id ws_en_001

# Eval
.venv/bin/python scripts/run_claude_code.py --stage eval --instance_id ws_en_001
```

## CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `haiku` | Model name or alias passed to the claude-code CLI (e.g. `haiku`, `sonnet`, `claude-haiku-4-5-20251001`) |
| `--model_config_name` | `claude-code` | Label used in output file names |
| `--stage` | `both` | `infer`, `eval`, or `both` |
| `--instance_id` | (empty = all) | Comma-separated instance IDs |
| `--trial_num` | `1` | Number of trials per task |
| `--eval_model_config_name` | `default_eval_config` | Eval LLM config from `src/utils/config.py` |
| `--response_root` | `data/output` | Directory for response JSONL files |
| `--result_save_root` | `data/output` | Directory for evaluation result files |
| `--use_cache` | off | Skip tasks whose output files already exist |
| `--thread_num` | `4` | Concurrent threads for infer/eval |
| `--timeout` | `1800` | Timeout (seconds) for each claude-code call |

## Output files

All outputs go to `data/output/` by default:

| File pattern | Content |
|---|---|
| `claude-code_{instance_id}_{trial}_response.jsonl` | Raw claude-code response in WideSearch format |
| `claude-code_{instance_id}_{trial}_eval_result.csv` | Detailed evaluation (per-row matching) |
| `claude-code_{instance_id}_{trial}_eval_result.json` | Evaluation summary for this instance |
| `claude-code_trial_num_{n}_summary.json` | Aggregated metrics across all instances |

## Parity experiment stages

1. **Sanity check** - Run 1-2 tasks, verify the end-to-end flow works.
2. **Small scale** - Run ~10 tasks, inspect trajectories and result overlap with the Harbor side.
3. **Full single round** - Run all 200 tasks once on both sides, compare metrics.
4. **Multi-round** - Run 3 rounds for mean + std, final parity comparison.

## Model

Both sides of the parity experiment use `haiku` (resolves to `claude-haiku-4-5-20251001`) for cost-effectiveness.

## Known limitations

- Claude-code relies on Bash tools (curl, etc.) for web access. Some websites block automated scraping, causing the agent to fail on tasks that require data from those sites.
- The upstream `openai_complete()` in `src/utils/llm.py` uses `AzureOpenAI`. The script monkey-patches it to use the standard `OpenAI` client. If you need Azure, unset `OPENAI_BASE_URL` and configure `default_eval_config` in `src/utils/config.py` directly.
