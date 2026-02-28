# Tutorial: Setting up sb-cli for SWE-bench Lite with uv

This tutorial sets up [sb-cli](https://www.swebench.com/sb-cli/) so you can **evaluate** or **submit** predictions from our SWE-bench Lite runs (e.g. [Orla](orla_swebench_lite.md) or [mini-SWE-agent](mini_swe_agent_swebench_lite.md)). sb-cli talks to the SWE-bench API to run evaluations in the cloud and retrieve reports.

## What you need

- **[uv](https://docs.astral.sh/uv/)** installed.
- (Optional) A [SWE-bench API key](https://www.swebench.com/sb-cli/authentication/) if you want to submit runs or use cloud evaluation.

## 1. Use the sb-cli project

We provide a small project under the Orla repo that pins sb-cli and its dependencies so you avoid environment issues when running the CLI.

From the Orla repo:

```bash
cd examples/swe_bench_lite/sb_cli
uv sync
```

Run sb-cli via the project environment:

```bash
uv run sb-cli --help
```

You should see commands such as `submit`, `get-report`, `list-runs`, and `quota`. Use `uv run sb-cli <command> ...` for all commands in this tutorial when using the project.

## 3. Prediction files from our runs

| Run | Predictions file / format |
|-----|---------------------------|
| **mini-SWE-agent** | `mini_swe_agent_preds/preds.json` (JSON dict: `instance_id` → `{ model_patch, model_name_or_path }`) |
| **Orla** | JSONL written when you pass `-output <path>` to the example (one JSON object per line with `instance_id`, `model_patch`, `model_name_or_path`) |

sb-cli expects a **predictions file** path. For SWE-bench Lite the **subset** is `swe-bench_lite` and **splits** are `dev` or `test`. See [sb-cli User Guide](https://www.swebench.com/sb-cli/user-guide/) and [Submit](https://www.swebench.com/sb-cli/user-guide/submit/) for the exact JSON/JSONL format and options.

## 4. Submit predictions (example)

After setting `SWEBENCH_API_KEY`, you can submit mini-SWE-agent predictions for the **dev** split. From `examples/swe_bench_lite/sb_cli` (or use `uv run sb-cli` from that directory):

```bash
uv run sb-cli submit swe-bench_lite dev \
  --predictions_path /path/to/mini_swe_agent_preds/preds.json \
  --run_id my_run_$(date +%Y%m%d_%H%M%S)
```

Use `--predictions_path` for your Orla JSONL file if you ran the Orla example with `-output preds.jsonl`. The CLI will upload, run evaluation, and you can fetch the report with `uv run sb-cli get-report swe-bench_lite dev <run_id>`.

## 5. Get a report and list runs

```bash
uv run sb-cli get-report swe-bench_lite dev <run_id>
uv run sb-cli list-runs swe-bench_lite dev
```

For more options (e.g. test split, quotas, local evaluation), see the official [SWE-bench evaluation guide](https://www.swebench.com/SWE-bench/guides/evaluation/) and [sb-cli documentation](https://www.swebench.com/sb-cli/).

## Conclusion

You can run sb-cli from the **sb_cli project** (`examples/swe_bench_lite/sb_cli`) with `uv run sb-cli` to submit and evaluate SWE-bench Lite predictions from [Orla](orla_swebench_lite.md) or [mini-SWE-agent](mini_swe_agent_swebench_lite.md) runs. The project pins sb-cli and `typing_extensions` so the CLI runs reliably.
