# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""
Run WideSearch inference using claude-code CLI and evaluate with the original pipeline.

Usage:
    python scripts/run_claude_code.py --instance_id ws_en_001 --trial_num 1
    python scripts/run_claude_code.py --stage infer --instance_id ws_en_001,ws_en_002
    python scripts/run_claude_code.py --stage eval --instance_id ws_en_001
    python scripts/run_claude_code.py --trial_num 3  # all tasks, 3 rounds
"""

import dataclasses
import json
import os
import subprocess
import sys
import time
import traceback
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
from loguru import logger

from src.evaluation.data_loader import (
    WideSearchDataLoaderHF,
    WideSearchQuery,
    WideSearchResponse,
    WideSearchResponseLoader,
)
from src.evaluation.evaluation import EvaluationResult, evaluate_single_query
from src.utils.config import model_config

logger.remove()
logger.add(sys.stderr, level="INFO")

# ---------------------------------------------------------------------------
# Inject eval LLM credentials from environment variables so we don't need
# to hard-code secrets in config.py.
# ---------------------------------------------------------------------------
_eval_api_key = os.environ.get("OPENAI_API_KEY", "")
_eval_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
if _eval_api_key:
    model_config["default_eval_config"]["api_key"] = _eval_api_key
if _eval_base_url:
    model_config["default_eval_config"]["base_url"] = _eval_base_url

# ---------------------------------------------------------------------------
# The upstream openai_complete() uses AzureOpenAI which is incompatible with
# standard OpenAI API keys.  Monkey-patch it to use the standard OpenAI
# client so evaluation works with a regular OPENAI_API_KEY.
# ---------------------------------------------------------------------------
def _patch_openai_complete():
    from openai import OpenAI
    from tenacity import retry, stop_after_attempt, wait_incrementing
    import src.utils.llm as _llm_mod

    @retry(stop=stop_after_attempt(8), wait=wait_incrementing(8, 8))
    def openai_complete_standard(
        base_url, api_key, messages, tools=None, model_name="gpt-4.1-2025-04-14",
        retry_if_empty=False, **generate_kwargs,
    ):
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=300)
        completion = client.chat.completions.create(
            messages=messages, model=model_name, tools=tools, **generate_kwargs,
        )
        message = None
        try:
            message = completion.choices[0].message
        except Exception as e:
            logger.warning(f"Error during completion: {e}")
            return None
        if retry_if_empty and not message.content and not message.tool_calls:
            raise RuntimeError("[openai_complete] empty response, retry")
        return message

    _llm_mod.openai_complete = openai_complete_standard

_patch_openai_complete()


def run_claude_code(query_text: str, model: str, timeout: int) -> str:
    """Invoke the claude-code CLI and return its stdout."""
    try:
        result = subprocess.run(
            [
                "claude",
                "-p", query_text,
                "--print",
                "--model", model,
                "--dangerously-skip-permissions",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"claude-code timed out after {timeout}s")
        return "NULL"

    if result.returncode != 0:
        logger.error(f"claude-code exited with code {result.returncode}: {result.stderr}")
        return "NULL"

    return result.stdout


class ClaudeCodeTask:
    """A single infer + eval unit driven by the claude-code CLI."""

    def __init__(
        self,
        query: WideSearchQuery,
        response_path: str,
        result_save_path: str,
        trial_idx: int = 0,
        model: str = "haiku",
        timeout: int = 1800,
        use_cache: bool = False,
        eval_model_config_name: str = "default_eval_config",
    ):
        self.query = query
        self.response_path = response_path
        self.result_save_path = result_save_path
        self.trial_idx = trial_idx
        self.model = model
        self.timeout = timeout
        self.use_cache = use_cache
        self.eval_model_config_name = eval_model_config_name
        self.eval_result_path = self.result_save_path.replace(".csv", ".json")

    def load_response(self) -> list[WideSearchResponse]:
        if not os.path.exists(self.response_path):
            raise FileNotFoundError(f"response_path {self.response_path} not found")
        return WideSearchResponseLoader.load_response(self.response_path)

    def infer(self) -> list[WideSearchResponse]:
        if self.use_cache and os.path.exists(self.response_path):
            logger.info(f"response_path {self.response_path} exists, skip")
            return self.load_response()

        logger.info(f"infer start, instance_id: {self.query.instance_id}")
        start_time = time.time()

        response_text = run_claude_code(self.query.query, self.model, self.timeout)

        response_list = [
            WideSearchResponse(
                instance_id=self.query.instance_id,
                response=response_text,
                messages=None,
                trial_idx=self.trial_idx,
            )
        ]

        WideSearchResponseLoader.dump_response(response_list, self.response_path)
        elapsed = time.time() - start_time
        logger.info(
            f"infer end, instance_id: {self.query.instance_id}, cost(s): {elapsed:.2f}"
        )
        return response_list

    def eval(self) -> EvaluationResult:
        start_time = time.time()

        if os.path.exists(self.eval_result_path) and self.use_cache:
            with open(self.eval_result_path, "r") as f:
                eval_result = EvaluationResult(**json.load(f))
        else:
            if not os.path.exists(self.response_path):
                logger.error(f"response_path {self.response_path} not found, skip")
                response_list = [None]
            else:
                response_list = self.load_response()
            assert response_list, f"response is empty, response_path: {self.response_path}"

            eval_result = evaluate_single_query(
                self.query,
                response_list[0],
                self.result_save_path,
                self.eval_model_config_name,
            )
            eval_result_dict = dataclasses.asdict(eval_result)
            with open(self.eval_result_path, "w") as f:
                json.dump(eval_result_dict, f, ensure_ascii=False, indent=4)

        elapsed = time.time() - start_time
        logger.info(
            f"eval end, instance_id: {self.query.instance_id}, cost(s): {elapsed:.2f}"
        )
        return eval_result


def calc_summary_results(
    tasks: list[ClaudeCodeTask],
    summary_result_path: str,
    trial_num: int,
):
    """Aggregate per-instance evaluation results across trials."""
    metrics = [
        "score",
        "precision_by_row",
        "recall_by_row",
        "f1_by_row",
        "precision_by_item",
        "recall_by_item",
        "f1_by_item",
    ]

    all_results: dict[str, list] = {m: [] for m in metrics}
    id_to_task: dict[str, list[ClaudeCodeTask]] = {}
    for task in tasks:
        id_to_task.setdefault(task.query.instance_id, []).append(task)

    for iid, task_list in id_to_task.items():
        trial_metrics: dict[str, list] = {m: [] for m in metrics}
        for task in task_list:
            if not os.path.exists(task.eval_result_path):
                continue
            with open(task.eval_result_path, "r") as f:
                result = json.load(f)
            for m in metrics:
                if m in result:
                    trial_metrics[m].append(result[m])

        for m in metrics:
            values = trial_metrics[m]
            if not values or len(values) < trial_num:
                logger.info(f"Skipping {m} for instance {iid}, not enough trials")
                raise ValueError(
                    f"Not enough trials for metric {m} on instance {iid}. "
                    f"Expected {trial_num}, got {len(values)}."
                )
            all_results[m].append(
                {
                    "avg_n": float(np.mean(values)),
                    "max_n": float(np.max(values)),
                    "min_n": float(np.min(values)),
                }
            )

    summary: dict = {}
    for m in metrics:
        vals = all_results[m]
        if not vals:
            continue
        summary[m] = {
            "avg_n": float(np.mean([v["avg_n"] for v in vals])),
            "max_n": float(np.mean([v["max_n"] for v in vals])),
            "min_n": float(np.mean([v["min_n"] for v in vals])),
        }
    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(summary_result_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    parser = ArgumentParser(description="Run WideSearch with claude-code CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="haiku",
        help="Model name or alias passed to claude-code CLI (e.g. haiku, sonnet, claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--model_config_name",
        type=str,
        default="claude-code",
        help="Label used in output file names (default: claude-code)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["eval", "infer", "both"],
        help="Stage to run",
    )
    parser.add_argument(
        "--response_root",
        type=str,
        default="data/output",
        help="Directory for response JSONL files",
    )
    parser.add_argument(
        "--result_save_root",
        type=str,
        default="data/output",
        help="Directory for evaluation result files",
    )
    parser.add_argument(
        "--eval_model_config_name",
        type=str,
        default="default_eval_config",
        help="Eval LLM config name from src/utils/config.py",
    )
    parser.add_argument(
        "--trial_num", type=int, default=1, help="Number of trials per task"
    )
    parser.add_argument(
        "--instance_id",
        type=str,
        default="",
        help="Comma-separated instance IDs to run (empty = all)",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Skip tasks whose response/eval files already exist",
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        default=4,
        help="Number of threads for infer and eval",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for each claude-code invocation",
    )

    args = parser.parse_args()

    trial_num = args.trial_num
    model_config_name = args.model_config_name
    response_root = args.response_root
    result_save_root = args.result_save_root

    data_loader = WideSearchDataLoaderHF()
    instance_id_list = data_loader.get_instance_id_list()

    tasks: list[ClaudeCodeTask] = []

    for instance_id in instance_id_list:
        if args.instance_id and instance_id not in args.instance_id.split(","):
            continue
        query = data_loader.load_query_by_instance_id(instance_id)

        for trial_idx in range(trial_num):
            response_path = (
                f"{response_root}/{model_config_name}_{instance_id}"
                f"_{trial_idx}_response.jsonl"
            )
            result_save_path = (
                f"{result_save_root}/{model_config_name}_{instance_id}"
                f"_{trial_idx}_eval_result.csv"
            )
            os.makedirs(result_save_root, exist_ok=True)

            tasks.append(
                ClaudeCodeTask(
                    query=deepcopy(query),
                    response_path=response_path,
                    result_save_path=result_save_path,
                    trial_idx=trial_idx,
                    model=args.model,
                    timeout=args.timeout,
                    use_cache=args.use_cache,
                    eval_model_config_name=args.eval_model_config_name,
                )
            )

    logger.info(f"total task num: {len(tasks)}")

    if args.stage in ["infer", "both"]:
        with ThreadPoolExecutor(max_workers=args.thread_num) as executor:
            results = executor.map(lambda t: t.infer(), tasks)
            try:
                for result in results:
                    logger.info(f"infer success, instance_id: {result[0].instance_id}")
            except Exception:
                logger.error(f"infer error: {traceback.format_exc()}")

    if args.stage in ["eval", "both"]:
        with ThreadPoolExecutor(max_workers=args.thread_num) as executor:
            results = executor.map(lambda t: t.eval(), tasks)
            try:
                for result in results:
                    logger.info(f"eval success, instance_id: {result.instance_id}")
            except Exception as e:
                logger.error(f"eval error: {e}")

        summary_result_path = (
            f"{result_save_root}/{model_config_name}"
            f"_trial_num_{trial_num}_summary.json"
        )
        calc_summary_results(
            tasks=tasks,
            summary_result_path=summary_result_path,
            trial_num=trial_num,
        )
