import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .backends.backend_utils import (
    BackendUnavailableError,
    cancel_submission,
    get_backend_url,
    get_batch_results,
    submit_batch,
    submit_code,
)
from .utils import (
    EFFIBENCH_REGISTRY,
    execute_with_timeout,
    get_full_code,
    get_sandbox_lang,
    materialize_function_from_code,
)


def raise_error(results: list[dict], code: str) -> None:
    for idx, result in enumerate(results):
        status = result["status"]
        test_input = result["input"] if len(result["input"]) < 2000 else result["input"][:2000] + "...truncated..."
        test_case_str = json.dumps({"input": test_input, "output": result["output"]})
        if status in ("waiting", "processing"):
            continue
        elif status == "timeout":
            raise TimeoutError(f"Test Case {idx + 1}: timed out.\nTest case: {test_case_str}")
        elif status == "oom":
            raise MemoryError(f"Test Case {idx + 1}: exceeded memory limit.\nTest case: {test_case_str}")
        elif status == "error" or result["exit_code"] != 0:
            program_output = result["text"]
            output_display = program_output if len(program_output) < 2000 else program_output[:2000] + "...truncated..."
            raise RuntimeError(
                f"Test Case {idx + 1}: runtime error.\nTest case: {test_case_str}\nProgram output: {output_display}\nCode: {code}\n"
            )


def postprocess_text(text: str) -> str:
    """Remove ANSI escape codes from a string."""
    # More comprehensive regex to catch all ANSI escape sequences
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)
    text = text.replace("\r\r\n", "\n")
    text = text.replace("\r\n", "\n").strip()

    return text


class EvaluatorTimeoutError(TimeoutError):
    pass


def run_tests(
    lang: str,
    solution: str,
    test_cases: list,
    evaluator: str,
    test_runner: str | None = None,
    time_limit: int = 10,
    memory_limit: int = 1024,
    early_stop: bool = True,
    raise_on_error: bool = True,
    as_batch: bool = True,
    backend_retries: int = 5,
    eval_timeout: int = 30,  # Increased from 10 to 30 seconds
    polling_interval: int = 5,
) -> list[dict]:
    """Runs a solution against a set of test cases and evaluates the results.

    This function takes a code solution, a set of test cases, and an evaluator,
    and then runs the solution against each test case in a sandboxed environment.
    It supports batch submission, time and memory limits, and retries.

    Args:
        lang: The programming language of the solution (e.g., "python", "java").
        solution: A string containing the source code of the solution to be tested.
        test_cases: A list of test cases, where each test case is a dictionary
            containing at least an "input" key for stdin.
        evaluator: A string containing Python code for an `evaluate` function
            that will be used to check the correctness of the solution's output.
        test_runner: An optional string containing test runner code to be combined
            with the solution code.
        time_limit: The time limit for each test case in seconds. Defaults to 10.
        memory_limit: The memory limit for each test case in megabytes. Defaults to 1024.
        early_stop: If True, testing will stop after the first failed test case.
            Defaults to True.
        raise_on_error: If True, an exception will be raised if an error occurs
            during testing. Defaults to True.
        as_batch: If True, all test cases are submitted to the backend as a single
            batch. Otherwise, they are submitted concurrently. Defaults to True.
        backend_retries: The number of times to retry connecting to the backend.
            Defaults to 5.
        eval_timeout: The timeout for the entire evaluation process in seconds.
            Defaults to 10.
        polling_interval: The interval in seconds at which to poll for test
            results. Defaults to 5.

    Returns:
        A list of dictionaries, where each dictionary represents the result of a
        single test case evaluation.
    """
    code = get_full_code(lang, solution, test_runner) if test_runner else solution
    sandbox_lang = get_sandbox_lang(lang)
    libraries = EFFIBENCH_REGISTRY.get(lang, {}).get("packages", [])

    if lang == "java" and "class Codechef" in code:
        code = code.replace("Codechef", "Main")

    batch_requests = [
        {
            "code": code,
            "language": sandbox_lang,
            "libraries": libraries,
            "stdin": tc["input"],
            "time_limit": time_limit,
            "memory_limit": memory_limit,
        }
        for tc in test_cases
    ]
    # Estimate payload size (stdin dominates) to tune request timeout for large inputs
    try:
        total_stdin_bytes = sum(len(req["stdin"]) if req.get("stdin") else 0 for req in batch_requests)
    except Exception:
        total_stdin_bytes = 0
    estimated_payload_mb = total_stdin_bytes / (1024 * 1024)
    submit_timeout = 1200
    if estimated_payload_mb >= 200:
        submit_timeout = 3600
    logging.debug(f"Estimated submit payload: {estimated_payload_mb:.1f} MB; request_timeout={submit_timeout}s")

    evaluate = materialize_function_from_code(evaluator, "evaluate")
    for retry_count in range(backend_retries + 1):
        try:
            backend_base_url = get_backend_url()

            if as_batch:
                sids = submit_batch(
                    batch_requests,
                    backend_base_url=backend_base_url,
                    request_timeout=submit_timeout,
                )
            else:
                sids = [None] * len(batch_requests)
                with ThreadPoolExecutor(len(batch_requests)) as executor:
                    future_to_tid = {
                        executor.submit(
                            submit_code,
                            code=req["code"],
                            language=req["language"],
                            libraries=req["libraries"],
                            stdin=req["stdin"],
                            time_limit=req["time_limit"],
                            memory_limit=req["memory_limit"],
                            backend_base_url=backend_base_url,
                            request_timeout=submit_timeout,
                        ): tid
                        for tid, req in enumerate(batch_requests)
                    }

                    for future in as_completed(future_to_tid):
                        sids[future_to_tid[future]] = future.result()

            sid_to_tid = {sid: tid for tid, sid in enumerate(sids)}
            all_results = [None] * len(test_cases)
            pending_ids = set(sids)
            # Retry tracker for rare cases where stdout is empty despite successful execution
            empty_text_retries: dict[str, int] = {}
            # Fallback tracker: resubmit at most once per test to avoid permanent empty stdout
            resubmit_counts: dict[int, int] = {}

            while len(pending_ids):
                # 动态缩短轮询间隔：当存在空 stdout 的快速重试时，优先用更短的等待
                if empty_text_retries:
                    time.sleep(min(0.1, polling_interval))
                else:
                    time.sleep(polling_interval)

                batch_results = get_batch_results(list(pending_ids), backend_base_url=backend_base_url)
                new_result_ids = set()
                for result_data in batch_results:
                    sid = result_data["submission_id"]
                    assert sid in pending_ids, f"Submission ID {sid} not in pending IDs"

                    if result_data["status"] not in ("waiting", "processing"):
                        tid = sid_to_tid[sid]
                        # Guard against rare race where stdout is not yet captured
                        try:
                            output_text = postprocess_text(result_data.get("text", ""))
                            expected_text = postprocess_text(test_cases[tid].get("output", ""))
                        except Exception:
                            output_text, expected_text = (
                                result_data.get("text", ""),
                                test_cases[tid].get("output", ""),
                            )

                        if result_data.get("exit_code") == 0 and output_text == "" and expected_text != "":
                            # Retry polling this submission a few times to allow backend to finalize stdout
                            retry_count_local = empty_text_retries.get(sid, 0)
                            # 降低最大重试数以更快进入后续处理/回退逻辑
                            max_empty_text_retries = 3
                            if retry_count_local < max_empty_text_retries:
                                empty_text_retries[sid] = retry_count_local + 1
                                # Exponential backoff bounded to 4s to reduce race likelihood
                                # 更快的指数退避：起始 50ms，最大 500ms
                                backoff = min(0.05 * (2**retry_count_local), 0.5)
                                time.sleep(backoff)
                                continue
                            # If stdout is still empty after retries, resubmit this single test once.
                            tid_for_resubmit = sid_to_tid[sid]
                            if resubmit_counts.get(tid_for_resubmit, 0) < 1:
                                req = batch_requests[tid_for_resubmit]
                                new_sid = submit_code(
                                    code=req["code"],
                                    language=req["language"],
                                    libraries=req["libraries"],
                                    stdin=req["stdin"],
                                    time_limit=req["time_limit"],
                                    memory_limit=req["memory_limit"],
                                    backend_base_url=backend_base_url,
                                )
                                # Replace old sid with new sid in tracking structures
                                pending_ids.discard(sid)
                                sid_to_tid.pop(sid, None)
                                sid_to_tid[new_sid] = tid_for_resubmit
                                pending_ids.add(new_sid)
                                resubmit_counts[tid_for_resubmit] = 1
                                empty_text_retries[new_sid] = 0
                                # Skip finalization for the old sid; wait for the new one
                                continue

                        all_results[tid] = {**result_data, **test_cases[tid]}
                        pending_ids.remove(sid)
                        new_result_ids.add(sid)
                new_results = [all_results[sid_to_tid[sid]] for sid in new_result_ids]

                try:
                    raise_error(new_results, code)
                except Exception:
                    if raise_on_error or early_stop:
                        if pending_ids:
                            with ThreadPoolExecutor(max_workers=len(pending_ids)) as cancel_executor:
                                cancel_futures = [
                                    cancel_executor.submit(
                                        cancel_submission,
                                        sid_to_cancel,
                                        backend_base_url=backend_base_url,
                                    )
                                    for sid_to_cancel in pending_ids
                                ]
                                [None for f in as_completed(cancel_futures)]

                    if raise_on_error:
                        raise
                    if early_stop:
                        return [res for res in all_results if res]

                for idx, result_data in enumerate(new_results):
                    # Skip already evaluated results
                    if not result_data or "passed" in result_data:
                        continue

                    # For successful executions, evaluate the output
                    if result_data.get("status") == "done" and result_data.get("exit_code") == 0:
                        output = postprocess_text(result_data.get("text", ""))
                        expected = postprocess_text(result_data.get("output", ""))
                        try:
                            passed = execute_with_timeout(evaluate, eval_timeout, output, expected)
                            result_data["passed"] = passed

                            if not passed and raise_on_error:
                                test_input = result_data.get("input", "")
                                test_input_display = (
                                    test_input if len(test_input) < 2000 else test_input[:2000] + "...truncated..."
                                )
                                test_case_str = json.dumps(
                                    {
                                        "input": test_input_display,
                                        "output": result_data.get("output", ""),
                                    }
                                )
                                output_display = output if len(output) < 2000 else output[:2000] + "...truncated..."
                                raise AssertionError(
                                    f"Test Case {idx + 1}: failed.\nTest case: {test_case_str}\nProgram output: {output_display}\nCode: {code}\n"
                                )
                        except TimeoutError:
                            result_data["passed"] = False
                            if raise_on_error:
                                test_input = result_data.get("input", "")
                                test_input_display = (
                                    test_input if len(test_input) < 2000 else test_input[:2000] + "...truncated..."
                                )
                                test_case_str = json.dumps(
                                    {
                                        "input": test_input_display,
                                        "output": result_data.get("output", ""),
                                    }
                                )
                                raise EvaluatorTimeoutError(
                                    f"Solution Evaluator timed out after {eval_timeout} seconds.\nTest case: {test_case_str}\nCode: {code}\n"
                                )
                        except Exception:
                            result_data["passed"] = False

                    # Default to failed unless explicitly evaluated as passing
                    if "passed" not in result_data:
                        result_data["passed"] = False

            assert all(result_data for result_data in all_results), "Some test cases failed to run"
            return all_results

        except BackendUnavailableError as e:
            if retry_count < backend_retries:
                # 计算重试延迟：基础延迟 + 指数退避 + 随机抖动
                base_delay = 2.0  # 基础延迟2秒
                exponential_delay = base_delay * (2**retry_count)  # 指数退避
                jitter = random.uniform(0.5, 1.5)  # 随机抖动因子
                retry_delay = min(exponential_delay * jitter, 30.0)  # 最大延迟30秒

                logging.warning(
                    f"Backend unavailable during execution (attempt {retry_count + 1}/{backend_retries + 1}): {e}. "
                    f"Retrying in {retry_delay:.2f} seconds..."
                )
                time.sleep(retry_delay)
                continue
            logging.error(f"All backends failed after {backend_retries} retries: {e}")
            raise
    return None
