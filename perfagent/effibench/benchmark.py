import concurrent.futures
import os
from typing import Any, Dict, List

from .analysis import analyze_runtimes
from .run_tests import run_tests


def run_performance_benchmark(
    lang: str,
    solution: str,
    test_cases: List[Dict[str, Any]],
    evaluator: str,
    test_runner: str | None = None,
    num_runs: int = 5,
    time_limit: int = 10,
    memory_limit: int = 1024,
    trim_ratio: float = 0.1,
    max_workers: int = 4,
) -> Dict[str, float]:
    """
    Runs a performance benchmark for a given solution against a set of test cases.

    This function executes the solution multiple times concurrently to gather runtime data,
    calculates the pass rate, and then computes a trimmed mean of the runtimes for the
    successful runs to provide a robust performance metric.

    Args:
        lang: The programming language of the solution.
        solution: The source code of the solution to be benchmarked.
        test_cases: A list of test cases, where each test case is a dictionary.
        evaluator: The function used to evaluate the correctness of the solution's output.
        test_runner: The function used to run the solution against a test case.
        num_runs: The number of times to run the solution against the test cases.
        time_limit: The time limit in seconds for each test run.
        memory_limit: The memory limit in megabytes for each test run.
        trimming_fraction: The fraction of runtimes to trim from each end before calculating the mean.
        max_workers: The maximum number of concurrent workers to use for running tests.

    Returns:
        A dictionary containing the pass rate and the trimmed mean runtime.
    """
    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_tests,
                lang=lang,
                solution=solution,
                test_cases=test_cases,
                evaluator=evaluator,
                test_runner=test_runner,
                time_limit=time_limit,
                memory_limit=memory_limit,
                early_stop=False,  # We need to run all tests for performance
                raise_on_error=False,
                as_batch=True,
                polling_interval=10,
            )
            for _ in range(num_runs)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                # Handle exceptions from run_tests if necessary
                print(f"A test run failed with an exception: {e}")

    if not all_results:
        # If there are no results, return a default failure structure.
        analysis_results = {
            "original_n": 0,
            "n": 0,
            "mean": float("inf"),
            "std": float("inf"),
            "min": float("inf"),
            "max": float("inf"),
            "max_diff": float("inf"),
            "95%_CI": (float("inf"), float("inf")),
            "trimmed_mean": float("inf"),
        }
        return {
            "performance_analysis": analysis_results,
            "first_run_details": [],
            "failed_submission_exit_codes": [],
            "pass_rates": [],
            "pass_rate_consistent": False,
        }

    first_run_results = all_results[0]

    # Collect detailed information about failed test cases from the first run
    failed_test_details = []
    for tc in first_run_results:
        if not tc.get("passed", False):
            failure_details = {
                "status": tc.get("status"),
                "text": tc.get("text"),
                "exit_code": tc.get("exit_code"),
                # Provide more context for debugging mismatches
                "input": tc.get("input"),
                "expected": tc.get("output"),
            }
            failed_test_details.append(failure_details)

    # Compute pass rate consistency across all runs
    pass_rates = []
    for test_case_results in all_results:
        total_cases_run = len(test_case_results)
        num_passed_run = sum(1 for tc in test_case_results if tc.get("passed", False))
        pr = num_passed_run / total_cases_run if total_cases_run > 0 else 0.0
        pass_rates.append(pr)

    pass_rate_consistent = len(set(pass_rates)) == 1

    # Calculate pass rate from the first run
    num_passed = sum(1 for tc in first_run_results if tc.get("passed", False))
    total_cases = len(first_run_results)
    pass_rate = num_passed / total_cases if total_cases > 0 else 0.0

    # Print brief pass-rate consistency summary
    try:
        pr_str = ", ".join(f"{pr:.2f}" for pr in pass_rates)
    except Exception:
        pr_str = ", ".join(str(pr) for pr in pass_rates)
    consistency_label = "consistent" if pass_rate_consistent else "inconsistent"
    if not pass_rate_consistent:
        print(
            f"Pass rate consistency across {len(all_results)} runs: {consistency_label} | pass_rates: [{pr_str}] | first_run: {pass_rate:.2f}"
        )

    # Collect runtimes only if all tests passed
    successful_runtimes = []
    if pass_rate == 1.0:
        for test_case_results in all_results:
            # This check is for robustness, assuming subsequent runs should also pass if the first did.
            if bool(test_case_results) and all(tc.get("passed", False) for tc in test_case_results):
                total_runtime_ns = sum(tc.get("runtime", 0) for tc in test_case_results)
                total_runtime_s = total_runtime_ns / 1_000_000_000.0
                successful_runtimes.append(total_runtime_s)

    # Analyze runtimes if we have any successful (and complete) runs
    if successful_runtimes:
        analysis_results = analyze_runtimes(successful_runtimes, trim_ratio=trim_ratio)
    else:
        # This path is taken if pass_rate < 1.0 or if all runs failed unexpectedly
        analysis_results = {
            "original_n": 0,
            "n": 0,
            "mean": float("inf"),
            "std": float("inf"),
            "min": float("inf"),
            "max": float("inf"),
            "max_diff": float("inf"),
            "95%_CI": (float("inf"), float("inf")),
            "trimmed_mean": float("inf"),
        }

    # Construct the final return dictionary as requested
    return {
        "performance_analysis": analysis_results,
        "first_run_details": first_run_results,
        "failed_test_details": failed_test_details,
        "pass_rates": pass_rates,
        "pass_rate_consistent": pass_rate_consistent,
    }
