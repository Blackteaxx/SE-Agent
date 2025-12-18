import json
import argparse
from pathlib import Path


def generate_order(json_path: str, output_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Assume the first key is the model name
    model_name = list(data.keys())[0]
    per_task = data[model_name].get("per_task", {})

    passed_tasks = []
    failed_tasks = []

    for task_name, lang_data in per_task.items():
        # Assume one language per task or take the first one
        if not lang_data:
            failed_tasks.append(task_name)
            continue

        # Get the first language dict (e.g., "python3")
        lang_key = list(lang_data.keys())[0]
        scores = lang_data[lang_key]

        integral_score = scores.get("integral_score", 0.0)

        if integral_score > 0:
            passed_tasks.append((task_name, integral_score))
        else:
            failed_tasks.append(task_name)

    # Sort passed tasks by integral_score descending
    passed_tasks.sort(key=lambda x: x[1], reverse=True)

    # Extract just the names
    passed_task_names = [t[0] for t in passed_tasks]

    # Combine
    final_order = passed_task_names + failed_tasks

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        for task in final_order:
            f.write(f"{task}\n")

    print(f"Generated order file at {output_path}")
    print(f"Passed tasks: {len(passed_tasks)}")
    print(f"Failed tasks: {len(failed_tasks)}")
    print(f"Total tasks: {len(final_order)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate task order from report JSON")
    parser.add_argument("json_path", help="Path to the report JSON file")
    parser.add_argument("output_path", help="Path to the output order file")

    args = parser.parse_args()
    generate_order(args.json_path, args.output_path)
