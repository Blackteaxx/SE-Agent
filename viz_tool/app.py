import json
import os
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__, template_folder="templates", static_folder="static")

ROOT_DIR = os.environ.get(
    "VIZ_ROOT",
    "/workspace/CodeEfficiency/SE-Agent/trajectories_perf/deepseek-v3.1/deepseek-v3.1-Plan-AutoSelect-context_aware_direct_20251205_081856",
)


def _safe_read_json(path: Path):
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_llm_io(llm_path: Path):
    by_iter: dict[str, list[dict]] = {}
    if not llm_path.exists():
        return {"by_iteration": {}}
    try:
        with llm_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                # Strictly use iteration_index when available, per user requirement
                it = entry.get("iteration_index")
                if it is None:
                    it = entry.get("iteration")
                # If still None, skip this entry to avoid incorrect grouping
                if it is None:
                    continue
                try:
                    it = int(it)
                except Exception:
                    continue

                simplified = {
                    "ts": entry.get("ts"),
                    "context": entry.get("context"),
                    "model": entry.get("model"),
                    # Keep messages available for future use, but frontend will only display context
                    "messages": [
                        {
                            "role": (m.get("role") if isinstance(m, dict) else None),
                            "content": (m.get("content") if isinstance(m, dict) else None),
                        }
                        for m in (entry.get("messages") or [])
                    ],
                }
                by_iter.setdefault(str(it), []).append(simplified)
    except Exception:
        return {"by_iteration": {}}
    return {"by_iteration": by_iter}


def _load_instances(root: Path):
    result = {}
    if not root.exists() or not root.is_dir():
        return result
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        traj_path = p / "traj.pool"
        llm_path = p / "llm_io.jsonl"
        if not traj_path.exists():
            continue
        pool_data = _safe_read_json(traj_path)
        if not isinstance(pool_data, dict):
            continue
        for inst, inst_data in pool_data.items():
            if not isinstance(inst_data, dict):
                continue
            inst_data = dict(inst_data)
            inst_data["llm_io"] = _load_llm_io(llm_path)
            result[inst] = inst_data
    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def get_data():
    root = request.args.get("root") or ROOT_DIR
    data = _load_instances(Path(root))
    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
