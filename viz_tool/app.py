import json

from flask import Flask, jsonify, render_template

app = Flask(__name__, template_folder="templates", static_folder="static")

DATA_PATH = "trajectories_perf/deepseek/deepseek-v3.1-Plan-AutoSelect-slot_20251203_131049/atcoder_abc380d_strange-mirroring/traj_backup.pool"


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def get_data():
    data = load_data()
    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
