# # run Qwen3-32B-3it-10Traj.yaml 3 times
# python SE/perf_run.py --config SE/configs/se_perf_configs/Qwen3-32B-3it-10Traj.yaml --mode execute
# python SE/perf_run.py --config SE/configs/se_perf_configs/Qwen3-32B-3it-10Traj.yaml --mode execute
# run Qwen3-32B-1it-10Traj.yaml 3 times
# python SE_single/perf_run.py --config SE/configs/se_perf_configs/Qwen3-32B-1it-10Traj.yaml --mode execute
# python SE_single/perf_run.py --config SE/configs/se_perf_configs/Qwen3-32B-1it-10Traj.yaml --mode execute
# python SE_single/perf_run.py --config SE/configs/se_perf_configs/Qwen3-32B-1it-10Traj.yaml --mode execute

BACKEND_BASE_URL=http://192.168.100.115:8000 python SE_AddiReq/perf_run.py --config configs/se_perf_configs/deepseek-v3.1-climb.yaml --mode execute