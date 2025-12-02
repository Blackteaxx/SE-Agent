export BACKEND_BASE_URL=http://192.168.100.115:8000,http://localhost:8000


# python SE_Perf/perf_run.py --config configs/se_perf_configs/deepseek-v3.1-Plan-AutoSelect.yaml --mode execute

python SE_Perf/instance_runner.py \
    --config configs/se_perf_configs/deepseek-v3.1-Plan-RandomSelect.yaml \
    --mode execute \
    --max-parallel 20

