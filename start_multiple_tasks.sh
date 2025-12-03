export BACKEND_BASE_URL=http://192.168.100.115:8000,http://192.168.100.116:8000
config_path=configs/se_perf_configs/deepseek-v3.1-test.yaml

# python SE_Perf/perf_run.py --config configs/se_perf_configs/deepseek-v3.1-Plan-AutoSelect.yaml --mode execute

python SE_Perf/instance_runner.py \
    --config $config_path \
    --mode execute \
    --max-parallel 20 \
    --limit 1

