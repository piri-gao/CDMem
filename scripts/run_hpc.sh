source ./set_api.sh

cd ..
python src/main.py \
        --num_trials 1 \
        --num_envs 134 \
        --run_name "hpc_logs_testenv" \
        --model "gpt-3.5-turbo"  \
        --agent "hpc"   \
        --env "alfworld"