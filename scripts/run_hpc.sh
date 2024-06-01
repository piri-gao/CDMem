source ./set_api.sh

cd ..
python src/main.py \
        --num_trials 1 \
        --num_envs 5 \
        --run_name "hpc_logs_t1e1" \
        --model "gpt-3.5-turbo"  \
        --agent "hpc"   \
        --env "alfworld"