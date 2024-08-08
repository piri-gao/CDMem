source ./set_api.sh

cd ..
python src/main.py \
        --num_trials 1 \
        --num_envs 134 \
        --run_name "react_logs_t5e67" \
        --model "gpt-4o-mini"  \
        --agent "react"   \
        --env "alfworld" \
