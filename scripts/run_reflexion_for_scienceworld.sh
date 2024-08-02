source ./set_api.sh

cd ..
python src/main.py \
        --num_trials 5 \
        --num_envs 67 \
        --run_name "reflexion_logs_t5e67" \
        --model "gpt-3.5-turbo"  \
        --agent "reflect"   \
        --env "scienceworld"