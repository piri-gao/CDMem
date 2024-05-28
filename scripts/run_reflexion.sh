source ./set_api.sh

cd ..
python src/main.py \
        --num_trials 2 \
        --num_envs 2 \
        --run_name "reflexion_logs_test" \
        --model "gpt-3.5-turbo"  \
        --agent "reflect"   \
        --env "alfworld"