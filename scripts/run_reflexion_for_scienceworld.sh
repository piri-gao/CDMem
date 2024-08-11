source ./set_api.sh

cd ..
python src/main.py \
        --num_trials 5 \
        --num_envs 50 \
        --run_name "scworld_logs_reflect" \
        --model "gpt-4o-mini"  \
        --agent "reflect"   \
        --env "scienceworld" \
        --start_trial_num 0