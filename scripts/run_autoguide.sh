source ./set_api.sh

cd ..
python src/main.py \
        --num_trials 5 \
        --num_envs 67 \
        --run_name "autoguide_logs_t5e67" \
        --model "gpt-4o-mini"  \
        --agent "autoguide"   \
        --env "alfworld" \
        --is_resume \
        --resume_dir logs/autoguide_logs_t5e67_0807_081555 \
        --start_trial_num 1