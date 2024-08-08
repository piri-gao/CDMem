source ./set_api.sh

cd ..
python src/main.py \
        --num_trials 5 \
        --num_envs 134 \
        --run_name "reflexion_logs_t5e67" \
        --model "gpt-4o-mini"  \
        --agent "reflect"   \
        --env "alfworld" \
        # --is_resume \
        # --resume_dir logs/reflexion_logs_t5e67_0628_120045_test \
        # --start_trial_num 3