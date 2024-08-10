source ./set_api.sh

cd ..
python src/main.py \
        --num_trials 5 \
        --num_envs 67 \
        --run_name "expel_logs_t5e134_test" \
        --model "gpt-4o-mini"  \
        --agent "expel"   \
        --env "alfworld" \
        # --is_resume \
        # --resume_dir logs/expel_logs_t5e67_0806_051212 \
        # --start_trial_num 1