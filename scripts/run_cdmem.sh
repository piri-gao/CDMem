source ./set_api.sh

cd ..
python src/main.py \
        --num_trials 5 \
        --num_envs 134 \
        --run_name "cdmem" \
        --model "gpt-4o-mini"  \
        --agent "cdmem"   \
        --env "alfworld" \
        # --is_vector
        # --is_resume \
        # --resume_dir logs/ \
        # --start_trial_num 4
        