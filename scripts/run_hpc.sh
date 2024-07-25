source ./set_api.sh

cd ..
python src/main.py \
        --num_trials 1 \
        --num_envs 20 \
        --run_name "hpc_logs_t1e20" \
        --model "gpt-3.5-turbo"  \
        --agent "hpc"   \
        --env "alfworld" \
        # --is_vector
        # --is_resume \
        # --resume_dir logs/hpc_logs_t5e67_0628_050752 \
        # --start_trial_num 4
        