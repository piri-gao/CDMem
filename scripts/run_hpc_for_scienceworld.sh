source ./set_openai.sh

cd ..
python src/main.py \
        --num_trials 5 \
        --num_envs 50 \
        --run_name "scworld_logs_hpc" \
        --model "gpt-4o-mini"  \
        --agent "hpc"   \
        --env "scienceworld" \
        --start_trial_num 0
        # --is_resume \
        # --resume_dir logs/hpc_logs_t5e67_0628_050752 \
        
