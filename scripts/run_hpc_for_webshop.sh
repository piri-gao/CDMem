source ./set_openai.sh

cd ..
python src/main.py \
        --num_trials 5 \
        --num_envs 100 \
        --run_name "hpc_logs_webshop" \
        --model "gpt-3.5-turbo"  \
        --agent "hpc"   \
        --env "webshop" \
        # --is_resume \
        # --resume_dir logs/hpc_logs_t5e67_0628_050752 \
        # --start_trial_num 4
        
