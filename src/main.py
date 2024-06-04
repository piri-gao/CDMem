import os
from datetime import datetime
import argparse
from agents import AGENT
from envs import ENV
from llms import LLM_WRAPPER
from memory import SHORT_MEMORY, LOCAL_MEMORY, GLOBAL_MEMORY
from prompts import PROMPT_BUILDER
from retrievals import FEWSHOT_BUILDER

from typing import Any, List, Dict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, help="The number of trials to run")
    parser.add_argument("--num_envs", type=int, help="The number of environments per trial")
    parser.add_argument("--max_steps", type=int, help="he number of steps per trajectory", default=20)
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--is_resume", action='store_true', help="To resume run")
    parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
    parser.add_argument("--start_trial_num", type=int, help="If resume, the start trial num", default=0)
    parser.add_argument("--model", type=str, help="The model to use. One of `gpt-4`, `gpt-3.5-turbo`, or `text-davinci-003")
    parser.add_argument("--agent", type=str, help="The agent to use. One of `reflect`, `hpc`")
    parser.add_argument("--env", type=str, help="The enviroment to use. One of `alfworld`")

    args = parser.parse_args()

    assert args.num_trials > 0, "Number of trials should be positive"
    assert args.num_envs > 0, "Number of environments should be positive"

    return args

def main(args):
    if args.is_resume:
        if not os.path.exists(args.resume_dir):
            raise ValueError(f"Resume directory `{args.resume_dir}` does not exist")
        logging_dir = args.resume_dir
    else:
        now = datetime.now()
        timestamp = now.strftime("%m%d_%H%M%S")
        logging_dir = os.path.join('./logs/', args.run_name + '_' + timestamp)
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
    
    agent = AGENT[args.env][args.agent](
                            num_trials = args.num_trials,
                            num_envs = args.num_envs,
                            max_steps = args.max_steps,
                            logging_dir=logging_dir,
                            model = args.model,
                            env = ENV[args.env],
                            llm_wrapper=LLM_WRAPPER[args.model.split('-')[0]],
                            short_memory = SHORT_MEMORY[args.env][args.agent],
                            local_memory = LOCAL_MEMORY[args.env][args.agent],
                            global_memory = GLOBAL_MEMORY[args.env][args.agent],
                            prompt_builder = PROMPT_BUILDER[args.env][args.agent],
                            fewshot_builder = FEWSHOT_BUILDER[args.env][args.agent]
                            )

    if args.is_resume:
        print(f"""
            -----
            Resuming run with the following parameters:
            Run name: {args.run_name}
            Number of trials: {args.num_trials}
            Number of environments: {args.num_envs}
            Agent: {args.agent}
            Resume trial number: {args.start_trial_num}

            Sending all logs to `{logging_dir}`
            -----
            """)
    else:
        print(f"""
            -----
            Starting run with the following parameters:
            Run name: {args.run_name}
            Number of trials: {args.num_trials}
            Number of environments: {args.num_envs}
            Agent: {args.agent}

            Sending all logs to `{logging_dir}`
            -----
            """)
    agent.run()
    
    
if __name__ == '__main__':
    args = get_args()
    main(args)

    