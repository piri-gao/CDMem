import sys
import os
import json

from typing import List, Callable

class ReactAgent:
    """
    Relection Agent class.
    """
    def __init__(self, 
                 num_envs,
                 max_steps,
                 logging_dir, 
                 env, 
                 llm_wrapper, 
                 model, 
                 short_memory, 
                 prompt_builder, 
                 fewshot_builder, 
                 *args, 
                 **kwargs):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.logging_dir = logging_dir
        self.env = env()
        self.model = model
        self.llm = llm_wrapper(model)
        self.short_memory = short_memory()
        self.prompt_builder = prompt_builder()
        self.fewshot_builder = fewshot_builder()

    def run(self):
        world_log_path = os.path.join(self.logging_dir, 'world.log')
        for trial_idx in range(1):
            with open(world_log_path, 'a') as wf:
                wf.write(f'\n\n***** Start Trial #{trial_idx} *****\n\n')
            trial_log_path: str = os.path.join(self.logging_dir, f'trial_{trial_idx}.log')
            trial_env_configs_log_path: str = os.path.join(self.logging_dir, f'env_results_trial_{trial_idx}.json')
            if os.path.exists(trial_log_path):
                open(trial_log_path, 'w').close()
            if os.path.exists(trial_env_configs_log_path):
                open(trial_env_configs_log_path, 'w').close()
            num_successes: int = 0
            num_additional_successes: int = 0
            for env_idx in range(self.num_envs):
                init_ob, info = self.env.reset()
                history_log, is_success = self.run_trajectory(env_idx, init_ob)
                if is_success:
                    status_str: str = f'Environment #{env_idx} Trial #{trial_idx}: SUCCESS'
                    num_successes += 1
                    num_additional_successes += 1
                else:
                    status_str: str = f'Environment #{env_idx} Trial #{trial_idx}: FAIL'
                with open(world_log_path, 'a') as f:
                    f.write(status_str + '\n')
                with open(trial_log_path, 'a') as wf:
                    wf.write(f'\n#####\n\nEnvironment #{env_idx}:\n{history_log}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')
                self.short_memory.reset()
            self.env.close()
            log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {self.num_envs - num_successes}
TOTAL: {self.num_envs}
ACCURACY: {round(num_successes / self.num_envs, 2)}
-----"""
            with open(trial_log_path, 'a') as wf:
                wf.write(log_str)
            with open(world_log_path, 'a') as wf:
                wf.write(log_str + '\n')
            with open(world_log_path, 'a') as wf:
                wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')
            self.env.reload()
    
    def run_trajectory(self, env_idx, init_ob, to_print=True):
        cur_step = 0
        while cur_step < self.max_steps:
            infer_prompt = self.build_infer_prompt(env_idx, init_ob)
            action = self.llm(infer_prompt, stop=["\n"]).strip()
            if ">" in action:
                action = action.replace(">", "").strip()
            action_words = action.split(" ")
            if "put" in action_words:
                for i in range(len(action_words)):
                    if action_words[i].strip().lower() == "in" or action_words[i].strip().lower() == 'on':
                        action_words[i] = "in/on"
                        action = " ".join(action_words)
            self.short_memory.add("action", action)
            observation, reward, done, exhausted, info = self.env.step(action)
            self.short_memory.add("observation", observation)
            if to_print:
                print(f'> {action}\n{observation}')
                sys.stdout.flush()
            if action.startswith('think:'):
                continue
            if done:
                history_log = self.build_infer_prompt(env_idx, init_ob)
                return history_log, True
            elif exhausted:
                history_log = self.build_infer_prompt(env_idx, init_ob)
                return history_log, False
            cur_step += 1
        history_log = self.build_infer_prompt(env_idx, init_ob)
        return history_log, False
            
    def build_infer_prompt(self, env_idx, init_ob):
        short_memories = self.short_memory.recall()
        fewshots = self.fewshot_builder.get_inference_fewshots(self.env.name)
        query = self.prompt_builder.get_inference_prompts(init_ob, fewshots, [], short_memories)
        return query
        