import sys
import os
import json

from typing import List, Callable

class HPCAgent:
    """
    Relection Agent class.
    """
    def __init__(self, 
                 num_trials,
                 num_envs,
                 max_steps,
                 logging_dir, 
                 env, 
                 llm_wrapper, 
                 model, 
                 short_memory, 
                 local_memory, 
                 prompt_builder, 
                 fewshot_builder, 
                 *args, 
                 **kwargs):
        self.num_trials = num_trials
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.logging_dir = logging_dir
        self.env = env()
        self.model = model
        self.llm = llm_wrapper(model)
        self.short_memory = short_memory()
        self.local_memory = local_memory(num_envs)
        self.prompt_builder = prompt_builder()
        self.fewshot_builder = fewshot_builder()
    
    def run(self):
        world_log_path = os.path.join(self.logging_dir, 'world.log')
        for trial_idx in range(self.num_trials):
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
                if self.local_memory.is_success(env_idx):
                    num_successes += 1
                    with open(world_log_path, 'a') as wf:
                        wf.write(f'Environment #{env_idx} Trial #{trial_idx}: SUCCESS\n')
                    with open(trial_log_path, 'a') as wf:
                        wf.write(f'\n#####\n\nEnvironment #{env_idx}: Success\n\n#####\n')
                    continue
                history_log, is_success = self.run_trajectory(env_idx, init_ob)
                if is_success:
                    status_str: str = f'Environment #{env_idx} Trial #{trial_idx}: SUCCESS'
                    self.local_memory.set_success(env_idx)
                    num_successes += 1
                    num_additional_successes += 1
                else:
                    status_str: str = f'Environment #{env_idx} Trial #{trial_idx}: FAIL'
                with open(world_log_path, 'a') as f:
                    f.write(status_str + '\n')
                with open(trial_log_path, 'a') as wf:
                    wf.write(f'\n#####\n\nEnvironment #{env_idx}:\n{history_log}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')
                self.short_memory.reset()
                with open(trial_log_path, 'r') as f:
                    full_log: str = f.read()
                    env_logs: List[str] = full_log.split('#####\n\n#####')
                self.update_local_memory(env_logs[env_idx], env_idx)
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
            with open(trial_env_configs_log_path, 'w') as wf:
                json.dump(self.local_memory.history, wf, indent=4)
            with open(world_log_path, 'a') as wf:
                wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')
            self.env.reload()
    
    def run_trajectory(self, env_idx, init_ob, to_print=True):
        cur_step = 0
        while cur_step < self.max_steps:
            infer_prompt = self.build_infer_prompt(env_idx, init_ob)
            action = self.llm(infer_prompt,  stop=['\n']).strip('>').strip()
            self.short_memory.add("action", action)
            observation, reward, done, exhausted, info = self.env.step(action)
            self.short_memory.add("observation", observation)
            if to_print:
                print(f'> {action}\n{observation}')
                sys.stdout.flush()
            if done:
                history_log = self.build_infer_prompt(env_idx, init_ob)
                return history_log, True
            elif exhausted:
                history_log = self.build_infer_prompt(env_idx, init_ob)
                return history_log, False
            cur_step += 1
        history_log = self.build_infer_prompt(env_idx, init_ob)
        return history_log, False
    
    def update_local_memory(self, log_str, env_idx):
        if not self.local_memory.is_success(env_idx) and not self.local_memory.is_skip(env_idx):
            local_memories = self.local_memory.recall(env_idx)
            if len(local_memories) > 3:
                local_memories = local_memories[-3:]
            reflection_prompt = self.build_reflection_prompt(log_str, env_idx)
            reflection = self.llm(reflection_prompt) 
            self.local_memory.add(env_idx, reflection)
            
    def build_infer_prompt(self, env_idx, init_ob):
        short_memories = self.short_memory.recall()
        local_memories = self.local_memory.recall(env_idx)
        if len(local_memories) > 3:
            local_memories = local_memories[-3:]
        fewshots = self.fewshot_builder.get_inference_fewshots(self.env.name)
        query = self.prompt_builder.get_inference_prompts(init_ob, fewshots, local_memories, short_memories)
        return query
        
    def build_reflection_prompt(self, log_str, env_idx):
        local_memories = self.local_memory.recall(env_idx)
        fewshots = self.fewshot_builder.get_reflection_fewshots()
        query = self.prompt_builder.get_reflection_prompts(log_str, fewshots, local_memories)
        return query