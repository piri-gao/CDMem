import sys
import os
import json
import itertools
import re

from typing import List, Callable

class ExpelAgent:
    """
    Expel Agent class.
    """
    def __init__(self, 
                 num_trials,
                 num_envs,
                 max_steps,
                 logging_dir, 
                 env, 
                 llm_wrapper, 
                 model, 
                 start_trial_num,
                 short_memory, 
                 local_memory, 
                 global_memory,
                 prompt_builder, 
                 fewshot_builder, 
                 *args, 
                 **kwargs):
        self.num_trials = num_trials
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.logging_dir = logging_dir
        self.start_trial_num = start_trial_num
        self.env = env()
        self.model = model
        self.llm = llm_wrapper(model)
        self.short_memory = short_memory()
        self.local_memory = local_memory(num_envs)
        self.global_memory = global_memory(logging_dir)
        self.prompt_builder = prompt_builder()
        self.fewshot_builder = fewshot_builder()
        if self.start_trial_num > 0:
            trial_env_configs_log_path: str = os.path.join(self.logging_dir, f'local_memory_trial_{self.start_trial_num-1}.json')
            self.local_memory.resume(trial_env_configs_log_path)
        self.logger = Logger(self.logging_dir, self.num_trials, self.num_envs, self.start_trial_num, self.local_memory, self.global_memory)
    
    def run(self):
        for trial_idx in range(self.start_trial_num, self.num_trials):
            self.logger.log_world_start(trial_idx)
            num_successes: int = 0
            num_additional_successes: int = 0
            for env_idx in range(self.num_envs):
                init_ob, info = self.env.reset()
                print(f"{env_idx} using {self.env.name}")
                if self.local_memory.is_success(env_idx):
                    num_successes += 1
                    self.logger.log_world_success(trial_idx, env_idx)
                    self.logger.log_trial_success(trial_idx, env_idx)
                    continue
                history_log, is_success = self.run_trajectory(env_idx, init_ob)
                if is_success:
                    self.logger.log_world_success(trial_idx, env_idx)
                    self.local_memory.set_success(env_idx)
                    num_successes += 1
                    num_additional_successes += 1
                    self.update_global_memory(history_log, trial_idx, env_idx)
                else:
                    self.logger.log_world_fail(trial_idx, env_idx)
                self.logger.log_trial_content(history_log, is_success, trial_idx, env_idx)
                self.update_local_memory(history_log, env_idx)
                self.logger.log_local_memory(trial_idx)
            self.env.close()
            self.logger.log_world_end(trial_idx, num_successes, num_additional_successes)
            self.env.reload()
    
    def run_trajectory(self, env_idx, init_ob, to_print=True):
        cur_step = 0
        print(init_ob)
        self.short_memory.reset()
        while cur_step < self.max_steps:
            infer_prompt = self.build_infer_prompt(env_idx, init_ob)
            action = self.llm(infer_prompt, stop=["\n"]).strip()
            action = self.env.action_parser(action)
            self.short_memory.add("action", action)
            observation, reward, done, exhausted, info = self.env.step(action)
            self.short_memory.add("observation", observation)
            print_text = f'> {action}\n{observation}'    
            if to_print:
                print(print_text)
                sys.stdout.flush()
            if done:
                history_log = self.build_infer_prompt(env_idx, init_ob)
                return history_log, True
            elif exhausted:
                history_log = self.build_infer_prompt(env_idx, init_ob)
                return history_log, False
            if action.startswith('think:'):
                continue
            cur_step += 1
        history_log = self.build_infer_prompt(env_idx, init_ob)
        return history_log, False
    
    def update_local_memory(self, log_str, env_idx):
        if not self.local_memory.is_success(env_idx) and not self.local_memory.is_skip(env_idx):
            reflection_prompt = self.build_reflection_prompt(log_str, env_idx)
            reflection = self.llm(reflection_prompt) 
            self.local_memory.add(env_idx, reflection)
            
    def update_global_memory(self, history_log, trial_idx, env_idx):
        success_traj = history_log.split("Here is the task:")[-1].strip()
        existing_rules = self.global_memory.recall()
        success_guidelines_prompt = self.build_success_guidelines_prompts(success_traj, existing_rules)
        new_guidelines = self.llm(success_guidelines_prompt)
        self.global_memory.update(new_guidelines)
        for idx in range(trial_idx):
            with open(self.logger.trial_log_paths[idx], 'r') as f:
                full_log = f.read()
            fail_traj = full_log.split("#####\n\n#####")[env_idx].split("Here is the task:")[-1].split("STATUS:")[0].strip()
            pair_guidelines_prompt = self.build_pair_guidelines_prompts(fail_traj, success_traj, existing_rules)
            new_guidelines = self.llm(pair_guidelines_prompt)
            self.global_memory.update(new_guidelines)

    def build_infer_prompt(self, env_idx, init_ob):
        short_memories = self.short_memory.recall()
        local_memories = self.local_memory.recall(env_idx)
        existing_rules = self.global_memory.recall()
        if len(local_memories) > 3:
            local_memories = local_memories[-3:]
        fewshots = self.fewshot_builder.get_inference_fewshots(self.env.name)
        query = self.prompt_builder.get_inference_prompts(init_ob, fewshots, local_memories, short_memories, existing_rules)
        # if existing_rules and existing_rules != 'None':
        #     import pdb;pdb.set_trace()
        return query
        
    def build_reflection_prompt(self, log_str, env_idx):
        local_memories = self.local_memory.recall(env_idx)
        if len(local_memories) > 3:
            local_memories = local_memories[-3:]
        fewshots = self.fewshot_builder.get_reflection_fewshots()
        query = self.prompt_builder.get_reflection_prompts(log_str, fewshots, local_memories)
        return query
    
    def build_pair_guidelines_prompts(self, fail_traj, success_traj, existing_rules):
        query = self.prompt_builder.get_pair_guidelines_prompts(fail_traj, success_traj, existing_rules)
        return query
    
    def build_success_guidelines_prompts(self, success_trajs, existing_rules):
        query = self.prompt_builder.get_success_guidelines_prompts(success_trajs, existing_rules)
        return query
    
        
        
class Logger:
    def __init__(self, logging_dir, num_trials, num_envs, start_trial_num, local_memory, global_memory):
        self.logging_dir = logging_dir
        self.num_trials = num_trials
        self.num_envs = num_envs
        self.local_memory = local_memory
        self.global_memory = global_memory
        self.world_log_path = os.path.join(self.logging_dir, 'world.log')
        self.trial_log_paths = [os.path.join(self.logging_dir, f'trial_{trial_idx}.log') for trial_idx in range(self.num_trials)]
        self.local_memory_paths = [os.path.join(self.logging_dir, f'local_memory_trial_{trial_idx}.json') for trial_idx in range(self.num_trials)]
        self.global_memory_paths = [os.path.join(self.logging_dir, f'global_memory_trial_{trial_idx}.json') for trial_idx in range(self.num_trials)]
        for path in list(itertools.chain(self.trial_log_paths[start_trial_num:], self.local_memory_paths[start_trial_num:], 
                                            self.global_memory_paths[start_trial_num:])):
            if os.path.exists(path):
                open(path, 'w').close()
                
    def log_world_start(self, trial_idx):
        with open(self.world_log_path, 'a') as wf:
            wf.write(f'\n\n***** Start Trial #{trial_idx} *****\n\n')
            
    def log_world_success(self, trial_idx, env_idx):
        with open(self.world_log_path, 'a') as wf:
            wf.write(f'Environment #{env_idx} Trial #{trial_idx}: SUCCESS\n')
            
    def log_world_fail(self, trial_idx, env_idx):
        with open(self.world_log_path, 'a') as wf:
            wf.write(f'Environment #{env_idx} Trial #{trial_idx}: FAIL\n')  
            
    def log_world_end(self, trial_idx, num_successes, num_additional_successes):
        log_str = self._get_stats_str(num_successes, num_additional_successes)
        with open(self.world_log_path, 'a') as wf:
            wf.write(log_str + '\n')
            wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')
    
    def log_trial_success(self, trial_idx, env_idx):
        with open(self.trial_log_paths[trial_idx], 'a') as wf:
            wf.write(f'\n#####\n\nEnvironment #{env_idx}: Success\n\n#####\n')
            
    def log_trial_content(self, content, is_success, trial_idx, env_idx):
        with open(self.trial_log_paths[trial_idx], 'a') as wf:
            wf.write(f'\n#####\n\nEnvironment #{env_idx}:\n{content}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')
    
    def log_trial_end(self, trial_idx, num_successes, num_additional_successes):
        log_str = self._get_stats_str(num_successes, num_additional_successes)
        with open(self.trial_log_paths[trial_idx], 'a') as wf:
            wf.write(log_str)
    
    def log_local_memory(self, trial_idx):
        with open(self.local_memory_paths[trial_idx], 'w') as wf:
            json.dump(self.local_memory.history, wf, indent=4)
            
    def log_global_memory(self, trial_idx):
        with open(self.global_memory_paths[trial_idx], 'w') as wf:
            json.dump(self.global_memory.history, wf, indent=4)
            
    def _get_stats_str(self, num_successes, num_additional_successes):
        log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {self.num_envs - num_successes}
TOTAL: {self.num_envs}
ACCURACY: {round(num_successes / self.num_envs, 2)}
-----"""
        return log_str