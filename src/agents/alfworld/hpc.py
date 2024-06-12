import sys
import os
import json
import re
import itertools

from typing import List, Callable

class HPCAgent:
    """
    HPC Agent class.
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
                 global_memory,
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
        self.global_memory = global_memory(logging_dir)
        self.prompt_builder = prompt_builder()
        self.fewshot_builder = fewshot_builder()
        self.logger = Logger(self.logging_dir, self.num_trials, self.num_envs, self.local_memory, self.global_memory)
    
    def run(self):
        for trial_idx in range(self.num_trials):
            self.logger.log_world_start(trial_idx)
            num_successes: int = 0
            num_additional_successes: int = 0
            for env_idx in range(self.num_envs):
                init_ob, info = self.env.reset()
                print(f"{env_idx} using {self.env.name}")
                # if not init_ob.__contains__('put some vase on safe.') and not info.__contains__('put some vase on safe.'):
                #     continue
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
                else:
                    self.logger.log_world_fail(trial_idx, env_idx)
                self.logger.log_trial_content(history_log, is_success, trial_idx, env_idx)
                expert_trajectory = self.update_local_memory(history_log, is_success, env_idx)
                self.logger.log_local_memory(trial_idx)
                self.update_global_memory(expert_trajectory, env_idx, trial_idx)
                self.logger.log_global_memory(trial_idx)
            self.env.close()
            self.logger.log_trial_end(trial_idx, num_successes, num_additional_successes)
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
    
    def update_local_memory(self, history_log, is_success, env_idx):
        if not self.local_memory.is_skip(env_idx):
            reflection_prompt = self.build_reflection_prompt(history_log, is_success, env_idx)
            reflection_result = self.llm(reflection_prompt, max_tokens=512) 
            expert_trajectory = self.process_after_reflection(reflection_result, history_log, is_success)
            self.local_memory.add(env_idx, expert_trajectory)
        return expert_trajectory
            
    def update_global_memory(self, expert_trajectory, env_idx, trial_idx):
        env_summary = task_summary = ''
        env_query, task_query = self.build_summary_prompt(expert_trajectory, env_idx, trial_idx)
        if env_query:
            env_summary = self.llm(env_query, max_tokens=512) 
            self.global_memory.add(env_summary, expert_trajectory, mode='env')
        if task_query:
            task_summary = self.llm(task_query, max_tokens=512) 
            self.global_memory.add(task_summary, expert_trajectory, mode='task')

    def build_infer_prompt(self, env_idx, init_ob):
        short_memories = self.short_memory.recall()
        local_memories = self.local_memory.recall(env_idx)
        if len(local_memories) > 3:
            local_memories = local_memories[-3:]
        env_description, task_description = self.process_before_infer(init_ob)
        known_obs_history, action_guidance_history = self.global_memory.recall(env_description , task_description)
        fewshots = self.fewshot_builder.get_inference_fewshots(self.env.name, env_description , task_description, self.global_memory, self.logging_dir)
        query = self.prompt_builder.get_inference_prompts(init_ob, fewshots, local_memories, short_memories, known_obs_history, action_guidance_history)
        return query
        
    def build_reflection_prompt(self, history_log, is_success, env_idx):
        local_memories = self.local_memory.recall(env_idx)
        if len(local_memories) > 3:
            local_memories = local_memories[-3:]
        fewshots = self.fewshot_builder.get_reflection_fewshots(is_success)
        query = self.prompt_builder.get_reflection_prompts(history_log, is_success, fewshots, local_memories)
        return query
    
    def build_summary_prompt(self, expert_trajectory, env_idx, trial_idx):
        env_query = task_query = ''
        increment_env, increment_task = \
                self.global_memory.short2long(expert_trajectory, env_idx, trial_idx)
        is_success = expert_trajectory['is_success']
        if len(increment_env) != 0:
            env_fewshots = self.fewshot_builder.get_summary_fewshots('env')
            env_query = self.prompt_builder.env_summary_prompts(increment_env, env_fewshots)
        if len(increment_task) != 0:
            task_fewshots = self.fewshot_builder.get_summary_fewshots('task', is_success)
            task_query = self.prompt_builder.task_summary_prompts(increment_task, task_fewshots, is_success)
        return env_query, task_query
    
    def process_after_reflection(self, reflection_result, history_log, is_success):
        scenario = history_log.split("Here is the task:")[-1].strip()
        env_pattern = r'You are in the middle of a room\..*?(?=\n)'
        task_pattern = r'Your task is to: (.*?)(?=\n)'
        known_obs_pattern = r'KNOWN OBS: (.*?)(?:\n|$)'
        my_actions_pattern = r'MY ACTIONS: (.*?)(?:\n|$)'
        reflection_pattern = r'REFLECTION: (.*?)(?:\n|$)'

        env_description = task_description  = ''
        known_obs = my_actions = reflection = ''
        
        env_match = re.search(env_pattern, scenario, re.DOTALL)
        if env_match:
            env_description = env_match.group(0).strip()

        task_match = re.search(task_pattern, scenario, re.DOTALL)
        if task_match:
            task_description = task_match.group(1).strip()

        known_obs_match = re.search(known_obs_pattern, reflection_result, re.DOTALL)
        if known_obs_match:
            known_obs = known_obs_match.group(1).strip()

        my_actions_match = re.search(my_actions_pattern, reflection_result, re.DOTALL)
        if my_actions_match:
            my_actions = my_actions_match.group(1).strip()

        reflection_match = re.search(reflection_pattern, reflection_result, re.DOTALL)
        if reflection_match:
            reflection = reflection_match.group(1).strip()
            
        expert_trajectory = dict(env=env_description,
                             task=task_description,
                             known_obs=known_obs,
                             my_actions=my_actions,
                             memory=reflection,
                             is_success=is_success
                             )
        return expert_trajectory
    
    def process_before_infer(self, init_ob): 
        env_description = task_description = ''
        env_pattern = r'You are in the middle of a room\..*?(?=\n)'
        task_pattern = r'Your task is to:\s*(.*)'

        env_match = re.search(env_pattern, init_ob, re.DOTALL)
        if env_match:
            env_description = env_match.group(0).strip()

        task_match = re.search(task_pattern, init_ob, re.DOTALL)
        if task_match:
            task_description = task_match.group(1).strip()
        
        return env_description, task_description
    

class Logger:
    def __init__(self, logging_dir, num_trials, num_envs, local_memory, global_memory):
        self.logging_dir = logging_dir
        self.num_trials = num_trials
        self.num_envs = num_envs
        self.local_memory = local_memory
        self.global_memory = global_memory
        self.world_log_path = os.path.join(self.logging_dir, 'world.log')
        self.trial_log_paths = [os.path.join(self.logging_dir, f'trial_{trial_idx}.log') for trial_idx in range(self.num_trials)]
        self.local_memory_paths = [os.path.join(self.logging_dir, f'local_memory_trial_{trial_idx}.json') for trial_idx in range(self.num_trials)]
        self.global_env_paths = [os.path.join(self.logging_dir, f'global_env_trial_{trial_idx}.json') for trial_idx in range(self.num_trials)]
        self.global_task_paths = [os.path.join(self.logging_dir, f'global_task_trial_{trial_idx}.json') for trial_idx in range(self.num_trials)]
        for path in list(itertools.chain(self.trial_log_paths, self.local_memory_paths, self.global_env_paths, self.global_task_paths)):
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
        with open(self.global_env_paths[trial_idx], 'w') as wf:
            json.dump(self.global_memory.env_memory, wf, indent=4)
        with open(self.global_task_paths[trial_idx], 'w') as wf:
            json.dump(self.global_memory.task_memory, wf, indent=4)
            
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
        