import sys
import os
import json

from typing import List, Callable

from tqdm import tqdm

from src.utils.scworld_utils import action_type_description, findValidActionNew


class ReflectAgent:
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
                 start_trial_num,
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
        self.start_trial_num = start_trial_num
        self.env = env()
        self.model = model
        self.llm = llm_wrapper(model)
        self.short_memory = short_memory()
        self.local_memory = local_memory(num_envs)
        self.prompt_builder = prompt_builder()
        self.fewshot_builder = fewshot_builder()
        if self.start_trial_num > 0:
            trial_env_configs_log_path: str = os.path.join(self.logging_dir, f'env_results_trial_{self.start_trial_num-1}.json')
            self.local_memory.resume(trial_env_configs_log_path)
        self.task_names = self.env.getTaskNames()
        self.rooms = ["hallway", "greenhouse", "kitchen", "bathroom", "outside", "workshop", "art studio",
                      "foundry", "bedroom", "living room"]
        self.current_room = 'hallway'
    
    def run(self):
        world_log_path = os.path.join(self.logging_dir, 'world.log')
        for trial_idx in tqdm(range(self.start_trial_num, self.num_trials)):

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

            short_jobs = [7, 11, 12, 13, 14, 18, 20, 21, 22, 24]
            variations = [0, 1, 2, 3, 4]
            # create a list of list of intersection of short_jobs and variations, e.g., [[7, 0], [7, 1], ..., [24, 4]]
            job_params = []
            for job_id in short_jobs:
                for variation in variations:
                    job_params.append([job_id, variation])

            for env_idx in tqdm(range(self.num_envs)): # assume 50 envs

                job_id = job_params[env_idx][0]
                var_id = job_params[env_idx][1]
                task_name = self.task_names[job_id]
                self.env.load(task_name, var_id, simplificationStr='easy')
                init_ob, info = self.env.reset()
                task_description = self.env.taskdescription()[18:].strip()

                if self.local_memory.is_success(env_idx):
                    num_successes += 1
                    with open(world_log_path, 'a') as wf:
                        wf.write(f'Environment #{env_idx} Trial #{trial_idx}: SUCCESS\n')
                    with open(trial_log_path, 'a') as wf:
                        wf.write(f'\n#####\n\nEnvironment #{env_idx}: Success\n\n#####\n')
                    continue
                history_log, is_success = self.run_trajectory(env_idx, init_ob, task_description, info=info)
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
            # self.env.close()
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
            # self.env.reload()
    
    def run_trajectory(self, env_idx, init_ob, task_description, to_print=True, info=None):
        cur_step = 0
        while cur_step < self.max_steps:

            infer_prompt = self.build_infer_prompt(env_idx, init_ob, task_description)

            system_msg = """You are the agent to interact in a household to solve a task.
                                You need to output your thinking/reason/plan to solve the task, and select a correct action to execute.

                                Please using json format to output, e.g.,
                                The json output is:
                                {
                                    "reason": "To solve the task, I need to be in same location as water and have substance alone in a single container",
                                    "action": "go to the kitchen"
                                }
                                """

            response = self.llm(infer_prompt, sys_msg=system_msg, use_json=True)
            reason = response['reason']
            action = response['action']
            # print('\n\n=== GPT action begin ===\n', response, '\n\n=== GPT action end ===\n')
            action = action.replace('(', '').replace(')', '')
            # action = self.env.action_parser(action)
            self.short_memory.add("think", reason)
            self.short_memory.add("action", action)
            action = findValidActionNew([action], self.env, info['look'],
                                        recent_actions=self.short_memory.recent_actions())
            # print('\n\n=== Correct action begin ===\n', action, '\n\n=== Correct action end ===\n')
            observation, reward, done, info = self.env.step(action)
            self.short_memory.add("observation", observation)
            if action.__contains__('go to') and observation.__contains__('move to'):
                for room in self.rooms:
                    if observation.__contains__(room):
                        self.current_room = room
                        self.short_memory.add("look", info['look'])
                        break

            if done:
                score = info['score']
                history_log = self.build_infer_prompt(env_idx, init_ob, task_description, score=score)
                if score == 100:
                    return history_log, True
                else:
                    print('Failed task, id =', env_idx, ', score =', score)
                    return history_log, False

            cur_step += 1
        history_log = self.build_infer_prompt(env_idx, init_ob, task_description)
        return history_log, False
    
    def update_local_memory(self, log_str, env_idx):
        if not self.local_memory.is_success(env_idx) and not self.local_memory.is_skip(env_idx):
            reflection_prompt = self.build_reflection_prompt(log_str, env_idx)
            reflection = self.llm(reflection_prompt) 
            self.local_memory.add(env_idx, reflection)
            
    def build_infer_prompt(self, env_idx, init_ob, task_description, score=None):
        short_memories = self.short_memory.recall()
        local_memories = self.local_memory.recall(env_idx)
        if len(local_memories) > 3:
            local_memories = local_memories[-3:]
        fewshots = self.fewshot_builder.get_inference_fewshots(env_idx)
        action_guides = self.combine_action_guides()
        query = self.prompt_builder.get_inference_prompts(init_ob, fewshots, local_memories, short_memories, task_description, action_guides)
        if score:
            query += '\nFinal score: ' + str(score) + '\n'
        return query
        
    def build_reflection_prompt(self, log_str, env_idx):
        local_memories = self.local_memory.recall(env_idx)
        if len(local_memories) > 3:
            local_memories = local_memories[-3:]
        fewshots = self.fewshot_builder.get_reflection_fewshots()
        query = self.prompt_builder.get_reflection_prompts(log_str, fewshots, local_memories)
        return query

    def combine_action_guides(self):
        guides_list = action_type_description
        guides_str = []
        for guide in guides_list:
            action_type = guide.get('action_type')
            desc = guide['desc']
            guides_str.append(f'{action_type}: {desc}')
        return "Action guides:\n\t" + '\n\t'.join(guides_str)