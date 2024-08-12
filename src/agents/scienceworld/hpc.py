import sys
import os
import json
import re
import itertools

from typing import List, Callable

from tqdm import tqdm

from src.utils.scworld_utils import findValidActionNew, action_type_description


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
        if 'is_vector' in kwargs:
            self.is_vector = kwargs['is_vector']
        self.global_memory = global_memory(logging_dir, self.is_vector)
        self.prompt_builder = prompt_builder()
        self.fewshot_builder = fewshot_builder()
        self.logger = Logger(self.logging_dir, self.num_trials, self.num_envs, self.start_trial_num, self.local_memory,
                             self.global_memory)
        self.task_names = self.env.getTaskNames()
        self.rooms = ["hallway", "greenhouse", "kitchen", "bathroom", "outside", "workshop", "art studio",
                      "foundry", "bedroom", "living room"]
        self.current_room = ''

    def run(self):
        for trial_idx in range(self.start_trial_num, self.num_trials):
            self.logger.log_world_start(trial_idx)
            num_successes: int = 0
            num_additional_successes: int = 0
            short_jobs = [7, 11, 12, 13, 14, 18, 20, 21, 22, 24]
            variations = [0, 1, 2, 3, 4]
            # create a list of list of intersection of short_jobs and variations, e.g., [[7, 0], [7, 1], ..., [24, 4]]
            job_params = []
            for job_id in short_jobs:
                for variation in variations:
                    job_params.append([job_id, variation])

            for env_idx in tqdm(range(self.num_envs)):  # consider 50 envs
                job_id = job_params[env_idx][0]
                var_id = job_params[env_idx][1]
                task_name = self.task_names[job_id]
                self.env.load(task_name, var_id, simplificationStr='easy')
                _, info = self.env.reset()
                task_description = self.env.taskdescription()[18:].strip()
                init_ob = info['look']
                # print()
                # print('=== Task ===\n', task_description)
                # print('=== Init Observation ===\n', init_ob)
                # for token in init_ob.split(' '):
                #     token = token.replace('green house', 'greenhouse')
                #     if token in self.rooms:
                #         self.current_room = token
                #         break
                for room in self.rooms:
                    if room in init_ob.split('\n')[0]:
                        # print('new room detect:', room)
                        self.current_room = room
                        break
                # print('=== Info ===\n', info)
                # print()
                # print(f"{env_idx} using {self.env.env_name}")
                if self.local_memory.is_success(env_idx):
                    num_successes += 1
                    self.logger.log_world_success(trial_idx, env_idx)
                    self.logger.log_trial_success(trial_idx, env_idx)
                    continue
                history_log, is_success = self.run_trajectory(env_idx, init_ob, task_description, info=info)
                if is_success:
                    self.logger.log_world_success(trial_idx, env_idx)
                    self.local_memory.set_success(env_idx)
                    num_successes += 1
                    num_additional_successes += 1
                else:
                    self.logger.log_world_fail(trial_idx, env_idx)
                self.logger.log_trial_content(history_log, is_success, trial_idx, env_idx)
                expert_trajectory = self.update_local_memory(history_log, is_success, env_idx, task_description)
                self.logger.log_local_memory(trial_idx)
                self.update_global_memory(expert_trajectory, env_idx, trial_idx)
                self.logger.log_global_memory(trial_idx)
            # self.env.close()
            # self.logger.log_trial_end(trial_idx, num_successes, num_additional_successes)
            self.logger.log_world_end(trial_idx, num_successes, num_additional_successes)
            # self.env.reload()

    def run_trajectory(self, env_idx, init_ob, task_description, to_print=True, info=None):
        cur_step = 0
        # print(init_ob)
        score = 0
        self.short_memory.reset()
        while cur_step < self.max_steps:
            infer_prompt = self.build_infer_prompt(env_idx, init_ob, task_description)

            # print('\n\n=== Infer prompt begin ===\n', infer_prompt, '\n\n=== Infer prompt end===\n')

            system_msg = """You are the agent to interact in a household to solve a task.
                    This is a big house with following rooms:
                    ["hallway", "greenhouse", "kitchen", "bathroom", "outside", "workshop", "art studio", "foundry", "bedroom", "living room"]
                    
                    These rooms are connected by doors, and you can move to a room by saying "go to the room" when door is open.
                    "hallway" has doors to ["kitchen", "bathroom", "bedroom", "workshop", "art studio", "living room", "greenhouse"].
                    "kitchen" has a door to "outside", "greenhouse" has a door to "outside" and a door to "hallway"
                    "foundry" has a door to "outside" only.
                    
                    Please Note:
                    You need to output your thinking/reason/plan to solve the task, and select a correct action to execute.
                    Please read the current task description very carefully and never misunderstand your task. Your thinking should strictly follows the current task.
                    You need to go to the target room first before you manage some object.
                    And you should clearly know your current room (where you are) and the target room (where to go), try to arrive the target location first.
                    
                    You selected action will be executed in the environment. Please be carefully design the correct action command.  
                    Here are some action guides:
                    
                    action_type_description = [
                        {"action_type": "wait",
                         "desc": "wait for something to be done, for example, an object on stove to be boiled. Usage: 'wait#', where # is the number of turns you want to wait. only 'wait' means wait for 10 iterations."},
                        {"action_type": "read", "desc": "read an object such as a recipe or a book. Usage: 'read recipe in inventory'"},
                        {"action_type": "pick up", "desc": "pick up an object and put it into your inventory. Usage: 'pick up metal pot'"},
                        {"action_type": "open",
                         "desc": "open an object with doors before you search or put things in it. Usage: 'open door in kitchen', 'open drawer in counter', 'open glass jar'"},
                        {"action_type": "activate",
                         "desc": "activate and turn on an object such as sink (then the water flow from it) or stove, so that you can use it. Usage: 'activate stove', 'activate sink'"},
                        {"action_type": "deactivate", "desc": "deactivate turn off the object"},
                        {"action_type": "examine",
                         "desc": "look at an object carefully. Note that you cannot examine a location. You can only 'examine something' or 'examine substance in something' Usage: 'examine substance in metal pot', 'examine ice'"},
                        {"action_type": "move", "desc": "move/place the object to a place. Usage: 'move cupboard to red box'"},
                        {"action_type": "use",
                         "desc": "use an object A on object B, for example, For example, to check the temperature: Usage: 'use thermometer in inventory on ice', 'use thermometer in inventory on substance in metal pot'"},
                        {"action_type": "pour",
                         "desc": "pour the object A into the container B. Usage: 'pour jug into flower pot 4'"},
                        {"action_type": "focus",
                         "desc": "focus on an important object that are required by the task description (e.g., a substance, a plant, an animal, and so on). Usage: 'focus on cupboard'"},
                    ]
                    
                    Please note, interactive trajectory is realtime feedback from environment. You are required to interact with the environment to complete the task.
                    So, you need to output your thinking and an action, and the action will be executed in the environment.
                    But if your action is invalid, you will receive two types of feedback:
                    1. 'No known action matches that input.' means the environment can not execute the command. It may be due to an inability to reach the target position or a syntax error. Please rethink and output the correct action.
                    2. 'Ambiguous request: Please enter the number for the action you intended (or blank to cancel): <followed by some options with their index numbers. Format is #: xxx>', In this situation, you need to choose the correct action by entering the number of the option index you intended. This means your action should be a number. An example of an option: '0: move apple seed'
                    
                    Please using json format to output, e.g.,
                    
                    The json output is:
                    {
                        "reason": "To solve the task, I need to be in same location as water and have substance alone in a single container",
                        "action": "go to the kitchen"
                    }
                    
                    or:
                    {
                        "reason": "To solve the task, I need to..."
                        "action": "0"
                    }
                    """

            response = self.llm(infer_prompt, sys_msg=system_msg, use_json=True)
            reason = response['reason']
            action = response['action'].replace(' the ', ' ').replace('  ', ' ')
            # print('\n\n=== GPT action begin ===\n', response, '\n\n=== GPT action end ===\n')
            action = action.replace('(', '').replace(')', '')
            # action = self.env.action_parser(action)
            self.short_memory.add("think", reason)
            self.short_memory.add("action", action)
            action = findValidActionNew([action], self.env, info['look'], recent_actions=self.short_memory.recent_actions())
            # print('\n\n=== Correct action begin ===\n', action, '\n\n=== Correct action end ===\n')
            observation, reward, done, info = self.env.step(action)
            if info['score']:
                score = info['score']
            self.short_memory.add("observation", observation)
            if action.__contains__('go to') and observation.__contains__('move to'):
                for room in self.rooms:
                    if observation.__contains__(room):
                        # print('new room detect:', room)
                        self.current_room = room
                        # self.short_memory.add("look", info['look'])
                        init_ob = info['look']
                        break
            # print('\n\n=== Observation begin ===\n', observation, '\n\n=== Observation end ===\n')
            # print('\n\n=== Look begin ===\n', info['look'], '\n\n=== Look end ===\n')

            # if to_print:
            #     print(f'> {action}\n{observation}')
            #     sys.stdout.flush()
            if done:
                history_log = self.build_infer_prompt(env_idx, init_ob, task_description, score=score)
                if score == 100:
                    return history_log, True
                else:
                    # print('Failed task, id =', env_idx, ', score =', score)
                    return history_log, False

            cur_step += 1
        history_log = self.build_infer_prompt(env_idx, init_ob, task_description, score)
        return history_log, False

    def update_local_memory(self, history_log, is_success, env_idx, task_description):
        if not self.local_memory.is_skip(env_idx):
            expert_prompt = self.build_expert_prompt(history_log)
            expert_result = self.llm(expert_prompt, max_tokens=512)
            reflection_prompt = self.build_reflection_prompt(history_log, is_success, expert_result, env_idx)
            reflection_result = self.llm(reflection_prompt, max_tokens=512)
            expert_trajectory = self.process_after_reflection(expert_result, reflection_result, history_log, is_success, task_description)
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

    def build_infer_prompt(self, env_idx, init_ob, task_description, score=None):
        short_memories = self.short_memory.recall()
        local_memories = self.local_memory.recall(env_idx)
        if len(local_memories) > 3:
            local_memories = local_memories[-3:]
        env_description = self.process_before_infer(init_ob)
        known_obs_history, action_guidance_history = self.global_memory.recall(env_description, task_description)
        fewshots = self.fewshot_builder.get_inference_fewshots(env_idx, env_description, task_description,
                                                               self.global_memory, self.logging_dir)
        # action_guides = self.combine_action_guides()
        query = self.prompt_builder.get_inference_prompts(init_ob, fewshots, local_memories, short_memories,
                                                          known_obs_history, action_guidance_history, task_description, self.current_room)

        if score:
            query += '\nFinal score: ' + str(score) + '\n'
        return query

    def build_expert_prompt(self, history_log):
        fewshots = self.fewshot_builder.get_expert_fewshots()
        query = self.prompt_builder.get_expert_prompts(history_log, fewshots)
        return query

    def build_reflection_prompt(self, history_log, is_success, expert_result, env_idx):
        local_memories = self.local_memory.recall(env_idx)
        if len(local_memories) > 3:
            local_memories = local_memories[-3:]
        fewshots = self.fewshot_builder.get_reflection_fewshots(is_success)
        query = self.prompt_builder.get_reflection_prompts(history_log, is_success, fewshots, local_memories,
                                                           expert_result)
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

    def process_after_reflection(self, expert_result, reflection_result, history_log, is_success, task_description):
        scenario = history_log.split("Here is your real task:")[-1].strip()
        env_pattern = r'This room is called the\..*?(?=\n)'
        # task_pattern = r'Your task is to: (.*?)(?=\n)'
        location_pattern = r'\(1\)locations:(.*?)\(2\)functions:'
        function_pattern = r'\(2\)functions:(.*?)Expert Actions:'
        action_pattern = r'Expert Actions:(.*)'
        reflection_pattern = r'Reflection: (.*?)(?:\n|$)'

        env_description = ''
        location = function = action = reflection = ''

        env_match = re.search(env_pattern, scenario, re.DOTALL)
        if env_match:
            env_description = env_match.group(0).strip()

        # task_match = re.search(task_pattern, scenario, re.DOTALL)
        # if task_match:
        #     task_description = task_match.group(1).strip()

        location_match = re.search(location_pattern, expert_result, re.DOTALL)
        if location_match:
            location = location_match.group(1).strip()

        function_match = re.search(function_pattern, expert_result, re.DOTALL)
        if function_match:
            function = function_match.group(1).strip()

        action_match = re.search(action_pattern, expert_result, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()

        reflection_match = re.search(reflection_pattern, reflection_result, re.DOTALL)
        if reflection_match:
            reflection = reflection_match.group(1).strip()

        expert_trajectory = dict(env=env_description,
                                 task=task_description,
                                 location=location,
                                 function=function,
                                 action=action,
                                 reflection=reflection,
                                 is_success=is_success
                                 )
        return expert_trajectory

    def process_before_infer(self, init_ob):
        # env_description = task_description = init_ob
        # # return env_description, task_description
        # env_pattern = r'You are in the middle of a room\..*?(?=\n)'
        # task_pattern = r'Your task is to:\s*(.*)'
        env_description = ''
        first_line = init_ob.split('\n')[0]
        first_line = first_line.replace("green house", "greenhouse")
        for room in self.rooms:
            if first_line.__contains__(room):
                env_description = room
        # task_match = re.search(task_pattern, init_ob, re.DOTALL)
        # if task_match:
        #     task_description = task_match.group(1).strip()

        return env_description

    def combine_action_guides(self):
        guides_list = action_type_description
        guides_str = []
        for guide in guides_list:
            action_type = guide.get('action_type')
            desc = guide['desc']
            guides_str.append(f'{action_type}: {desc}')
        return "Action guides:\n\t" + '\n\t'.join(guides_str)


class Logger:
    def __init__(self, logging_dir, num_trials, num_envs, start_trial_num, local_memory, global_memory):
        self.logging_dir = logging_dir
        self.num_trials = num_trials
        self.num_envs = num_envs
        self.local_memory = local_memory
        self.global_memory = global_memory
        self.world_log_path = os.path.join(self.logging_dir, 'world.log')
        self.trial_log_paths = [os.path.join(self.logging_dir, f'trial_{trial_idx}.log') for trial_idx in
                                range(self.num_trials)]
        self.local_memory_paths = [os.path.join(self.logging_dir, f'local_memory_trial_{trial_idx}.json') for trial_idx
                                   in range(self.num_trials)]
        self.global_env_paths = [os.path.join(self.logging_dir, f'global_env_trial_{trial_idx}.json') for trial_idx in
                                 range(self.num_trials)]
        self.global_task_paths = [os.path.join(self.logging_dir, f'global_task_trial_{trial_idx}.json') for trial_idx in
                                  range(self.num_trials)]
        for path in list(
                itertools.chain(self.trial_log_paths[start_trial_num:], self.local_memory_paths[start_trial_num:],
                                self.global_env_paths[start_trial_num:], self.global_task_paths[start_trial_num:])):
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
            wf.write(
                f'\n#####\n\nEnvironment #{env_idx}:\n{content}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

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
