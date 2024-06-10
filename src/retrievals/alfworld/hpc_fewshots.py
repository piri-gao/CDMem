import os
import json
import random
import copy

FOLDER = './prompts'
PROMPT_FILE = 'alfworld_3prompts.json'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

class HPCFewshotBuilder:
    def __init__(self):
        pass
    
    def get_inference_fewshots(self, name, env_description , task_description, global_memory, logging_dir):
        num_examples = 2
        example_ids = []
        task_memory = global_memory.task_memory
        env_memory = global_memory.env_memory
        task_type = self._convert_task_description(task_description)
        if task_type in task_memory and 'success' in task_memory[task_type]:
            task_all_traj = copy.deepcopy(task_memory[task_type]['success']['all_traj'])
            task_all_traj = [tuple(traj.values()) for traj in task_all_traj]
            if env_description in env_memory:
                env_all_traj = copy.deepcopy(env_memory[env_description]['all_traj'])
                env_all_traj = [tuple(traj.values()) for traj in env_all_traj]
                intersection = list(set(task_all_traj) & set(env_all_traj))
                if len(intersection) >= num_examples:
                    choose_traj = random.sample(intersection, num_examples)
                    example_ids.extend(choose_traj)
                    task_all_traj = [traj for traj in task_all_traj if traj not in choose_traj]
                    num_examples = 0
                else:
                    example_ids.extend(intersection)
                    task_all_traj = [traj for traj in task_all_traj if traj not in intersection]
                    num_examples -= len(intersection)
            if num_examples > 0:
                if len(task_all_traj) >= num_examples:
                    choose_traj = random.sample(task_all_traj, num_examples)
                    example_ids.extend(choose_traj)
                    task_all_traj = [traj for traj in task_all_traj if traj not in choose_traj]
                    num_examples = 0
                else:
                    example_ids.extend(task_all_traj)
                    task_all_traj = []
                    num_examples -= len(task_all_traj)
        examples = []
        if len(example_ids) > 0:
            for example_idx in example_ids:
                examples.append(self._ids2example(logging_dir, example_idx))
        if num_examples > 0:
            default_examples = self._default_inference_fewshots(name)
            default_examples = random.sample(default_examples, num_examples)
            default_examples = ''.join(default_examples)
            examples.append(default_examples)
        return '\n'.join(examples)
                         
    def _ids2example(self, logging_dir, example_idx):
        env_idx, trial_idx = example_idx
        trial_path = os.path.join(logging_dir, f'trial_{trial_idx}.log')
        with open(trial_path, 'r') as f:
            trial_log = f.read()
        example_log = trial_log.split('#####\n\n#####')[env_idx].split("Here is the task:")[-1].strip().replace("\n\nSTATUS: OK\n\n#####", '')
        return example_log

    def _default_inference_fewshots(self, name):
        for i, (k, v) in enumerate(PREFIXES.items()):
            if name.startswith(k):
                return [d[f'react_{v}_1'] , d[f'react_{v}_0']]
                
    def get_reflection_fewshots(self, is_success):
        with open("./prompts/expert_few_shot_example.txt", 'r') as f:
            FEW_SHOT_EXAMPLES = f.read()
        examples = FEW_SHOT_EXAMPLES.split('\n\n\n')
        if is_success:
            return examples[:1]
        else:
            return '\n\n\n'.join(examples[2:])
    
    def get_summary_fewshots(self, mode, is_success=None):
        if mode == 'env':
            with open("./prompts/env_few_shot_example.txt", 'r') as f:
                FEW_SHOT_EXAMPLES = f.read()
            return FEW_SHOT_EXAMPLES
        elif mode == 'task':
            if is_success:
                with open("./prompts/task_few_shot_example_success.txt", 'r') as f:
                    FEW_SHOT_EXAMPLES = f.read()
                return FEW_SHOT_EXAMPLES
            else:
                with open("./prompts/task_few_shot_example_fail.txt", 'r') as f:
                    FEW_SHOT_EXAMPLES = f.read()
                return FEW_SHOT_EXAMPLES
        else:
            raise ValueError(f"Unseen mode type: {mode}")
        
    def _convert_task_description(self, task_description):
        if task_description.startswith("look"):
            return "look"
        elif task_description.startswith("put"):
            return "put"
        elif task_description.startswith("examine"):
            return "examine"
        elif task_description.startswith("cool") and "put" in task_description:
            return "cool_put"
        elif task_description.startswith("clean") and "put" in task_description:
            return "clean_put"
        elif task_description.startswith("heat") and "put" in task_description:
            return "heat_put"
        elif task_description.startswith("find") and "put" in task_description:
            return "find_put"
        else:
            raise ValueError(f"Unseen type: {task_description}")
            