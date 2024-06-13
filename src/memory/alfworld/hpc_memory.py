from typing import List, Dict, Any
import os
import json
from openai import OpenAI
import numpy as np
import chromadb
from chromadb.config import Settings

class ShortMemory:
    def __init__(self) -> None:
        self.history: List[Dict[str, str]] = []

    def add(self, label: str, value: str) -> None:
        assert label in ['action', 'observation']
        self.history += [{
            'label': label,
            'value': value,
        }]

    def reset(self) -> None:
        self.history = []

    def recall(self, with_think: bool = True) -> str:
        s: str = '\n'
        for i, item in enumerate(self.history):
            if item['label'] == 'action':
                if not with_think and item["value"].startswith('think:'):
                    continue
                s += f'> {item["value"]}'
            elif item['label'] == 'observation':
                s += item['value']
            if i != len(self.history) - 1:
                s += '\n'
        return s
    
class LocalMemory:
    def __init__(self, num_envs):
        self.history: List[Dict[str, Any]] = []
        for i in range(num_envs):
             self.history += [{
                'name': f'env_{i}',
                'memory': [],
                'is_success': False,
                'skip': False
            }]
    
    def set_success(self, idx: int):
        self.history[idx]['is_success'] = True
        
    def is_success(self, idx: int):
        return self.history[idx]['is_success']
    
    def is_skip(self, idx: int):
        return self.history[idx]['skip']
        
    def add(self, idx, input_dict):
        for key in input_dict:
            if key == 'memory':
                self.history[idx][key] += [input_dict[key]]
            else:
                self.history[idx][key] = input_dict[key]
    
    def recall(self, idx: int):
        return self.history[idx]['memory']

class GlobalMemory:
    def __init__(self, logging_dir, is_vector, env_batch_size = 3, task_batch_size = 5):
        self.env_memory = dict()
        self.task_memory = dict()
        self.env_bs = env_batch_size
        self.task_bs = task_batch_size
        self.logging_dir = logging_dir
        self.is_vector = is_vector
        if self.is_vector:
            self.task_db = dict()
            self.db = Vectorizor()
    
    def short2long(self, expert_trajectory, env_idx, trial_idx): 
        increment_env, increment_task = {}, {}
        env_description = expert_trajectory['env']
        task_description = expert_trajectory['task']
        task_type = self._convert_task_description(task_description)
        status = 'success' if expert_trajectory['is_success'] else 'fail'
        retrieve_idx = dict(env_idx=env_idx, trial_idx=trial_idx)
        # 更新env_memory
        # 判断是否属于好奇心
        env_curiocity = task_curiocity = False
        if env_description not in self.env_memory:
            self.env_memory[env_description] = {'known_obs':'', 
                                                'increment_traj':[],
                                                'all_traj':[]
                                                }
            env_curiocity = True
        self.env_memory[env_description]['increment_traj'].append(retrieve_idx)
        self.env_memory[env_description]['all_traj'].append(retrieve_idx)
        # 属于好奇心或重复，取出增量记忆进行反思  
        if env_curiocity or len(self.env_memory[env_description]['increment_traj']) > self.env_bs:
            samples = self._get_samples(self.env_memory[env_description]['increment_traj'])
            increment_known_obs = [sample['known_obs'] for sample in samples]
            increment_env = dict(known_obs=self.env_memory[env_description]['known_obs'],
                                    increment_known_obs=increment_known_obs)
            self.env_memory[env_description]['increment_traj'] = []
        # 更新task_memory
        # 判断是否属于好奇心
        if task_type not in self.task_memory:
            self.task_memory[task_type] = {}
            if self.is_vector:
                self.task_db[task_type] = {}
        if status not in self.task_memory[task_type]:
            self.task_memory[task_type][status] = {
                'action_guidance': '',
                'increment_traj': [],
                'all_traj': []
            }
            if self.is_vector:
                self.task_db[task_type][status] = self.db.create_collection(name=task_type + '_' + status)
            task_curiocity = True
        self.task_memory[task_type][status]['increment_traj'].append(retrieve_idx)
        self.task_memory[task_type][status]['all_traj'].append(retrieve_idx)

        # 属于好奇心或重复，取出增量记忆进行反思  
        if task_curiocity or len(self.task_memory[task_type][status]['increment_traj']) > self.task_bs:
            samples = self._get_samples(self.task_memory[task_type][status]['increment_traj'])
            increment_action_guidance = [(dict(
                                                task=sample['task'],
                                                my_actions=sample["my_actions"],
                                                is_success=sample["is_success"],
                                                reflection=sample["memory"][-1] if len(sample["memory"]) > 0 else '', 
                                                )
                                        ) for sample in samples]
            increment_task = dict(task_type=task_type,
                                    action_guidance=self.task_memory[task_type][status]['action_guidance'],
                                    increment_action_guidance=increment_action_guidance
                                    )
        # 计算increment的embedding
            if self.is_vector:
                samples = self._get_samples(self.task_memory[task_type][status]['increment_traj'])
                ids = [str(traj['trial_idx']) + '_' + str(traj['env_idx']) for traj in self.task_memory[task_type][status]['increment_traj']]
                sample_reflections = [sample["memory"][-1] for sample in samples]
                sample_reflection_embeddings = [self.db.get_embedding(reflection) for reflection in sample_reflections]
                self.task_db[task_type][status].add(embeddings=sample_reflection_embeddings,ids=ids)
            self.task_memory[task_type][status]['increment_traj'] = []
        return increment_env, increment_task
            
    def _convert_task_description(self, task_description):
        if "put" in task_description:
            if "heat" in task_description or "hot" in task_description:
                return "pick_heat_then_place"
            elif "clean" in task_description:
                return "pick_clean_then_place"
            elif "cool" in task_description:
                return "pick_cool_then_place"
            elif "two" in task_description:
                return "pick_two_obj"
            else:
                return "pick_and_place"
        elif "find two" in task_description:
            return "pick_two_obj"
        elif "examine" in task_description or "look at" in task_description:
            return "look_at_obj"
        else:
            raise ValueError(f"Unseen type: {task_description}")
    
    def add(self, summary, expert_trajectory, mode):
        env_description = expert_trajectory['env']
        task_description = expert_trajectory['task']
        task_type = self._convert_task_description(task_description)
        status = 'success' if expert_trajectory['is_success'] else 'fail'
        if mode == 'env':
            self.env_memory[env_description]['known_obs'] = summary
        elif mode == 'task':
            self.task_memory[task_type][status]['action_guidance'] = summary
        else:
            raise ValueError(f"Unseen mode: {mode}")
        
    def recall(self, env_description, task_description):
        env_recall = task_recall = ''
        if env_description:
            if env_description in self.env_memory:
                env_recall = self.env_memory[env_description]['known_obs']
        if task_description:
            task_type = self._convert_task_description(task_description)
            if task_type in self.task_memory:
                item_idx = 1
                if 'success' in self.task_memory[task_type]:        
                    split_summary = self._split_summary(self.task_memory[task_type]['success']['action_guidance'])
                    if self.is_vector:
                        repeat_scores = []
                        collection = self.task_db[task_type]['success']
                        for summary_item in split_summary:
                            summary_item_embedding = self.db.get_embedding(summary_item)
                            results = collection.query(query_embeddings=summary_item_embedding, n_results=collection.count())
                            repeat_score = 1 / sum(results['distances'][0])
                            repeat_scores.append(repeat_score)
                    for i in range(len(split_summary)):
                        if self.is_vector:
                            task_recall += f"{item_idx}. {split_summary[i]} {round(repeat_scores[i],2)}\n"
                        else:
                            task_recall += f"{item_idx}. {split_summary[i]}\n"
                        item_idx += 1
                if 'fail' in self.task_memory[task_type]:
                    split_summary = self._split_summary(self.task_memory[task_type]['fail']['action_guidance'])
                    if self.is_vector:
                        repeat_scores = []
                        collection = self.task_db[task_type]['fail']
                        for summary_item in split_summary:
                            summary_item_embedding = self.db.get_embedding(summary_item)
                            results = collection.query(query_embeddings=summary_item_embedding, n_results=collection.count())
                            repeat_score = 1 / sum(results['distances'][0])
                            repeat_scores.append(repeat_score)
                    for i in range(len(split_summary)):
                        if self.is_vector:
                            task_recall += f"{item_idx}. {split_summary[i]} {round(repeat_scores[i],2)}\n"
                        else:
                            task_recall += f"{item_idx}. {split_summary[i]}\n"
                        item_idx += 1
        return env_recall , task_recall
    
    def _get_samples(self, trajs):
        samples = []
        for retrieve_idx in trajs:
            sample_trial_idx = retrieve_idx['trial_idx']
            sample_env_idx = retrieve_idx['env_idx']
            sample_path = os.path.join(self.logging_dir, f'local_memory_trial_{sample_trial_idx}.json')
            with open(sample_path, 'r') as f:
                sample_list = json.load(f)
            samples.append(sample_list[sample_env_idx])
        return samples

    def _split_summary(self, summary):
        lines = summary.split('\n')
        result = [line.split('. ', 1)[1] for line in lines if len(line.split('. ')) > 1]
        return result
        
class Vectorizor:
    def __init__(self):
        self.embed_client = OpenAI(
        base_url=os.getenv('OPENAI_API_BASE_URL') if 'OPENAI_API_BASE_URL' in os.environ else None,
        api_key=os.getenv('OPENAI_API_KEY'),
        )
        self.chroma_client = chromadb.Client(settings=Settings(allow_reset=True))
        self.chroma_client.reset()
    
    def get_embedding(self, text, model="text-embedding-3-small"):
        return self.embed_client.embeddings.create(input=text, model=model).data[0].embedding
    
    def create_collection(self, name):
        collection = self.chroma_client.create_collection(name)
        return collection