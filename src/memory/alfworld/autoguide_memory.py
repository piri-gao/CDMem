from typing import List, Dict, Any
import json


class ShortMemory:
    def __init__(self) -> None:
        self.history: List[Dict[str, str]] = []

    def add(self, label: str, value: str) -> None:
        assert label in ['action', 'observation','status_summary']
        self.history += [{
            'label': label,
            'value': value,
        }]

    def reset(self) -> None:
        self.history = []

    def recall(self, with_think: bool = True, with_status_summary: bool = True) -> str:
        s: str = '\n'
        for i, item in enumerate(self.history):
            if item['label'] == 'action':
                if not with_think and item["value"].startswith('think:'):
                    continue
                s += f'> {item["value"]}'
            elif item['label'] == 'observation':
                s += item['value']
            elif item['label'] == 'status_summary' and with_status_summary:
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
        
    def add(self, idx: int, reflection: str):
        self.history[idx]['memory'] += [reflection]
    
    def recall(self, idx: int):
        return self.history[idx]['memory']
    
    def resume(self, resume_path):
        with open(resume_path, 'r') as f:
            sample_list = json.load(f)
        self.history = sample_list
        
class GlobalMemory:
    def __init__(self, logging_dir):
        self.history = {}
    
    def add(self, status, guideline):
        self.history.setdefault(status, []).append(guideline)

    def get_status_list(self):
        return list(self.history.keys())
    
    def recall(self, status):
        return self.history[status]
    
        