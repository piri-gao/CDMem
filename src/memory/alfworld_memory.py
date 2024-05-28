from typing import List, Dict, Any


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
        
    def add(self, idx: int, reflection: str):
        self.history[idx]['memory'] += [reflection]
    
    def recall(self, idx: int):
        return self.history[idx]['memory']