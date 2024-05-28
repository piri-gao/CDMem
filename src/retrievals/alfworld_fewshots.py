import os
import json

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

class AlfworldFewshotBuilder:
    def __init__(self):
        pass
    
    def get_inference_fewshots(self, name):
        for i, (k, v) in enumerate(PREFIXES.items()):
            if name.startswith(k):
                return d[f'react_{v}_1'] + d[f'react_{v}_0']
                
    def get_reflection_fewshots(self):
        with open("./prompts/reflexion_few_shot_examples.txt", 'r') as f:
            FEW_SHOT_EXAMPLES = f.read()
        return FEW_SHOT_EXAMPLES