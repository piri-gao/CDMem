import os
import json

FOLDER = './prompts'
PROMPT_FILE = 'science_world_prompts.jsonl'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r', encoding='utf-8') as f:
    d = json.load(f)

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

class ReflectFewshotBuilder:
    def __init__(self):
        pass
    
    def get_inference_fewshots(self, name):
        map_list = [7, 11, 12, 13, 14, 18, 20, 21, 22, 24]
        full_map_list = []
        for item in map_list:
            for i in range(5):
                full_map_list.append(item)
        name_to_map = full_map_list[name]
        return d[str(name_to_map)]
                
    def get_reflection_fewshots(self):
        with open("prompts/reflexion_few_shot_examples.txt", 'r') as f:
            FEW_SHOT_EXAMPLES = f.read()
        return FEW_SHOT_EXAMPLES