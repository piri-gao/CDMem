import re

text = '''
KNOWN OBS: Shelf 7 contains a statue 3, shelf 8 contains a keychain 2, shelf 10 contains a creditcard 3, cabinet 3 contains a box 1 and a tissuebox 1.
MY ACTIONS: I checked shelves 1-12, then opened cabinet 1 and cabinet 3.
REFLECTION: I spent too much time checking empty shelves and cabinets. I should have focused on the most likely places to find a vase, such as a shelf or a cabinet with decorative items. I will prioritize checking those areas first in the future to complete the task more efficiently.
'''

# 定义用于提取每一部分的正则表达式模式
known_obs_pattern = r'KNOWN OBS: (.*?)(?:\n|$)'
my_actions_pattern = r'MY ACTIONS: (.*?)(?:\n|$)'
reflection_pattern = r'REFLECTION: (.*?)(?:\n|$)'

# 提取 KNOWN OBS 部分
known_obs_match = re.search(known_obs_pattern, text, re.DOTALL)
if known_obs_match:
    known_obs = known_obs_match.group(1).strip()
    print("KNOWN OBS:", known_obs)

# 提取 MY ACTIONS 部分
my_actions_match = re.search(my_actions_pattern, text, re.DOTALL)
if my_actions_match:
    my_actions = my_actions_match.group(1).strip()
    print("MY ACTIONS:", my_actions)

# 提取 REFLECTION 部分
reflection_match = re.search(reflection_pattern, text, re.DOTALL)
if reflection_match:
    reflection = reflection_match.group(1).strip()
    print("REFLECTION:", reflection)
