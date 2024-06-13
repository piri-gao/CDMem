import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

# load config
config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval="eval_out_of_distribution")
env = env.init_env(batch_size=1)

# interact
obs, info = env.reset()
# while True:
#     # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
#     admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
#     random_actions = [np.random.choice(admissible_commands[0])]

#     # step
#     obs, scores, dones, infos = env.step(random_actions)
#     print("Action: {}, Obs: {}".format(random_actions[0], obs[0]))

dd = {}
tt = {}
for i in range(134):
    obs, info = env.reset()
    name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
    for i, (k, v) in enumerate(PREFIXES.items()):
        if name.startswith(k):
            break
    task = obs[0].split("Your task is to: ")[-1]
    print(f"{k}:{task}")
    if k not in dd:
        dd[k] = []
    else:
        dd[k].append(task)
print(dd)