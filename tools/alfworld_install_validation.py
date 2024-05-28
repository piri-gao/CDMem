import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

# load config
config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval="eval_out_of_distribution")
env = env.init_env(batch_size=1)

# interact
obs, info = env.reset()
# print(obs)
# print(info)
# num = 0
# while True:
#     new_obs , new_info = env.reset()
#     print(new_obs)
#     print(new_info)
#     if new_obs == obs:
#         break 
#     else:
#         num += 1
#     print(num)
# print(new_obs)
# print(new_info)
while True:
    # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
    admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
    random_actions = [np.random.choice(admissible_commands[0])]

    # step
    obs, scores, dones, infos = env.step(random_actions)
    print("Action: {}, Obs: {}".format(random_actions[0], obs[0]))