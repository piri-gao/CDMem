import yaml
import importlib
import alfworld
import alfworld.agents.environment

class AlfworldEnv:
    def __init__(self):
        importlib.reload(alfworld)
        importlib.reload(alfworld.agents.environment)
        with open('data/alfworld/base_config.yaml') as reader:
            config = yaml.safe_load(reader)
        split = "eval_out_of_distribution"
        self.env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
        self.env = self.env.init_env(batch_size=1)
        self.last_action = None
        
    def step(self, action):
        observation, reward, done, info = self.env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        exhausted = False
        if self.last_action == action:
            exhausted = True
        else:
            self.last_action = action
        return observation, reward, done, exhausted, info
    
    def reset(self):
        self.last_action = None
        ob, info = self.env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        self.name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        return ob, info
    
    def reload(self):
        self.__init__()
    
    def close(self):
        self.env.close()


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob