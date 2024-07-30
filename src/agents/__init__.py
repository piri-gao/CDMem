from .alfworld import ReflectAgent, HPCAgent
from .webshop import ReflectAgentWS, HPCAgentWS

AGENT = dict(alfworld=dict(reflect=ReflectAgent, hpc=HPCAgent), webshop=dict(reflect=ReflectAgentWS, hpc=HPCAgentWS))