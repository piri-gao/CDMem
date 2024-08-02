from .alfworld import ReflectAgent, HPCAgent
from .webshop.hpc import HPCAgentWS
from .webshop.reflect import ReflectAgentWS

AGENT = dict(alfworld=dict(reflect=ReflectAgent, hpc=HPCAgent), webshop=dict(reflect=ReflectAgentWS, hpc=HPCAgentWS))