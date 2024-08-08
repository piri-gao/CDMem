# from .alfworld import ReflectAgent, HPCAgent
# from .webshop.hpc import HPCAgentWS
# from .webshop.reflect import ReflectAgentWS
from .scienceworld.hpc import HPCAgent
from .scienceworld.reflect import ReflectAgent
# AGENT = dict(alfworld=dict(reflect=ReflectAgent, hpc=HPCAgent), webshop=dict(reflect=ReflectAgentWS, hpc=HPCAgentWS))
AGENT = dict(scienceworld=dict(reflect=ReflectAgent, hpc=HPCAgent))
