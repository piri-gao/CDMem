from .alfworld import ReflectAgent, HPCAgent, AutoguideAgent, ExpelAgent, ReactAgent

AGENT = dict(alfworld=dict(reflect=ReflectAgent, hpc=HPCAgent, autoguide=AutoguideAgent, expel=ExpelAgent, react=ReactAgent))