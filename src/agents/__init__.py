from .alfworld import ReflectAgent, HPCAgent, AutoguideAgent, ExpelAgent

AGENT = dict(alfworld=dict(reflect=ReflectAgent, hpc=HPCAgent, autoguide=AutoguideAgent, expel=ExpelAgent))