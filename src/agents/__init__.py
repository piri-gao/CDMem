from .alfworld import ReflectAgent, CDMemAgent, AutoguideAgent, ExpelAgent, ReactAgent

AGENT = dict(alfworld=dict(reflect=ReflectAgent, cdmem=CDMemAgent, autoguide=AutoguideAgent, expel=ExpelAgent, react=ReactAgent))