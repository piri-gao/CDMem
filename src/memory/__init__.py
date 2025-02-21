from .alfworld import ReflectShortMemory, ReflectLocalMemory, CDMemShortMemory, CDMemLocalMemory, CDMemGlobalMemory
from .alfworld import AutoguideShortMemory, AutoguideLocalMemory, AutoguideGlobalMemory
from .alfworld import ExpelShortMemory, ExpelLocalMemory, ExpelGlobalMemory

SHORT_MEMORY = dict(alfworld=dict(reflect=ReflectShortMemory, cdmem=CDMemShortMemory, autoguide=AutoguideShortMemory, expel=ExpelShortMemory, react=ReflectShortMemory))
LOCAL_MEMORY = dict(alfworld=dict(reflect=ReflectLocalMemory, cdmem=CDMemLocalMemory, autoguide=AutoguideLocalMemory, expel=ExpelLocalMemory, react=ReflectLocalMemory))
GLOBAL_MEMORY = dict(alfworld=dict(cdmem=CDMemGlobalMemory, reflect=CDMemGlobalMemory, autoguide=AutoguideGlobalMemory, expel=ExpelGlobalMemory, react=CDMemGlobalMemory))