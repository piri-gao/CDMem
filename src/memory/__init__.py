from .alfworld import ReflectShortMemory, ReflectLocalMemory, HPCShortMemory, HPCLocalMemory, HPCGlobalMemory
from .alfworld import AutoguideShortMemory, AutoguideLocalMemory, AutoguideGlobalMemory
from .alfworld import ExpelShortMemory, ExpelLocalMemory, ExpelGlobalMemory

SHORT_MEMORY = dict(alfworld=dict(reflect=ReflectShortMemory, hpc=HPCShortMemory, autoguide=AutoguideShortMemory, expel=ExpelShortMemory))
LOCAL_MEMORY = dict(alfworld=dict(reflect=ReflectLocalMemory, hpc=HPCLocalMemory, autoguide=AutoguideLocalMemory, expel=ExpelLocalMemory))
GLOBAL_MEMORY = dict(alfworld=dict(hpc=HPCGlobalMemory, reflect=HPCGlobalMemory, autoguide=AutoguideGlobalMemory, expel=ExpelGlobalMemory))