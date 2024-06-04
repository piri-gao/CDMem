from .alfworld import ReflectShortMemory, ReflectLocalMemory, HPCShortMemory, HPCLocalMemory, GlobalMemory

SHORT_MEMORY = dict(alfworld=dict(reflect=ReflectShortMemory, hpc=HPCShortMemory))
LOCAL_MEMORY = dict(alfworld=dict(reflect=ReflectLocalMemory, hpc=HPCLocalMemory))
GLOBAL_MEMORY = dict(alfworld=dict(hpc=GlobalMemory, reflect=GlobalMemory))