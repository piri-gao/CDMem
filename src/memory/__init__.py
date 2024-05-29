from .alfworld import ReflectShortMemory, ReflectLocalMemory, HPCShortMemory, HPCLocalMemory

SHORT_MEMORY = dict(alfworld=dict(reflect=ReflectShortMemory, hpc=HPCShortMemory))
LOCAL_MEMORY = dict(alfworld=dict(reflect=ReflectLocalMemory, hpc=HPCLocalMemory))