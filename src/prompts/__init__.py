from .alfworld import HPCPromptBuilder, ReflectPromptBuilder

PROMPT_BUILDER = dict(alfworld=dict(reflect=ReflectPromptBuilder, hpc=HPCPromptBuilder))