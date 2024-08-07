from .alfworld import HPCPromptBuilder, ReflectPromptBuilder, AutoguidePromptBuilder, ExpelPromptBuilder

PROMPT_BUILDER = dict(alfworld=dict(reflect=ReflectPromptBuilder, hpc=HPCPromptBuilder, autoguide=AutoguidePromptBuilder, expel=ExpelPromptBuilder))