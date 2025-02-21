from .alfworld import CDMemPromptBuilder, ReflectPromptBuilder, AutoguidePromptBuilder, ExpelPromptBuilder

PROMPT_BUILDER = dict(alfworld=dict(reflect=ReflectPromptBuilder, cdmem=CDMemPromptBuilder, autoguide=AutoguidePromptBuilder, expel=ExpelPromptBuilder, react=ReflectPromptBuilder))