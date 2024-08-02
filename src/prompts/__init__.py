# from .alfworld import HPCPromptBuilder, ReflectPromptBuilder
#
# PROMPT_BUILDER = dict(alfworld=dict(reflect=ReflectPromptBuilder, hpc=
from .webshop import HPCPromptBuilder, ReflectPromptBuilder

PROMPT_BUILDER = dict(webshop=dict(reflect=ReflectPromptBuilder, hpc=HPCPromptBuilder))