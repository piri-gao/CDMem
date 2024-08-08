# from .alfworld import HPCPromptBuilder, ReflectPromptBuilder
#
# PROMPT_BUILDER = dict(alfworld=dict(reflect=ReflectPromptBuilder, hpc=
# from .webshop import HPCPromptBuilder, ReflectPromptBuilder
from .scienceworld import HPCPromptBuilder, ReflectPromptBuilder

PROMPT_BUILDER = dict(scienceworld=dict(reflect=ReflectPromptBuilder, hpc=HPCPromptBuilder))