# from .alfworld import ReflectFewshotBuilder, HPCFewshotBuilder
#
# FEWSHOT_BUILDER = dict(alfworld=dict(reflect=ReflectFewshotBuilder, hpc=HPCFewshotBuilder))
# from .webshop import ReflectFewshotBuilder, HPCFewshotBuilder
from .scienceworld import ReflectFewshotBuilder, HPCFewshotBuilder

FEWSHOT_BUILDER = dict(scienceworld=dict(reflect=ReflectFewshotBuilder, hpc=HPCFewshotBuilder))