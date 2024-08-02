# from .alfworld import ReflectFewshotBuilder, HPCFewshotBuilder
#
# FEWSHOT_BUILDER = dict(alfworld=dict(reflect=ReflectFewshotBuilder, hpc=HPCFewshotBuilder))
from .webshop import ReflectFewshotBuilder, HPCFewshotBuilder

FEWSHOT_BUILDER = dict(webshop=dict(reflect=ReflectFewshotBuilder, hpc=HPCFewshotBuilder))