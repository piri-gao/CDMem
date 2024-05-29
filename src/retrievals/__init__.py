from .alfworld import ReflectFewshotBuilder, HPCFewshotBuilder

FEWSHOT_BUILDER = dict(alfworld=dict(reflect=ReflectFewshotBuilder, hpc=HPCFewshotBuilder))