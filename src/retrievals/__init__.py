from .alfworld import ReflectFewshotBuilder, HPCFewshotBuilder, AutoguideFewshotBuilder, ExpelFewshotBuilder

FEWSHOT_BUILDER = dict(alfworld=dict(reflect=ReflectFewshotBuilder, hpc=HPCFewshotBuilder, autoguide=AutoguideFewshotBuilder, expel=ExpelFewshotBuilder))