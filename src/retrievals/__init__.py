from .alfworld import ReflectFewshotBuilder, CDMemFewshotBuilder, AutoguideFewshotBuilder, ExpelFewshotBuilder

FEWSHOT_BUILDER = dict(alfworld=dict(reflect=ReflectFewshotBuilder, cdmem=CDMemFewshotBuilder, autoguide=AutoguideFewshotBuilder, expel=ExpelFewshotBuilder,react=ReflectFewshotBuilder))