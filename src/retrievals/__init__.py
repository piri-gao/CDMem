from .alfworld import ReflectFewshotBuilder, CDMemFewshotBuilder, AutoguideFewshotBuilder, ExpelFewshotBuilder
from .scienceworld import ReflectFewshotBuilderSC, CDMemFewshotBuilderSC

FEWSHOT_BUILDER = dict(alfworld=dict(reflect=ReflectFewshotBuilder, cdmem=CDMemFewshotBuilder, autoguide=AutoguideFewshotBuilder, expel=ExpelFewshotBuilder,react=ReflectFewshotBuilder),
                       scienceworld=dict(reflect=ReflectFewshotBuilderSC, cdmem=CDMemFewshotBuilderSC))