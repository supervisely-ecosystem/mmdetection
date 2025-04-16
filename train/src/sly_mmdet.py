import sys
import os
from serve.src.main import MMDetectionModel


class MMDetModelBench(MMDetectionModel):
    in_train = True
