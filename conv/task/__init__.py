import sys
from .deep_learning import MinistClassify, ImageClassify

__all__ = ["MinistClassify", "ImageClassify"]

thismodule = sys.modules[__name__]
required_attribute = ["model", "optimizer", "loss_function"]


# class VerifyConfig:
#     def __init__(self):
#         self.lr = 0.01


# for task_name in __all__:
#     task = getattr(thismodule, task_name)(VerifyConfig())
#     for attr in required_attribute:
#         if not hasattr(task, attr):
#             raise ValueError(f"task {task_name} has no attribute {attr}")
#     del task
