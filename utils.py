import os
import torch

class Dotdict(dict):
    def __getattr__(self, name):
        return self[name]