from torch import nn
import torch
import torch.nn.functional as F
import random

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook
from datafree.utils import ImagePool, DataIter, clip_images

class ProbSynthesizer(BaseSynthesis):
    def __init_(self, teacher, student, num_classes, img_size, iterations=1000, lr_g=0.1, synthesis_batch_size=128, sample_batch_size=128, save_dir='run/probkd', transform=None, normalizer=None, device='cpu', use_fp16=False, distributed=False):
        super(ProbSynthesizer, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        self.transform = transform
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.use_fp16 =use_fp16
        self.device = device

        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m))

        assert len(self.hooks)>0, 'input model should contains at least one BN layer for DeepInversion and Probablistic KD.'

    def synthesize(self, targets=None):
        self.student.eval()