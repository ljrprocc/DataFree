import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .base import BaseSynthesis
from datafree.criterions import kldiv
from datafree.utils import ImagePool, DataIter, estimate_gradient_objective, compute_gradient, clip_images

class DFMESynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, img_size, loss='l1', no_logit=1, forward_differences=True, iterations=1, lr_g=1e-3, synthesis_batch_size=128, sample_batch_size=128, grad_epsilon=1e-3, grad_m=1, act=0, balance=0, criterion=None, normalizer=None, device='cpu', autocast=None, use_fp16=False, distributed=False, logit_correction='mean', num_classes=10):
        super(DFMESynthesizer, self).__init__(teacher, student)
        assert len(img_size) == 3, "image size should be a 3-dimension tuple"
        self.img_size = img_size 
        self.iterations = iterations
        self.nz = nz
        self.loss = loss
        if criterion is None:
            criterion = kldiv
        self.criterion = criterion
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        # scaling factors
        self.lr_g = lr_g
        self.grad_epsilon = grad_epsilon
        self.grad_m = grad_m
        self.forward_difference = forward_differences
        self.loss = loss
        self.no_logit = no_logit
        self.logit_correction = logit_correction
        # self.adv = adv
        # self.bn = bn
        # self.oh = oh
        # self.balance = balance
        self.act = act

        # generator
        self.generator = generator.to(device).train()
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g)
        self.distributed = distributed
        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.device = device
        self.num_classes = num_classes

    def synthesize(self):
        self.student.eval()
        self.generator.train()
        self.teacher.eval()
        for it in range(self.iterations):
            self.optimizer.zero_grad()
            z = torch.randn( size=(self.synthesis_batch_size, self.nz), device=self.device )
            inputs = self.generator(z)
            inputs = self.normalizer(inputs)
            # inputs = torch.tanh(inputs)
            # t_out = self.teacher(inputs)
            approx_grad_wrt_x, loss_G = estimate_gradient_objective(self.teacher, self.student, inputs, epsilon =self.grad_epsilon, m = self.grad_m, num_classes=self.num_classes,  device=self.device, pre_x=True, loss=self.loss, no_logits=self.no_logit, logit_correction=self.logit_correction)
            inputs.backward(approx_grad_wrt_x)
            self.optimizer.step()

        return { 'synthetic': self.normalizer(inputs.detach(), reverse=True) }

    @torch.no_grad()
    def sample(self):
        self.generator.eval()
        z = torch.randn( size=(self.sample_batch_size, self.nz), device=self.device )
        inputs = self.normalizer(self.generator(z))
        return inputs
