import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions import Dirichlet

from .base import BaseSynthesis
from .deepinversion import jitter_and_flip
from datafree.utils import ImagePool, clip_images, DataIter

def compute_simiarity(x, scale=1.0):
    x = F.normalize(x, p=2, dim=1)
    x = torch.matmul(x, x.T)
    norm1 = (x - x.min()) / (x.max() - x.min())
    return norm1 / scale

def getDirch(n, sim_matrix, row_idx, scale=1.0):
    sim = sim_matrix[row_idx]
    # tmp = (sim - sim.min()) / (sim.max() - sim.min())
    dist = Dirichlet(sim*scale+0.0001)
    x = dist.rsample((n, ))
    return x


class ZSKDSynthesis(BaseSynthesis):
    def __init__(self, teacher, student, num_classes, img_size, iterations=1000, lr_g=0.1, synthesis_batch_size=128, sample_batch_size=128, device='cpu', save_dir='run/zskd', distributed=False, normalizer=None, transform=None, T=20):
        super(ZSKDSynthesis, self).__init__(teacher, student)
        assert len(img_size)==3, "image size should be a 3-dimension tuple"
        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        self.transform = transform
        self.synthesize_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        # Scaling factors. TO DO
        
        self.num_classes = num_classes
        self.distributed = distributed
        self.device = device
        x = self.teacher.linear.weight
        # print(x.shape)
        self.sim_mat_big = compute_simiarity(x, 1)
        self.sim_mat_small = compute_simiarity(x, 0.1)
        self.T = T
        

    def synthesize(self, targets=None):
        self.student.eval()
        best_cost = 1e6
        inputs = torch.randn( size=[self.synthesize_batch_size, *self.img_size], device=self.device).requires_grad_()
        if targets is None:
            targets = torch.randint(0, self.num_classes, size=(self.synthesize_batch_size, ))
            targets = targets.sort()[0]
        targets = targets.to(self.device)

        optimizer = torch.optim.Adam([inputs], self.lr_g)

        best_inputs = inputs.data
        for c in range(self.num_classes):

            sampled_label = getDirch(self.sample_batch_size, self.sim_mat_small, c, 1)
            # print(sampled_label.shape)
            for i in range(self.iterations // self.num_classes):
                inputs_aug = jitter_and_flip(inputs)
                logit_t = self.teacher(inputs_aug)
                # print(logit_t.shape)
                # loss = -(F.log_softmax(logit_t / self.T, 1) * sampled_label.detach()).sum(1) / sampled_label.size(0)
                loss = self.T ** 2 * F.kl_div(torch.log_softmax(logit_t / self.T, 1), sampled_label.detach(), size_average=False) / sampled_label.size(0)
                # print(loss)
                if best_cost > loss.item():
                    best_cost = loss.item()
                    best_inputs = inputs.data
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # inputs.data = clip_images(inputs.data, self.normalizer.mean, self.normalizer.std)
            self.data_pool.add( best_inputs )
        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst) if self.distributed else None
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        return {'synthetic': best_inputs}

    def sample(self):
        return self.data_iter.next()
