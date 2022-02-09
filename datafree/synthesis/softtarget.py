from inspect import signature
from attr import has
import torch
from torch import nn
import torch.nn.functional as F
import random

from .base import BaseSynthesis
from datafree.utils import ImagePool, DataIter, clip_images
from .deepinversion import jitter_and_flip

class SoftTargetSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, num_classes, img_size, 
                 iterations=1000, lr_g=0.001, progressive_scale=False,
                 synthesis_batch_size=128, sample_batch_size=128, 
                 a=1.0, sigma=1.0,
                 save_dir='run/softtarget', transform=None, T=20,
                 normalizer=None, device='cpu',
                 # TODO: FP16 and distributed training 
                 autocast=None, use_fp16=False, distributed=False, layer='fc_2'):
        super(SoftTargetSynthesizer, self).__init__(teacher, student)
        assert len(img_size)==3, "image size should be a 3-dimension tuple"

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
        self.num_classes = num_classes
        self.distributed = distributed
        self.T = T

        self.progressive_scale = progressive_scale
        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.device = device
        self.sigma = sigma
        self.a = a
        assert layer in ['fc', 'fc_2']
        if layer == 'fc':
            self.weight = self.teacher.fc.weight
        else:
            if hasattr(self.teacher, 'layer4'):
                self.weight = self.teacher.layer4[-1].conv2.weight
            else:
                self.weight = self.teacher.block3.layer[-1].conv2.weight
            # print(self.weight.shape)
            # exit(-1)
            self.weight = F.adaptive_avg_pool2d(self.weight, (1, 1))
            self.weight = self.weight.view(self.weight.size(0), -1)

        self.covariance, self.R = self._generate_relation()

    def _generate_relation(self):
        # R = F.cosine_similarity(self.weight)
        # for i in range(R.size(0)):
        #     R[i, i] = 1.
        a = self.weight.cpu()
        n = a.norm(p=2, dim=0)
        R = torch.matmul((a / n.unsqueeze(0)).T, a / n.unsqueeze(0))
        # R = torch.round(R * 1000) / 1000
        
        sigma_matrix =torch.diag(self.sigma ** 0.5 * torch.ones(R.size(0)))
        covariance = torch.matmul(torch.matmul(sigma_matrix, R), sigma_matrix)
        # covariance = torch.round(covariance * 1000) / 1000
        # print(covariance, R)
        return covariance, R


    def synthesize(self, targets=None):
        '''
        Algorithm 1 of the original paper.

        '''
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        # inputs = torch.FloatTensor(size=[self.synthesis_batch_size, *self.img_size]).uniform_(-1, 1).to(self.device)
        inputs = torch.randn(self.synthesis_batch_size, *self.img_size).to(self.device)
        inputs = inputs.requires_grad_()
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
            targets = targets.sort()[0]
        
        targets = targets.to(self.device)

        optimizer = torch.optim.Adam([inputs], self.lr_g, betas=[0.5, 0.99])
        best_inputs = inputs.data
        distribution = torch.distributions.MultivariateNormal(torch.zeros(self.R.shape[0]), self.covariance)
        # distribution = torch.distributions.Normal(torch.zeros(self.R.shape[0]), scale=self.covariance)
        for it in range(self.iterations):
            inputs = jitter_and_flip(inputs)
            t_out, t_feat = self.teacher(self.normalizer(inputs), return_features=True)
            # sample feature
            
            s_samples = distribution.rsample((self.synthesis_batch_size, )).to(self.device)
            # with torch.no_grad():
            y_soft = self.teacher.linear(s_samples)
            # y_soft = torch.softmax(y_soft, 1)
            loss_act = - t_feat.abs().mean()
            loss_d = F.kl_div(torch.log_softmax(y_soft.detach() / self.T, 1), torch.softmax(t_out / self.T, 1))
            # loss_l2 = torch.norm(inputs, 2)
            # loss_oh = F.cross_entropy( t_out, targets )
            # print(torch.mean(inputs, (1,2,3)))
            loss = loss_d + self.a * loss_act
            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data
            # print(loss_d.item(), loss_act.item(), loss.item())
            # print(torch.mean(inputs, (1,2,3)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(torch.mean(inputs, (1,2,3)))
            # exit(-1)
            # inputs.data = clip_images(inputs.data, self.normalizer.mean, self.normalizer.std)

        self.student.train()
        # save best inputs and reset data loader
        # if self.normalizer:
        #     best_inputs = self.normalizer(best_inputs, True)
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
        

             
