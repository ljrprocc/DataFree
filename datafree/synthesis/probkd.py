from torch import nn
import torch
import torch.nn.functional as F
import random

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook
from datafree.utils import ImagePool, DataIter, clip_images
from datafree.criterions import jsdiv

class ProbSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, G_list, num_classes, img_size, nz, iterations=1, lr_g=0.1, synthesis_batch_size=128, sample_batch_size=128, save_dir='run/probkd', transform=None, normalizer=None, device='cpu', use_fp16=False, distributed=False, lmda_ent=0.5, adv=0.10, oh=0, act=0):
        super(ProbSynthesizer, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.normalizer = normalizer
        # self.data_pool = ImagePool(root=self.save_dir)
        # self.data_iter = None
        self.transform = transform
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.use_fp16 =use_fp16
        self.device = device
        self.num_classes = num_classes
        self.G_list = G_list
        self.lmda_ent = lmda_ent
        self.adv = adv
        self.nz = nz
        
        self.oh = oh
        self.act = act
        self._get_teacher_bn()
        self.optimizers = []
        for G in G_list:
            optimizer = torch.optim.Adam(G.parameters(), lr=self.lr_g, betas=(0.9,0.99))
            self.optimizers.append(optimizer)
            G.train()

        # self.hooks = []
        # for m in teacher.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         self.hooks.append(DeepInversionHook(m))

        # assert len(self.hooks)>0, 'input model should contains at least one BN layer for DeepInversion and Probablistic KD.'

    def _get_teacher_bn(self):
        # todo: more automated layer design.
        # ResNet type
        if hasattr(self.teacher, 'layer1'):
            layers = [self.teacher.bn1, self.teacher.layer1[-1].bn2, self.teacher.layer2[-1].bn2, self.teacher.layer3[-1].bn2]
            if hasattr(self.teacher, 'layer4'):
                # print(type(self.teacher.layer4))
                layers.append(self.teacher.layer4[-1].bn2)
        else:
            layers = [self.teacher.block1.layer[-1].bn2, self.teacher.block2.layer[-1].bn2, self.teacher.block3.layer[-1].bn2]
        self.stats = [(f.running_mean.data, f.running_var.data) for f in layers]
        self.bn_layers = layers

    def synthesize(self, targets=None):
        self.student.eval()
        self.teacher.eval()
        
        for i in range(self.iterations // len(self.G_list)):
            # l_rec = []
            # l_ent = []
            for l, G in enumerate(self.G_list):
                G.train()
                self.optimizers[l].zero_grad()
                z1 = torch.randn(self.synthesis_batch_size, self.nz).to(self.device)
                # Rec and variance
                mu_theta, logvar_theta = G(z1, l=l)
                # mu_theta = G(z1, l=l)
                if l > 0:
                    mu_l, var_l = self.stats[l]
                    samples = mu_theta + (logvar_theta / 2).exp() * torch.randn_like(mu_theta)
                    # mu_theta_mean = torch.mean(mu_theta, (0,2,3))
                    sample_mean = torch.mean(samples, (0,2,3))
                    nch = sample_mean.size(0)
                    sample_var = samples.permute(1,0,2,3).contiguous().view([nch, -1]).var(1, unbiased=False)
                    # var_theta = torch.mean(logvar_theta, (0,2,3)).exp()
                    
                    # logvar = var_theta.log()
                    # print(logvar.shape, mu_theta.shape)
                    ## Directly calculate KL divergence, treat statistics as normal distribution.
                    # rec = torch.norm(mu_theta_mean - mu_l, p=2, dim=1).mean()
                    #rec = torch.sum(-sample_var.log() / 2 + (sample_var + (sample_mean - mu_l) ** 2) / (2*var_l))
                    # rec = torch.sum((mu_theta_mean - mu_l) ** 2 + (var_theta - var_l) ** 2)
                    ## var alignment by l2-norm
                    rec = torch.norm(sample_mean - mu_l, p=2) + torch.norm(sample_var - var_l, p=2)
                    # Generate samples from q_{\theta}
                    # samples = mu_theta + torch.sqrt(var_l).unsqueeze(0).unsqueeze(2).unsqueeze(3) * torch.randn_like(mu_theta)
                    # samples = mu_theta
                    # samples = mu_theta + (logvar_theta / 2).exp() * torch.randn_like(mu_theta)
                    samples = torch.nn.functional.relu(samples)
                else:
                    samples = self.normalizer(mu_theta)
                    x_input = self.normalizer(samples.detach(), reverse=True)
                    rec = 0.0
                # print(samples.shape)
                t_out, t_feat = self.teacher(samples, l=l, return_features=True)
                p = F.softmax(t_out, dim=1).mean(0)
                ent = (p*p.log()).sum()
                loss_oh = F.cross_entropy( t_out, t_out.max(1)[1])
                loss_act = - t_feat.abs().mean()
                # ent = ent.mean()
                
                # Negative Divergence.
                if self.adv > 0:
                    s_out = self.student(samples, l=l)
                    l_js = jsdiv(s_out, t_out, T=3)
                    loss_adv = 1.0-torch.clamp(l_js, 0.0, 1.0)
                    # if loss_adv.item() == 0.0:
                    #     print('Warning: high js divergence between teacher and student')
                    #     print(ent, s_out.mean(), out_t_logit.mean())
                else:
                    loss_adv = torch.zeros(1).to(self.device)

                loss = rec + self.lmda_ent * ent + self.adv * loss_adv+ self.oh * loss_oh + self.act * loss_act
                # if l == 2:
                #     print(ent, loss_adv)
                loss.backward()
                self.optimizers[l].step()
        # z = torch.randn( size=(self.synthesis_batch_size, self.G_list[0].nz)).to(self.device)
        # inputs = self.G_list[0](z)
        return {'synthetic': x_input}

    @torch.no_grad()
    def sample(self, l=0):
        self.G_list[l].eval() 
        z = torch.randn( size=(self.sample_batch_size, self.nz), device=self.device )
        inputs, logvar_theta = self.G_list[l](z, l=l)
        # inputs = self.G_list[l](z, l=l)
        if l > 0:
            _, var_l = self.stats[l]
            # sample inputs
            # inputs = inputs + torch.sqrt(var_l).unsqueeze(0).unsqueeze(2).unsqueeze(3) * torch.randn_like(inputs)
            inputs = inputs + (logvar_theta / 2).exp() * torch.randn_like(inputs)
            # Activation at the last block
            inputs = torch.nn.functional.relu(inputs)
        return inputs


                

        
        
                


