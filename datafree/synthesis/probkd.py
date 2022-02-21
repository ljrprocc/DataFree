from torch import nn
import torch
import torch.nn.functional as F
import random

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook
from datafree.utils import ImagePool, DataIter, clip_images
from datafree.criterions import jsdiv

class ProbSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, G_list, num_classes, img_size, iterations=1000, lr_g=0.1, synthesis_batch_size=128, sample_batch_size=128, save_dir='run/probkd', transform=None, normalizer=None, device='cpu', use_fp16=False, distributed=False, lmda_ent=0.5, adv=0.1):
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
        self.num_classes = num_classes
        self.G_list = G_list
        self.lmda_ent = lmda_ent
        self.adv = adv
        self._get_teacher_bn()
        self.optimizers = []
        for G in G_list:
            optimizer = torch.optim.Adam(G.parameters(), lr=self.lr_g, betas=(0.9,0.99))
            self.optimizers.append(optimizer)

        # self.hooks = []
        # for m in teacher.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         self.hooks.append(DeepInversionHook(m))

        # assert len(self.hooks)>0, 'input model should contains at least one BN layer for DeepInversion and Probablistic KD.'

    def _get_teacher_bn(self):
        # todo: more automated layer design.
        layers = [self.teacher.bn1, self.teacher.layer1[-1].bn2, self.teacher.layer2[-1].bn2, self.teacher.layer3[-1].bn2, self.teacher.layer4[-1].bn2]
        self.stats = [(f.running_mean.data, f.running_var.data) for f in layers]

    def synthesize(self, targets=None):
        self.student.eval()
        self.teacher.eval()
        
        for i in range(self.iterations // len(self.G_list)):
            # l_rec = []
            # l_ent = []
            for l, G in enumerate(self.G_list):
                G.train()
                z1 = torch.randn(self.synthesis_batch_size, G.nz).to(self.device)
                # Rec and variance
                mu_theta = G(z1, l=l)
                if l > 0:
                    mu_l, var_l = self.stats[l]
                    mu_theta_mean = torch.mean(mu_theta, (2,3))
                    rec = torch.norm(mu_theta_mean - mu_l.unsqueeze(0), p=2, dim=1).mean()
                    # l_rec.append(rec)
                    # Entropy
                    # generate samples:
                    # s_l = mu_l + var_l * torch.randn_like(mu_l)
                    # else:
                    #     s_l = mu_theta
                    #     rec = 0.0
                    # Temporarily mu_theta, and constant var_theta = var_l
                    # generate samples:
                    samples = mu_theta + var_l.unsqueeze(0).unsqueeze(2).unsqueeze(3) * torch.randn_like(mu_theta)
                else:
                    samples = mu_theta
                    rec = 0.0
                # print(samples.shape)
                out_t_logit = self.teacher(samples, l=l)
                ent = - torch.sum(torch.log_softmax(out_t_logit, 1) * torch.softmax(out_t_logit, 1), 1)
                ent = ent.mean()
                
                # Negative Divergence.
                if self.adv > 0:
                    s_out = self.student(samples, l=l)
                    loss_adv = 1.0-torch.clamp(jsdiv(s_out, out_t_logit, T=3), 0.0, 1.0)
                else:
                    s_out = torch.zeros(1).to(self.device)

                loss = rec + self.lmda_ent * ent + self.adv * loss_adv
                # if l == 2:
                #     print(ent, loss_adv)

                self.optimizers[l].zero_grad()
                loss.backward()
                self.optimizers[l].step()
        z = torch.randn( size=(self.synthesis_batch_size, self.G_list[0].nz)).to(self.device)
        inputs = self.G_list[0](z)
        return {'synthetic': inputs.detach()}

    @torch.no_grad()
    def sample(self, l=0):
        for G in self.G_list:
            G.eval()
        z = torch.randn( size=(self.sample_batch_size, self.G_list[l].nz), device=self.device )
        inputs = self.G_list[l](z, l=l)
        return inputs


                

        
        
                


