from torch import nn
import torch
import torch.nn.functional as F
import random

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook
from datafree.utils import ImagePool, DataIter, clip_images
from datafree.criterions import jsdiv, kldiv

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class ProbSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, G_list, num_classes, img_size, nz, iterations=1, lr_g=0.1, synthesis_batch_size=128, sample_batch_size=128, save_dir='run/probkd', transform=None, normalizer=None, device='cpu', use_fp16=False, distributed=False, lmda_ent=0.5, adv=0.10, oh=0, act=0, l1=0.01, only_feature=False, depth=2, adv_type='js'):
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
        # self.E_list = E_list
        self.adv_type = adv_type
        self.lmda_ent = lmda_ent
        self.adv = adv
        self.nz = nz
        self.L = depth + 1
        self.l1 = l1
        
        self.oh = oh
        self.act = act
        self.only_feature = only_feature
        self._get_teacher_bn()
        self.optimizers = []
        for i, G in enumerate(self.G_list):
            reset_model(G)
            optimizer = torch.optim.Adam(G.parameters(), self.lr_g, betas=[0.9, 0.99])
            self.optimizers.append(optimizer)
        # self.G = G
        # self.optimizers = [None] * len(G_list)

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
            # layers = [self.teacher.bn1, self.teacher.layer1[-1].bn1, self.teacher.layer2[-1].bn1, self.teacher.layer3[-1].bn1]
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
        # optimizers = []
        
        
        
        for i in range(self.iterations // len(self.G_list)):
            if targets is None:
                targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
                targets = targets.sort()[0] # sort for better visualization
            targets = targets.to(self.device)
            z1 = torch.randn(self.synthesis_batch_size, self.nz).to(self.device)
            # l_rec = []
            # l_ent = []
            for l, G in enumerate(self.G_list):
                G.train()
                self.optimizers[l].zero_grad()
                
                # Rec and variance
                # mu_theta, logvar_theta = G(z1, l=l)
                mu_theta = G(z1, l=l, y=targets)
                if l > 0:
                    nch = mu_theta.size(1)
                    mu_l, var_l = self.stats[l]
                    sample_var = mu_theta.permute(1,0,2,3).contiguous().view([nch, -1]).var(1, unbiased=False)
                    # samples = mu_theta + (logvar_theta / 2).exp() * torch.randn_like(mu_theta)
                    mu_theta_mean = torch.mean(mu_theta, (0,2,3))
                   
                    ## Directly calculate KL divergence, treat statistics as normal distribution.
                    rec = torch.norm((mu_theta_mean - mu_l) / (2 * var_l), p=2)
                    
                    # Generate samples from q_{\theta}
                    samples = mu_theta + torch.sqrt(var_l).unsqueeze(0).unsqueeze(2).unsqueeze(3) * torch.randn_like(mu_theta)
                    layer = getattr(self.teacher, 'layer{}'.format(l))
                    
                    func = lambda x: F.relu(x)
                    samples = func(samples)
                else:
                    samples = self.normalizer(mu_theta)
                    x_input = self.normalizer(samples.detach(), reverse=True)
                    rec = torch.zeros(1).to(self.device)
                # print(samples.shape)
                t_out, t_feat = self.teacher(samples, l=l, return_features=True)
                p = F.softmax(t_out, dim=1).mean(0)
                ent = (p*p.log()).sum()
                # if targets is None:
                # loss_oh = F.cross_entropy( t_out, t_out.max(1)[1])
                loss_oh = F.cross_entropy( t_out, targets )
                loss_act = - t_feat.abs().mean()
                # ent = ent.mean()
                
                # Negative Divergence.
                if self.adv > 0:
                    s_out = self.student(samples, l=l)
                    if self.adv_type == 'js':
                        l_js = jsdiv(s_out, t_out, T=3)
                        loss_adv = 1.0-torch.clamp(l_js, 0.0, 1.0)
                    if self.adv_type == 'kl':
                        mask = (s_out.max(1)[1]==t_out.max(1)[1]).float()
                        loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean()
                    # if loss_adv.item() == 0.0:
                    #     print('Warning: high js divergence between teacher and student')
                    #     print(ent, s_out.mean(), out_t_logit.mean())
                else:
                    loss_adv = torch.zeros(1).to(self.device)

                # print(rec, ent, loss_adv, loss_oh, loss_act)
                # print(type(rec), type(ent))
                loss = rec + self.lmda_ent * ent + self.adv * loss_adv+ self.oh * loss_oh + self.act * loss_act
                # if l == 2:
                #     print(ent, loss_adv)
                loss.backward()
                self.optimizers[l].step()
                # if l == 1 and i % 50 == 0:
                #     # Debug not log.
                #     print(i, rec.item(), loss_adv.item())
        # z = torch.randn( size=(self.synthesis_batch_size, self.G_list[0].nz)).to(self.device)
        # inputs = self.G_list[0](z)
        return {'synthetic': x_input}

    @torch.no_grad()
    def sample(self, l=0):
        self.G_list[l].eval() 
        z = torch.randn( size=(self.sample_batch_size, self.nz), device=self.device )
        # inputs, logvar_theta = self.G_list[l](z, l=l)
        targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,), device=self.device)
        targets = targets.sort()[0]
        inputs = self.G_list[l](z, l=l, y=targets)
        if l > 0:
            _, var_l = self.stats[l]
            # sample inputs
            inputs = inputs + torch.sqrt(var_l).unsqueeze(0).unsqueeze(2).unsqueeze(3) * torch.randn_like(inputs)
            # inputs = inputs + (logvar_theta / 2).exp() * torch.randn_like(inputs)
            # Activation at the last block
            layer = getattr(self.teacher, 'layer{}'.format(l))
            
            # func = lambda x: F.relu(layer[-1].bn2(layer[-1].conv2(F.relu(x))))
            func = lambda x: F.relu(x)
            inputs = func(inputs)
        return inputs


                

        
        
                


