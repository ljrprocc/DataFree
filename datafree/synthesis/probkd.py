from torch import nn
import torch
import torch.nn.functional as F
import random

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook
from datafree.utils import ImagePool, DataIter, clip_images
from datafree.criterions import jsdiv

class ProbSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, G_list, num_classes, img_size, nz, iterations=1, lr_g=0.1, synthesis_batch_size=128, sample_batch_size=128, save_dir='run/probkd', transform=None, normalizer=None, device='cpu', use_fp16=False, distributed=False, lmda_ent=0.5, adv=0.10, oh=0, act=0, l1=0.01, only_feature=False, depth=2, E=None, E_list=None):
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
        self.E_list = E_list
        self.lmda_ent = [lmda_ent, lmda_ent*5]
        self.adv = adv
        self.nz = nz
        self.L = depth + 1
        self.l1 = l1
        
        self.oh = [oh, oh*10]
        self.act = act
        self.only_feature = only_feature
        self._get_teacher_bn()
        # self.G = G
        self.optimizers = []
        # self.E = E
        # for l in range(self.L):
        #     if l == 0:
        #         para_list = G.parameters()
        #     else:
        #         # print(len(E.trans_convs)) 
        #         para_list = list(G.project.parameters()) + list(G.main[:-(4*l-2)].parameters()) + list(G.trans_convs[-l].parameters()) + list(E.main[3*l:].parameters()) + list(E.trans_convs[l].parameters()) + list(E.fc_mu.parameters()) + list(E.fc_var.parameters())
        #     optimizer = torch.optim.Adam(para_list, lr=self.lr_g, betas=(0.9,0.99))
        #     self.optimizers.append(optimizer)

        # self.G.train()
        # self.E.train()
            
        for G in G_list:
            optimizer = torch.optim.Adam(G.parameters(), lr=self.lr_g, betas=(0.9,0.99))
            self.optimizers.append(optimizer)
            G.train()
        # for G, E in zip(G_list, E_list):
        #     optimizer = torch.optim.Adam(list(G.parameters())+list(E.parameters()), lr=self.lr_g, betas=(0.9,0.99))
        #     self.optimizers.append(optimizer)
        #     G.train()
        #     E.train()
            

        # if only_feature:
        #     self.L = 1
        # else:
        #     self.L = len(G_list)

        # self.hooks = []
        # for m in teacher.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         self.hooks.append(DeepInversionHook(m))

        # assert len(self.hooks)>0, 'input model should contains at least one BN layer for DeepInversion and Probablistic KD.'

    def _get_teacher_bn(self):
        # todo: more automated layer design.
        # ResNet type
        if hasattr(self.teacher, 'layer1'):
            # layers = [self.teacher.bn1, self.teacher.layer1[-1].bn1, self.teacher.layer2[-1].bn1, self.teacher.layer3[-1].bn1]
            layers = [self.teacher.bn1, self.teacher.layer1[-1].bn2, self.teacher.layer2[-1].bn2, self.teacher.layer3[-1].bn2]
            if hasattr(self.teacher, 'layer4'):
                # print(type(self.teacher.layer4))
                # layers.append(self.teacher.layer4[-1].bn1)
                layers.append(self.teacher.layer4[-1].bn2)
        else:
            layers = [self.teacher.block1.layer[-1].bn2, self.teacher.block2.layer[-1].bn2, self.teacher.block3.layer[-1].bn2]
        self.stats = [(f.running_mean.data, f.running_var.data) for f in layers]
        self.bn_layers = layers
        

    # def update_l(self, l=0):
    #     self.optimizers[l].zero_grad()
    #     z1 = torch.randn(self.synthesis_batch_size, self.nz).to(self.device)
    #     mu_theta = G(z1, l=l)
    #     if l > 0:
    #         mu_l, var_l = self.stats[l]
    #         mu_theta_mean = torch.mean(mu_theta, (0,2,3))

        

    def synthesize(self, targets=None):
        self.student.eval()
        self.teacher.eval()
        # G = self.G
        
        for i in range(self.iterations // self.L):
            # l_rec = []
            # l_ent = []
            for l in range(self.L):
                G, E = self.G_list[l], self.E_list[l]
                if self.only_feature and l < self.L - 1:
                    continue
                G.train()
                self.optimizers[l].zero_grad()
                z1 = torch.randn(self.synthesis_batch_size, self.nz).to(self.device)
                # Rec and variance
                # mu_theta, logvar_theta = G(z1, l=l)
                # mu_theta, output_t = G(z1, l=l)
                mu_theta = G(z1, l=l)
                if l > 0:
                    mu_l, var_l = self.stats[l]
                    # Get encoder samples
                    # print(mu_l.shape)
                    # samples_real = mu_l.unsqueeze(0).unsqueeze(2).unsqueeze(3) + torch.sqrt(var_l).unsqueeze(0).unsqueeze(2).unsqueeze(3) * torch.randn_like(mu_theta.detach())
                    # nch = var_l.size(0)
                    # samples = samples.permute(1,0,2,3).contiguous().view([nch, -1]).permute(1,0)
                    # generated_distribution = torch.distributions.Normal(torch.mean(mu_theta, (0,2,3)), var_l)
                    # estimated_real_distribution = torch.distributions.Normal(mu_l, var_l)
                    # g_samples = generated_distribution.rsample((samples.size(0), ))
                    # rec = torch.sum(estimated_real_distribution.log_prob(samples).exp().unsqueeze(0) * (estimated_real_distribution.log_prob(samples) - generated_distribution.log_prob(g_samples)), 1).mean()

                    # samples = mu_l + torch.sqrt(var_l) * torch.randn_like(mu_l.repeat(self.synthesis_batch_size, 1))
                    # mu, logvar = E(samples, l=l)
                    # rec_feature_map = G(E.repara(mu, logvar), l=l)
                    # rec = F.mse_loss(rec_feature_map, samples)
                    # kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
                    
                    # samples = mu_theta + (logvar_theta / 2).exp() * torch.randn_like(mu_theta)
                    # mu_theta_mean = torch.mean(mu_theta, (0,2,3))
                    samples = mu_theta

                    sample_mean = torch.mean(samples, (0,2,3))
                    nch = sample_mean.size(0)
                    sample_var = samples.permute(1,0,2,3).contiguous().view([nch, -1]).var(1, unbiased=False)
                    # var_theta = torch.mean(logvar_theta, (0,2,3)).exp()
                    
                    # logvar = var_theta.log()
                    # print(logvar.shape, mu_theta.shape)
                    ## Directly calculate KL divergence, treat statistics as normal distribution.
                    # rec = torch.norm(mu_theta_mean - mu_l, p=2)
                    #rec = torch.sum(-sample_var.log() / 2 + (sample_var + (sample_mean - mu_l) ** 2) / (2*var_l))
                    # rec = torch.sum((mu_theta_mean - mu_l) ** 2 + (var_theta - var_l) ** 2)
                    ## var alignment by l2-norm
                    # x_random = torch.randn_like(mu_theta)
                    # mu_0 = torch.mean(x_random, (0, 2, 3))
                    # var_0 = x_random.permute(1,0,2,3).contiguous().view([nch, -1]).var(1, unbiased=False)
                    # delta_l = torch.quantile(torch.abs(mu_0 - mu_l), q=0.9)
                    # gamma_l = torch.quantile(torch.abs(var_0 - var_l), q=0.9)

                    # rec = torch.norm(F.relu((sample_mean - mu_l).abs() - delta_l), p=2) + torch.norm(F.relu((sample_var - var_l).abs() - gamma_l), p=2)
                    # rec = torch.norm(sample_mean - mu_l, p=2) + torch.norm(sample_var- var_l, p=2)
                    rec = torch.norm(sample_mean - mu_l, p=2)
                    # t_out_real = self.teacher(samples_real, l=l)
                    # rec = 0.0
                    # Generate samples from q_{\theta}
                    samples = mu_theta + torch.sqrt(var_l).unsqueeze(0).unsqueeze(2).unsqueeze(3) * torch.randn_like(mu_theta)
                    # samples = G(z1, l=l)
                    # samples = g_samples
                    # samples = mu_theta
                    # samples = mu_theta + (logvar_theta / 2).exp() * torch.randn_like(mu_theta)
                    layer = getattr(self.teacher, 'layer{}'.format(l))
                    func = lambda x: F.relu(layer[-1].bn2(layer[-1].conv2(F.relu(layer[-1].bn1(x)))))
                    # func = lambda x: F.relu(layer[-1].bn2(x))
                    # func_real = lambda x: layer[-1].conv1(x)
                    # b, c, h, w = mu_theta.size()
                    # rec_real = F.l1_loss(func_real(output_t),  mu_theta, reduction='sum') / (b*h*w)
                    # func = lambda x, y: F.relu(layer[-1].bn2(layer[-1].conv2(F.relu(layer[-1].bn1(x)))) + layer[-1].shortcut(y))
                    # samples = func(samples, output_t)
                    # samples = torch.nn.functional.relu(samples)
                    samples = func(samples)
                else:
                    # mu_theta = G(z1, l=l)
                    samples = self.normalizer(mu_theta)
                    x_input = self.normalizer(samples.detach(), reverse=True)
                    rec = 0.0
                    # l1_loss = 0.0
                    # rec_real = 0.0
                # print(samples.shape)
                t_out, t_feat = self.teacher(samples, l=l, return_features=True)
                # if l > 0:
                #     l1_loss = F.l1_loss(t_out, t_out_real, reduction='none').sum(1).mean(0)
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
                
                if l > 0:
                    lmda_ent = self.lmda_ent[0]
                    oh = self.oh[0]
                else:
                    lmda_ent = self.lmda_ent[0]
                    oh = self.oh[0]

                loss = rec + lmda_ent * ent + self.adv * loss_adv+ oh * loss_oh + self.act * loss_act
                # if l == 2:
                #     print(ent, loss_adv)
                loss.backward()
                self.optimizers[l].step()
        if self.only_feature:
            return {}
        # z = torch.randn( size=(self.synthesis_batch_size, self.G_list[0].nz)).to(self.device)
        # inputs = self.G_list[0](z)
        return {'synthetic': x_input}

    @torch.no_grad()
    def sample(self, l=0):
        # self.G.eval()
        # self.E.eval()
        self.G_list[l].eval()
        # self.E_list[l].eval()
        z = torch.randn( size=(self.sample_batch_size, self.nz), device=self.device )
        # inputs, logvar_theta = self.G_list[l](z, l=l)
        inputs = self.G_list[l](z, l=l)
        # inputs, output_t = self.G_list[l](z, l=l)
        if l > 0:
            _, var_l = self.stats[l]
            layer = getattr(self.teacher, 'layer{}'.format(l))
            func = lambda x, y: F.relu(layer[-1].bn2(layer[-1].conv2(F.relu(layer[-1].bn1(x)))) + layer[-1].shortcut(y))
            # sample inputs
            # func = lambda x: F.relu(layer[-1].bn2(layer[-1].conv2(F.relu(layer[-1].bn1(x)))))
            # func = lambda x: F.relu(layer[-1].bn2(x))
            inputs = inputs + torch.sqrt(var_l).unsqueeze(0).unsqueeze(2).unsqueeze(3) * torch.randn_like(inputs)
            # inputs = inputs + (logvar_theta / 2).exp() * torch.randn_like(inputs)
            # Activation at the last block
            # inputs = func(inputs, output_t)
            inputs = func(inputs)
        return inputs


                

        
        
                


