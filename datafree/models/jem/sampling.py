
import torch as t
import numpy as np
from torch.nn.modules.loss import _Loss
from math import sqrt
import torchvision as tv


class Hamiltonian(_Loss):

    def __init__(self, layer, reg_cof=1e-4):
        super(Hamiltonian, self).__init__()
        self.layer = layer
        self.reg_cof = 0

    def forward(self, x, p):

        y = self.layer(x)
        H = t.sum(y * p)
        # H = H - self.reg_cof * l2
        return H

def plot(p, x):
    return tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

def init_random(bs):
    return t.FloatTensor(bs, 3, 32, 32).uniform_(-1, 1)

def init_inform(bs, n_classes=10):
    global conditionals
    n_ch = 3
    size = [3, 32, 32]
    im_sz = 32
    new = t.zeros(bs, n_ch, im_sz, im_sz)
    for i in range(bs):
        index = np.random.randint(n_classes)
        dist = conditionals[index]
        new[i] = dist.sample().view(size)
    return t.clamp(new, -1, 1).cpu()

def init_from_centers(args):
    global conditionals
    from torch.distributions.multivariate_normal import MultivariateNormal
    bs = args.buffer_size
    if args.dataset == 'svhn':
        size = [3, 28, 28]
    else:
        size = [3, 32, 32]
    if args.dataset == 'cifar_test':
        args.dataset = 'cifar10'
    centers = t.load('%s_mean.pt' % args.dataset)
    covs = t.load('%s_cov.pt' % args.dataset)

    buffer = []
    for i in range(args.n_classes):
        mean = centers[i].to(args.device)
        cov = covs[i].to(args.device)
        dist = MultivariateNormal(mean, covariance_matrix=cov + 1e-4 * t.eye(int(np.prod(size))).to(args.device))
        buffer.append(dist.sample((bs // args.n_classes, )).view([bs // args.n_classes] + size).cpu())
        conditionals.append(dist)
    return t.clamp(t.cat(buffer), -1, 1)

def sample_p_0(device, replay_buffer, bs, y=None):
    if y is not None:
        n_classes = 10
    if len(replay_buffer) == 0:
        return init_random(bs), []
    buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // n_classes
    if buffer_size > bs:
        inds = t.randint(0, buffer_size, (bs,))
    else:
        inds = t.arange(0, bs)
    # if cond, convert inds to class conditional inds
    if y is not None:
        inds = y.cpu() * buffer_size + inds
        # assert not args.uncond, "Can't drawn conditional samples without giving me y"
    buffer_samples = replay_buffer[inds]
    # if args.init == 'i':
    #     random_samples = init_inform(args, bs)
    # else:
    random_samples = init_random(bs)
    choose_random = (t.rand(bs) < 0.05).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(device), inds


def sample_q(f, replay_buffer, y=None, n_steps=10, in_steps=10, args=None):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """

    # f.eval()
    # get batch size
    bs = args.batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(args.device, replay_buffer, bs=bs, y=y)
    x_k = t.autograd.Variable(init_sample, requires_grad=True).to(args.device)
    # sgld
    if args.in_steps > 0:
        Hamiltonian_func = Hamiltonian(f.f.layer_one)

    eps = 1
    for it in range(n_steps):
        energies = f(x_k, y=y)
        e_x = energies.sum()
        # wgrad = f.f.conv1.weight.grad
        eta = t.autograd.grad(e_x, [x_k], retain_graph=True)[0]
        # e_x.backward(retain_graph=True)
        # eta = x_k.grad.detach()
        # f.f.conv1.weight.grad = wgrad

        if in_steps > 0:
            p = 1.0 * f.f.layer_one_out.grad
            p = p.detach()

        tmp_inp = x_k.data
        tmp_inp.requires_grad_()
        if args.sgld_lr > 0:
            # if in_steps == 0: use SGLD other than PYLD
            # if in_steps != 0: combine outter and inner gradients
            # default 0
            if eps > 0:
                eta = t.clamp(eta, -eps, eps)
            tmp_inp = x_k + eta * args.sgld_lr
            if eps > 0:
                tmp_inp = t.clamp(tmp_inp, -1, 1)

        for i in range(in_steps):

            H = Hamiltonian_func(tmp_inp, p)

            eta_grad = t.autograd.grad(H, [tmp_inp], only_inputs=True, retain_graph=True)[0]
            if eps > 0:
                eta_step = t.clamp(eta_grad, -eps, eps)
            else:
                eta_step = eta_grad * args.pyld_lr

            tmp_inp.data = tmp_inp.data + eta_step
            if eps > 0:
                tmp_inp = t.clamp(tmp_inp, -1, 1)

        x_k.data = tmp_inp.data

        if args.sgld_std > 0.0:
            x_k.data += args.sgld_std * t.randn_like(x_k)

    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples


def uncond_samples(f, args):

    if args.init == 'i':
        init_from_centers(args)
        replay_buffer = init_from_centers(args)
    else:
        replay_buffer = t.FloatTensor(args.buffer_size, 3, 32, 32).uniform_(-1, 1)
    for i in range(args.n_sample_steps):
        samples = sample_q(f, replay_buffer, y=None, n_steps=args.n_steps, in_steps=args.in_steps, args=args)
        if i % args.print_every == 0:
            plot('{}/samples_{}.png'.format(args.save_dir, i), samples)
        print(i)
    return replay_buffer