import torch

from .base import BaseSynthesis
from datafree.utils import UnlabelBufferDataset, ImagePool, DataIter
import torchvision.transforms as T
# from datafree.models.score_sde import sampling, configs
# from datafree.models.score_sde.sampling import (ReverseDiffusionPredictor, 
                    #   LangevinCorrector)

class PretrainedGenerativeSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, img_size, synthesis_batch_size=128, sample_batch_size=128, normalizer=None, device='cpu', mode='gan', use_ddim=False, replay_buffer=None, transform=None, distributed=False, sde=None, inverse_scaler=None):
        super(PretrainedGenerativeSynthesizer, self).__init__(teacher, student)
        self.mode = mode
        # assert len(img_size)==3, "image size should be a 3-dimension tuple"
        self.img_size = img_size 
        
        self.nz = nz
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        self.generator = generator
        # if self.mode == 'diffusion':
        #     self.generator, self.diffusion = generator
        self.device = device
        self.use_ddim = use_ddim
        self.replay_buffer = replay_buffer
        self.transform = transform
        self.distributed = distributed
        self.sde = sde
        self.inverse_scaler = inverse_scaler

    def synthesize(self, l=None):
        if self.mode == 'ebm' or self.mode == 'diffusion' or self.mode == 'sde':
            # transform = T.Compose(self.transform.transforms[1:])
            dst = UnlabelBufferDataset(self.replay_buffer, transform=self.transform)
            if self.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dst) if self.distributed else None
            else:
                train_sampler = None
            loader = torch.utils.data.DataLoader(
                dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
                num_workers=4, pin_memory=True, sampler=train_sampler)
            self.data_iter = DataIter(loader)

    
    @torch.no_grad()
    def sample(self):
        G = self.generator
        if self.mode == 'glow':
            self.generator.set_actnorm_init()
        # elif self.mode == 'diffusion':
        #     print(type(self.generator))
        #     self.generator, self.diffusion = self.generator
        if self.generator is not None:
            self.generator = self.generator.to(self.device)
            self.generator.eval()
        if self.mode == 'gan':
            z = torch.randn( size=(self.sample_batch_size, self.nz), device=self.device )
            inputs = self.generator(z)
            # print(inputs.min(), inputs.max())
            # exit(-1)
            return inputs
        elif self.mode == 'glow':
            z = torch.randn( size=(self.sample_batch_size, 48, 4, 4), device=self.device )
            x_intermideate = self.generator(z=z, temperature=1, reverse=True)
            

            # output = torch.sigmoid(x_intermideate)
            if torch.isnan(output.mean()):
                print(output, x_intermideate)
                exit(-1)
            output = x_intermideate.clamp_(-0.5, 0.5)
            output = output + 0.5
            # print(output.shape)
            # exit(-1)
            # if torch.isnan(output.mean()):
            #     print(x_intermideate)
            #     exit(-1)
            # print(output.min(), output.max())
            # exit(-1)
            return output
        elif self.mode == 'diffusion':
            # Online sampling, maybe slow.
            # sample_fn = (
            #     self.diffusion.p_sample_loop if not self.use_ddim else self.diffusion.ddim_sample_loop
            # )
            # sample = sample_fn(
            #     self.generator,
            #     (self.sample_batch_size, 3, self.img_size, self.img_size),
            #     clip_denoised=True,
            #     model_kwargs={}
            # )
            # sample = (sample + 1) / 2
            # sample = torch.clamp(sample, 0, 1)
            # return sample
            # Offline sampling

            return self.data_iter.next()

        elif self.mode == 'ebm':
            # UnlabelBufferDataset(self.replay_buffer, )
            return self.data_iter.next()

        elif self.mode == 'sde':
            #@title PC sampling
            # img_size = config.data.image_size
            # channels = config.data.num_channels
            # Online sampling process, maybe critically slow
            # shape = (self.sample_batch_size, 3, self.img_size, self.img_size)
            # predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
            # corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
            # snr = 0.16 #@param {"type": "number"}
            # n_steps =  1#@param {"type": "integer"}
            # probability_flow = False #@param {"type": "boolean"}
            # # inverse_scaler = datasets.get_data_inverse_scaler(config)
            # sampling_fn = sampling.get_pc_sampler(self.sde, shape, predictor, corrector,
            #                                     self.inverse_scaler, snr, n_steps=n_steps,
            #                                     probability_flow=probability_flow,
            #                                     continuous=True,
            #                                     eps=1e-5, device=self.device)

            # x, n = sampling_fn(self.generator)
            # print(x.max(), x.min())
            # exit(-1)
            # Offline sampling process, loading from replay buffer
            return self.data_iter.next()