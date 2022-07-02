import torch

from .base import BaseSynthesis
from guided_diffusion import dist_util, logger

class PretrainedGenerativeSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, img_size, synthesis_batch_size=128, sample_batch_size=128, normalizer=None, device='cpu', mode='gan', use_ddim=False):
        super(PretrainedGenerativeSynthesizer, self).__init__(teacher, student)
        self.mode = mode
        # assert len(img_size)==3, "image size should be a 3-dimension tuple"
        self.img_size = img_size 
        
        self.nz = nz
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        self.generator = generator
        if self.mode == 'diffusion':
            self.generator, self.diffusion = generator
        self.device = device
        self.use_ddim = use_ddim

    def synthesize(self):
        raise NotImplementedError('Should not update the generator')
    
    @torch.no_grad()
    def sample(self):
        G = self.generator
        if self.mode == 'glow':
            self.generator.set_actnorm_init()
        # elif self.mode == 'diffusion':
        #     print(type(self.generator))
        #     self.generator, self.diffusion = self.generator
        self.generator = self.generator.to(self.device)
        self.generator.eval()
        if self.mode == 'gan':
            z = torch.randn( size=(self.sample_batch_size, self.nz), device=self.device )
            inputs = self.generator(z)
            # print(inputs.min(), inputs.max())
            # exit(-1)
            return inputs
        elif self.mode == 'glow':
            x_intermideate = self.generator(temperature=1, reverse=True)
            

            # output = torch.sigmoid(x_intermideate)
            # if torch.isnan(output.mean()):
            #     print(output, x_intermideate)
            #     exit(-1)
            output = x_intermideate.clamp_(-1, 1)
            output = output / 2 + 0.5
            # if torch.isnan(output.mean()):
            #     print(x_intermideate)
            #     exit(-1)
            # print(output.min(), output.max())
            # exit(-1)
            return output
        elif self.mode == 'diffusion':
            sample_fn = (
                self.diffusion.p_sample_loop if not self.use_ddim else self.diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                self.generator,
                (self.sample_batch_size, 3, self.img_size, self.img_size),
                clip_denoised=True,
                model_kwargs={}
            )
            sample = (sample + 1) / 2
            sample = torch.clamp(sample, 0, 1)
            return sample