import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)

class Generator(nn.Module): # Used for tiny resnet or wider-resnet, mainly for cifar10
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),  
        )

    def forward(self, z, l=0):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        if l == 0:
            img = self.conv_blocks(out)
        else:
            img = self.conv_blocks[:-(4*l-1)](out)
        return img


class LargeGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(LargeGenerator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 4 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 4),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*4, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),  
        )

    def forward(self, z, l=0):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        if l == 0:
            img = self.conv_blocks(out)
        else:
            img = self.conv_blocks[:-(4*l-1)](out)
        # img = self.conv_blocks(out)
        return img
        

class DCGAN_Generator(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """
    def __init__(self, nz=100, ngf=64, nc=3, img_size=64, slope=0.2):
        super(DCGAN_Generator, self).__init__()
        self.nz = nz
        if isinstance(img_size, (list, tuple)):
            self.init_size = ( img_size[0]//16, img_size[1]//16 )
        else:    
            self.init_size = ( img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, ngf*8*self.init_size[0]*self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf*8),
            # cifar10, 1024x2x2
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x, 512x4x4

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x, 256x8x8
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x, 128x16x16

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x, 64x32x32, alternate output channel from ngf to ngf // 2, to match the feature map after first layer.

            nn.Conv2d(ngf, nc, 3, 1,1),
            nn.Sigmoid(),
            #nn.Sigmoid()
        )

    def forward(self, z, l=0):
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])

        if l == 0:
            output = self.main(proj)
        else:
            output = self.main[:-(3*l)](proj)
        return output

class DCGAN_Generator_CIFAR10(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """
    def __init__(self, nz=100, ngf=128, nc=3, img_size=64, slope=0.2, d=2):
        super(DCGAN_Generator_CIFAR10, self).__init__()
        self.nz = nz
        depth_factor = 2 ** d
        assert d in [2, 3]
        if isinstance(img_size, (list, tuple)):
            self.init_size = ( img_size[0]//depth_factor, img_size[1]//depth_factor )
        else:    
            self.init_size = ( img_size // depth_factor, img_size // depth_factor)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, ngf*depth_factor*self.init_size[0]*self.init_size[1]),
        )
        main_module = [nn.BatchNorm2d(ngf * depth_factor), nn.Upsample(scale_factor=2)]
        self.trans_convs = []
        for i in range(d - 1):
            main_module += [
                nn.Conv2d(ngf * depth_factor, ngf * depth_factor // 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ngf*depth_factor // 2),
                nn.LeakyReLU(slope, inplace=True),
                nn.Upsample(scale_factor=2)
            ]
            # Align the feature map channels with resnet.
            self.trans_convs.append(nn.Conv2d(ngf * depth_factor // 2, 64 * depth_factor, 1, 1, 0))
            # self.trans_convs.append()
            depth_factor = depth_factor // 2
        main_module += [
            nn.Conv2d(ngf*2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(ngf, nc, 3, 1,1),
            nn.Sigmoid(),
        ]
        self.main = nn.Sequential(*main_module)
        # self.main = nn.Sequential(
        #     nn.BatchNorm2d(ngf*8),
        #     nn.Upsample(scale_factor=2),
        #     # nn.ConvTranspose2d(nz, ngf*8, 4, 2, 1, bias=False),
        #     # nn.BatchNorm2d(ngf*8),
        #     # nn.LeakyReLU(slope, inplace=True),
            
        #     # # cifar10, 1024x2x2
        #     # nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
        #     # nn.BatchNorm2d(ngf*4),
        #     # nn.LeakyReLU(slope, inplace=True),
        #     # # 2x, 512x4x4

        #     nn.Conv2d(ngf*8, ngf*4, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(ngf*4),
        #     nn.LeakyReLU(slope, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     # # 4x, 256x8x8
            
        #     nn.Conv2d(ngf*4, ngf*2, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(ngf*2),
        #     nn.LeakyReLU(slope, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     # 8x, 128x16x16

        #     nn.Conv2d(ngf*2, ngf, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.LeakyReLU(slope, inplace=True),
        #     # nn.Upsample(scale_factor=2),
        #     # 16x, 64x32x32, alternate output channel from ngf to ngf // 2, to match the feature map after first layer.

        #     nn.Conv2d(ngf, nc, 3, 1,1),
        #     nn.Sigmoid(),
        #     #nn.Sigmoid()
        # )
        

    def forward(self, z, l=0):
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        # proj = z.unsqueeze(-1).unsqueeze(-1)

        if l == 0:
            output = self.main(proj)
        else:
            output = self.main[:-(4*l+1)](proj)
        return output

class DCGAN_CondGenerator(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """
    def __init__(self, num_classes,  nz=100, n_emb=50, ngf=64, nc=3, img_size=64, slope=0.2):
        super(DCGAN_CondGenerator, self).__init__()
        self.nz = nz
        self.emb = nn.Embedding(num_classes, n_emb)
        if isinstance(img_size, (list, tuple)):
            self.init_size = ( img_size[0]//16, img_size[1]//16 )
        else:    
            self.init_size = ( img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz+n_emb, ngf*8*self.init_size[0]*self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf*8),
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x

            nn.Conv2d(ngf, nc, 3, 1,1),
            #nn.Tanh(),
            nn.Sigmoid()
        )

    def forward(self, z, y):
        y = self.emb(y)
        z = torch.cat([z, y], dim=1)
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output

class Discriminator(nn.Module):
    def __init__(self, nc=3, img_size=32):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(nc, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

class DCGAN_Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(DCGAN_Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)