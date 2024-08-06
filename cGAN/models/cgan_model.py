import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from collections import OrderedDict

class CGANModel(BaseModel):
    def name(self):
        return 'cGAN'

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses to print out during training
        self.loss_names = ['D_real', 'D_fake', 'G_GAN', 'G_L1']
        # specify the images to save/display during training
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        self.threshold = opt.D_loss_threshold

        # specify the models to save to the disk
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Generator
            self.model_names = ['G']

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, 
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_type).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers= [self.optimizer_G, self.optimizer_D]

            self.schedulers = []
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            load_suffix = opt.load_epoch if opt.load_epoch > 0 else opt.which_epoch
            self.load_networks(load_suffix)

        self.print_networks(opt.verbose)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
        self.real_A = input_A
        self.real_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def test(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1) # need to feed both input and output for cGAN
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward() # calculating gradients

    def optimize_parameters(self):
        # compute fake images
        self.forward()              
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()             # set D's gradients to zero
        self.backward_D()                        # calculate gradients for D
        if self.loss_D >= self.threshold:
            self.optimizer_D.step()              # update D's weights if its loss above threshold

        # update G
        self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

    def get_current_errors(self): # get current losses
        D_real = self.loss_D_real.data.item()
        D_fake = self.loss_D_fake.data.item()
        G_GAN = self.loss_G_GAN.data.item()
        G_L1 = self.loss_G_L1.data.item()
        return OrderedDict([('D_real', D_real), ('D_fake', D_fake), ('G_GAN', G_GAN), ('G_L1', G_L1)])

