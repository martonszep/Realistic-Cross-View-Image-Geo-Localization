import os
from utils.base_wrapper import BaseModel
import numpy as np
import torch
from torch.nn import init
from networks import safa

def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_R(ret_method, polar, gpu_ids=[]):
    net = None
    if 'SAFA' == ret_method:
        sa_num = 8
        # VIGOR
        # sate_size = (160, 160) # 320, 320
        # pano_size = (160, 320) # 320, 640
        # CVUSA
        sate_size = (112, 616) #if polar else (256, 256) # this determines the output size of the spatial transformer if polar False
        pano_size = (112, 616)
        net = safa.SAFA(sa_num=sa_num, H1=pano_size[0], W1=pano_size[1], H2=sate_size[0], W2=sate_size[1], use_spatialtr=not polar)
    else:
        raise NotImplementedError('Retrieval model name [%s] is not recognized' % ret_method)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    return net

class ModelWrapper(BaseModel):

    def __init__(self, opt, log_file, net_R):
        BaseModel.__init__(self, opt, log_file)
        self.optimizers = []
        self.ret_best_acc = 0.0
        
        #initialize retrieval
        self.retrieval = net_R
        self.criterion_l1 = torch.nn.L1Loss()
        self.optimizer_R = torch.optim.Adam(self.retrieval.parameters(), lr=opt.lr_r, betas=(opt.b1, opt.b2))

        self.optimizers.append(self.optimizer_R)
        self.load_networks()

    def forward(self):
        None

    def backward_R(self, epoch):

        self.fake_street_out, self.street_out = self.retrieval(self.street, self.satellite)
        self.fake_street = self.retrieval.module.transformed_satellite if (self.retrieval.module.spatial_tr is not None) else None
        # self.r_loss = self.soft_margin_triplet_loss(self.fake_street_out, self.street_out, loss_weight=self.opt.lambda_sm, hard_topk_ratio=self.opt.hard_topk_ratio)

        # Scheduled hard negative mining
        if epoch <= 20:
            self.r_loss = self.soft_margin_triplet_loss(self.fake_street_out, self.street_out, loss_weight=self.opt.lambda_sm, hard_topk_ratio=self.opt.hard_topk_ratio)
        elif epoch > 20 and epoch <=40:
            self.r_loss = self.soft_margin_triplet_loss(self.fake_street_out, self.street_out, loss_weight=self.opt.lambda_sm, hard_topk_ratio=self.opt.hard_decay1_topk_ratio)
        elif epoch > 40 and epoch <=60:
            self.r_loss = self.soft_margin_triplet_loss(self.fake_street_out, self.street_out, loss_weight=self.opt.lambda_sm, hard_topk_ratio=self.opt.hard_decay2_topk_ratio)
        elif epoch > 60:
            self.r_loss = self.soft_margin_triplet_loss(self.fake_street_out, self.street_out, loss_weight=self.opt.lambda_sm, hard_topk_ratio=self.opt.hard_decay3_topk_ratio)
        
        if (self.fake_street is not None):
            self.l1_loss = self.criterion_l1(self.fake_street, self.polar)
            self.total_loss = self.r_loss * self.opt.lambda_ret1 +  self.l1_loss * self.opt.lambda_l1
        else:
            self.total_loss = self.r_loss
        self.total_loss.backward()

    def optimize_parameters(self, epoch):
        self.forward()

        # update retrieval network
        self.set_requires_grad(self.retrieval, True)
        self.optimizer_R.zero_grad()
        self.backward_R(epoch)
        self.optimizer_R.step()


    def eval_model(self):
        self.forward()
        self.fake_street_out_val, self.street_out_val = self.retrieval(self.street, self.satellite)


    def save_networks(self, epoch, out_dir, last_ckpt=False, best_acc=None, is_best=False):
        ckpt = {'last_epoch': epoch,
                'best_acc': best_acc,
                'retriebal_model_dict': self.retrieval.state_dict(),
                 'optimizer_R_dict': self.optimizer_R.state_dict(),
                }

        if last_ckpt:
            ckpt_name = 'last_ckpt.pth'
        elif is_best:
            ckpt_name = 'best_ckpt.pth'
        else:
            ckpt_name = 'ckpt_ep{}.pth'.format(epoch + 1)
        ckpt_path = os.path.join(out_dir, ckpt_name)
        torch.save(ckpt, ckpt_path)

    def load_networks(self):
        if self.opt.checkpoint is None:
            return

        ckpt_path = self.opt.checkpoint
        ckpt = torch.load(ckpt_path)

        self.opt.start_epoch = ckpt['last_epoch'] + 1
        self.ret_best_acc = ckpt['best_acc']

        # Load net state
        retrieval_dict = ckpt['retriebal_model_dict']

        self.retrieval.load_state_dict(retrieval_dict)
        self.optimizer_R.load_state_dict(ckpt['optimizer_R_dict'])

