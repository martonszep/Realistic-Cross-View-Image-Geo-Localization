import os
from utils.base_wrapper import BaseModel
from networks.c_gan import *
import numpy as np

class RGANWrapper(BaseModel):

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

