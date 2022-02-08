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
            ckpt_name = 'rgan_last_ckpt.pth'
        elif is_best:
            ckpt_name = 'rgan_best_ckpt.pth'
        else:
            ckpt_name = 'rgan_ckpt_ep{}.pth'.format(epoch + 1)
        ckpt_path = os.path.join(out_dir, ckpt_name)
        torch.save(ckpt, ckpt_path)

    def load_networks(self):
        if self.opt.rgan_checkpoint is None:
            return

        ckpt_path = self.opt.rgan_checkpoint
        ckpt = torch.load(ckpt_path)

        self.opt.start_epoch = ckpt['last_epoch'] + 1
        self.ret_best_acc = ckpt['best_acc']

        # Load net state
        retrieval_dict = ckpt['retriebal_model_dict']

        self.retrieval.load_state_dict(retrieval_dict)
        self.optimizer_R.load_state_dict(ckpt['optimizer_R_dict'])


#****************** this need not be necessarily here, we don't use anything from self
    def validate_top_VIGOR(self, dist_array, dataloader):
        # grd_descriptor = grd_descriptor.numpy() # torch tensors will not work with np.sum() on booleans, original code had tf tensors here
        # sat_descriptor = sat_descriptor.numpy() # torch tensors will not work with np.sum() on booleans, original code had tf tensors here
        accuracy = 0.0
        accuracy_top1 = 0.0
        accuracy_top5 = 0.0
        accuracy_top10 = 0.0
        accuracy_hit = 0.0

        data_amount = 0.0
        # dist_array = 2 - 2 * np.matmul(grd_descriptor, np.transpose(sat_descriptor))
        top1_percent = int(dist_array.shape[1] * 0.01) + 1
        top1 = 1
        top5 = 5
        top10 = 10

        for i in range(dist_array.shape[0]):

            gt_dist = dist_array[i, dataloader.test_label[i][0]]
            prediction = np.sum(dist_array[i, :] < gt_dist)
            dist_temp = np.ones(dist_array[i, :].shape[0])
            dist_temp[dataloader.test_label[i][1:]] = 0
            prediction_hit = np.sum((dist_array[i, :] < gt_dist) * dist_temp)

            if prediction < top1_percent:
                accuracy += 1.0
            if prediction < top1:
                accuracy_top1 += 1.0
            if prediction < top5:
                accuracy_top5 += 1.0
            if prediction < top10:
                accuracy_top10 += 1.0
            if prediction_hit < top1:
                accuracy_hit += 1.0
            data_amount += 1.0
        accuracy /= data_amount
        accuracy_top1 /= data_amount
        accuracy_top5 /= data_amount
        accuracy_top10 /= data_amount
        accuracy_hit /= data_amount
        return accuracy, accuracy_top1, accuracy_top5, accuracy_top10, accuracy_hit
