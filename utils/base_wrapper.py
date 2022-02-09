import torch
import numpy as np
from abc import ABC
from os.path import dirname

class BaseModel(ABC):
    """ Superclass model wrapper handling input feeding to model, loss calculation, validation metrics."""
    def __init__(self, opt, log_file):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.model_names = []
        # Seed and CUDA
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.opt.device = self.device
        self.save_dir = dirname(log_file)
        posDistThr = 25
        self.posDistSqThr = posDistThr * posDistThr

    def set_input(self, batch):
        sate_ims = batch['satellite']
        pano_ims = batch['street']
        
        if 'polar' in batch:
            polar_ims = batch['polar'] if (self.opt.polar is False) else None
        else:
            polar_ims = None

        # this is necessary because of the different batch sizes of the two image sets in the VIGOR validation code
        if sate_ims is not None:
            self.satellite = sate_ims.to(self.device)
        else:
            self.satellite = None

        # this is necessary because of the different batch sizes of the two image sets in the VIGOR validation code
        if pano_ims is not None:
            self.street = pano_ims.to(self.device)
        else:
            self.street = None

        if polar_ims is not None:
            self.polar = polar_ims.to(self.device)
        else:
            self.polar = None


    def mutual_topk_acc(self, dists, topk=1):
        pos_dists = torch.diag(dists)
        N = len(pos_dists)
        # Distances smaller than positive pair
        dist_s2p = pos_dists.unsqueeze(1) - dists
        dist_p2s = pos_dists - dists

        acc_s2p = 100.0 * ((dist_s2p > 0).sum(1) < topk).sum().float() / N
        acc_p2s = 100.0 * ((dist_p2s > 0).sum(0) < topk).sum().float() / N
        return acc_p2s.item(), acc_s2p.item()

    def soft_margin_triplet_loss(self, sate_vecs, pano_vecs, loss_weight=10, hard_topk_ratio=1.0):
        dists = 2 - 2 * torch.matmul(sate_vecs, pano_vecs.permute(1, 0))  # Pairwise matches within batch
        pos_dists = torch.diag(dists)
        N = len(pos_dists)
        diag_ids = np.arange(N)
        num_hard_triplets = int(hard_topk_ratio * (N * (N - 1))) if hard_topk_ratio < 1.0 else N * (N - 1)

        # Match from satellite to street pano
        triplet_dist_s2p = pos_dists.unsqueeze(1) - dists
        loss_s2p = torch.log(1 + torch.exp(loss_weight * triplet_dist_s2p))
        loss_s2p[diag_ids, diag_ids] = 0  # Ignore diagnal losses

        if hard_topk_ratio < 1.0:  # Hard negative mining
            loss_s2p = loss_s2p.view(-1)
            loss_s2p, s2p_ids = torch.topk(loss_s2p, num_hard_triplets)
        loss_s2p = loss_s2p.sum() / num_hard_triplets

        # Match from street pano to satellite
        triplet_dist_p2s = pos_dists - dists
        loss_p2s = torch.log(1 + torch.exp(loss_weight * triplet_dist_p2s))
        loss_p2s[diag_ids, diag_ids] = 0  # Ignore diagnal losses

        if hard_topk_ratio < 1.0:  # Hard negative mining
            loss_p2s = loss_p2s.view(-1)
            loss_p2s, p2s_ids = torch.topk(loss_p2s, num_hard_triplets)
        loss_p2s = loss_p2s.sum() / num_hard_triplets
        # Total loss
        loss = (loss_s2p + loss_p2s) / 2.0
        return loss

    def load_weights(self, weights_dir, device, key='state_dict'):
        map_location = lambda storage, loc: storage.cuda(device.index) if torch.cuda.is_available() else storage
        weights_dict = None
        if weights_dir is not None:
            weights_dict = torch.load(weights_dir, map_location=map_location)
        return weights_dict

    def set_requires_grad(self, nets, requires_grad=False):

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def validate_top_VIGOR(self, dist_array, dataloader):
        
        accuracy = 0.0
        accuracy_top1 = 0.0
        accuracy_top5 = 0.0
        accuracy_top10 = 0.0
        accuracy_hit = 0.0

        data_amount = 0.0
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