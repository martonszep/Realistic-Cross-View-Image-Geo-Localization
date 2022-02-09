from data.custom_transforms import *
from data.cvusa_utils import CVUSA, convert_image_np
from torch.utils.tensorboard import SummaryWriter
from utils import model_wrapper, base_wrapper, parser
from utils.setup_helper import *
import time
from argparse import Namespace
import os

if __name__ == '__main__':
    parse = parser.Parser()
    opt, log_file = parse.parse()
    opt.is_Train = True
    make_deterministic(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpu_ids)
    log = open(log_file, 'a')
    log_print = lambda ms: parse.log(ms, log) # txt logger
    writer = SummaryWriter(log_dir=log_file.replace('log.txt', '')) # tensorboard logger

    #define networks
    retrieval = model_wrapper.define_R(ret_method=opt.r_model, polar=opt.polar, gpu_ids=opt.gpu_ids)
    log_print('Init {} as retrieval model'.format(opt.r_model))

    model_wrapper = model_wrapper.ModelWrapper(opt, log_file, retrieval)
    total_params = sum(p.numel() for p in model_wrapper.retrieval.parameters() if p.requires_grad)
    log_print('No. of trainable parameters: {}'.format(total_params))

    # Configure data loader
    composed_transforms = transforms.Compose([RandomHorizontalFlip(),
                                                ToTensor()])
    train_dataset = CVUSA(root=opt.data_root, csv_file=opt.train_csv, use_polar=opt.polar, name=opt.name,
                        transform_op=composed_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    val_dataset = CVUSA(root=opt.data_root, csv_file=opt.val_csv, use_polar=opt.polar, name=opt.name,
                        transform_op=ToTensor())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    log_print('Load datasets from {}: train_set={} val_set={}'.format(opt.data_root, len(train_dataset), len(val_dataset)))

    ret_best_acc = model_wrapper.ret_best_acc
    log_print('Start training from epoch {} to {}, best acc: {}'.format(opt.start_epoch, opt.n_epochs, ret_best_acc))
    for epoch in range(opt.start_epoch, opt.n_epochs):
        start_time = time.time()
        batches_done = 0
        val_batches_done = 0
        street_batches_t = []
        fake_street_batches_t = []
        epoch_retrieval_loss_t = []
        epoch_l1_loss_t = []
        street_batches_v = []
        fake_street_batches_v = []
        epoch_retrieval_loss_v= []
        epoch_l1_loss_v = []

        log_print('>>> RGAN Epoch {}'.format(epoch))
        model_wrapper.retrieval.train()
        for i, data in enumerate(train_loader):  # inner loop within one epoch
            if i == 0:
                start_time_batches = time.time()

            model_wrapper.set_input(data)
            model_wrapper.optimize_parameters(epoch)

            fake_street_batches_t.append(model_wrapper.fake_street_out.cpu().data)
            street_batches_t.append(model_wrapper.street_out.cpu().data)
            epoch_retrieval_loss_t.append(model_wrapper.r_loss.item())
            epoch_l1_loss_t.append(model_wrapper.l1_loss.item()) if opt.polar is False else None

            if (i + 1) % 40 == 0 or (i + 1) == len(train_loader):
                fake_street_vec = torch.cat(fake_street_batches_t, dim=0)
                street_vec = torch.cat(street_batches_t, dim=0)
                dists = 2 - 2 * torch.matmul(fake_street_vec, street_vec.permute(1, 0))
                tp1 = model_wrapper.mutual_topk_acc(dists, topk=1)
                tp5 = model_wrapper.mutual_topk_acc(dists, topk=5)
                tp10 = model_wrapper.mutual_topk_acc(dists, topk=10)
                log_print('Batch:{} loss={:.3f} samples:{} tp1={tp1[0]:.2f}/{tp1[1]:.2f} ' \
                        'tp5={tp5[0]:.2f}/{tp5[1]:.2f} Time:{time:.2f}s'.format(i + 1, np.mean(epoch_retrieval_loss_t),
                                                len(dists), tp1=tp1, tp5=tp5, time=time.time() - start_time_batches))
                start_time_batches = time.time()
                street_batches_t.clear()
                fake_street_batches_t.clear()
        
        writer.add_scalar('Loss/Train', np.mean(epoch_retrieval_loss_t), epoch) # tensorboard logging
        writer.add_scalar('Loss/L1_train', np.mean(epoch_l1_loss_t), epoch) if opt.polar is False else None

        model_wrapper.save_networks(epoch, os.path.dirname(log_file), best_acc=ret_best_acc,
                                        last_ckpt=True)  # Always save last ckpt

        # validation loop
        model_wrapper.retrieval.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                model_wrapper.set_input(data)
                model_wrapper.eval_model()
                fake_street_batches_v.append(model_wrapper.fake_street_out_val.cpu().data)
                street_batches_v.append(model_wrapper.street_out_val.cpu().data)
                epoch_retrieval_loss_v.append(model_wrapper.r_loss.item())
                epoch_l1_loss_v.append(model_wrapper.l1_loss.item()) if opt.polar is False else None
        
        # validation metric calculation
        fake_street_vec = torch.cat(fake_street_batches_v, dim=0)
        street_vec = torch.cat(street_batches_v, dim=0)
        dists = 2 - 2 * torch.matmul(fake_street_vec, street_vec.permute(1, 0))
        tp1 = model_wrapper.mutual_topk_acc(dists, topk=1)
        tp5 = model_wrapper.mutual_topk_acc(dists, topk=5)
        tp10 = model_wrapper.mutual_topk_acc(dists, topk=10)
        street_batches_v.clear()
        fake_street_batches_v.clear()

        num = len(dists)
        tp1p = model_wrapper.mutual_topk_acc(dists, topk=0.01 * num)
        acc = Namespace(num=len(dists), tp1=tp1, tp5=tp5, tp10=tp10, tp1p=tp1p)

        log_print('\nEvaluate Samples:{num:d}\nRecall(p2s/s2p) tp1:{tp1[0]:.2f}/{tp1[1]:.2f} ' \
                    'tp5:{tp5[0]:.2f}/{tp5[1]:.2f} tp10:{tp10[0]:.2f}/{tp10[1]:.2f} ' \
                    'tp1%:{tp1p[0]:.2f}/{tp1p[1]:.2f}'.format(epoch + 1, num=acc.num, tp1=acc.tp1,
                                                            tp5=acc.tp5, tp10=acc.tp10, tp1p=acc.tp1p))
        
        # tensorboard eval logging
        writer.add_scalar('Loss/Val', np.mean(epoch_retrieval_loss_v), epoch)
        writer.add_scalar('Loss/L1_val', np.mean(epoch_l1_loss_v), epoch) if opt.polar is False else None
        writer.add_scalar('Recall/R@1', tp1[0], epoch)
        writer.add_scalar('Recall/R@5', tp5[0], epoch)
        writer.add_scalar('Recall/R@10', tp10[0], epoch)
        writer.add_scalar('Recall/R@1%', tp1p[0], epoch)

        # Save the best model
        tp1_p2s_acc = acc.tp1[0]
        if tp1_p2s_acc > ret_best_acc:
            ret_best_acc = tp1_p2s_acc
            model_wrapper.save_networks(epoch, os.path.dirname(log_file), best_acc=ret_best_acc, is_best=True)
            log_print('>>Save best model: epoch={} best_acc(tp1_p2s):{:.2f}'.format(epoch + 1, tp1_p2s_acc))

        # Program statistics
        rss, vms, cpu_mem_percent_os, cpu_av_mem_os, cpu_mem_percent, cpu_available_mem_percent = get_sys_mem()
        log_print('Memory usage: rss={:.2f}GB vms={:.2f}GB Time:{:.2f}s\n'.format(rss, vms, time.time() - start_time))

        # Visualize the STN transformation on some input batch
        if opt.polar is False and (epoch+1) % 5 == 0:
            with torch.no_grad():
                images = next(iter(val_loader))['satellite'][:8].to(model_wrapper.device)
                transformed_images = model_wrapper.retrieval.module.spatial_tr(images)
                writer.add_images(f"Spatial Transformer/Inputs", convert_image_np(images.cpu()), dataformats='NHWC')
                writer.add_images(f"Spatial Transformer/Outputs", convert_image_np(transformed_images.cpu()), global_step=epoch,  dataformats='NHWC')
    writer.close()