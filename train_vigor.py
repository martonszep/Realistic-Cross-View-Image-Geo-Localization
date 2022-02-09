from data.custom_transforms import *
from data.vigor_dataloader import DataLoader, convert_image_np_VIGOR
from torch.utils.tensorboard import SummaryWriter
from utils import model_wrapper, parser
from utils.setup_helper import *
import time
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
    retrieval = model_wrapper.define_R(ret_method=opt.r_model, polar=opt.polar, gpu_ids=opt.gpu_ids, 
                            sate_size=opt.sate_size, pano_size=opt.pano_size, use_tps=not opt.use_affine)

    log_print('Init {} as retrieval model'.format(opt.r_model))

    model_wrapper = model_wrapper.ModelWrapper(opt, log_file, retrieval)
    total_params = sum(p.numel() for p in model_wrapper.retrieval.parameters() if p.requires_grad)
    log_print('No. of trainable parameters: {}'.format(total_params))

    # Configure data loader
    composed_transforms = transforms.Compose([RandomHorizontalFlip(),
                                                ToTensorVIGOR()])

    dataloader = DataLoader(mode=opt.vigor_mode, root=opt.data_root, dim=opt.vigor_dim, 
                                same_area=True if 'same' in opt.vigor_mode else False, if_polar=opt.polar, logger=log_print)

    break_iter = int(dataloader.train_data_size / opt.batch_size)

    ret_best_acc = model_wrapper.ret_best_acc
    log_print('Start training from epoch {} to {}, best acc: {}'.format(opt.start_epoch, opt.n_epochs, ret_best_acc))
    for epoch in range(opt.start_epoch, opt.n_epochs):
        start_time = time.time()
        batches_done = 0
        val_batches_done = 0
        street_batches_t = []
        fake_street_batches_t = []
        epoch_retrieval_loss_t = []
        street_batches_v = []
        fake_street_batches_v = []
        epoch_retrieval_loss_v= []
        
        log_print('>>> RGAN Epoch {}'.format(epoch))
       
        model_wrapper.retrieval.train()
        iter = 0
        while True:
            if iter == 0:
                start_time_batches = time.time()

            # vigor dataloader patched into our training pipeline

            # we get here: batch_sat -> [B, H, W, 3], batch_grd -> [B, H, W, 3]
            batch_sat, batch_polar, batch_grd, _, _ = dataloader.get_next_batch(opt.batch_size)
            
            composed_data = {
                "satellite": None,
                "street": None,
                "polar": None
            }
            for i, elem in enumerate(batch_grd):
                data_to_be_transformed = {
                    "satellite": batch_sat[i],
                    "street": batch_grd[i],
                    "polar": batch_polar[i]
                }
                transformed_data = composed_transforms(data_to_be_transformed)
                if i == 0:
                    composed_data["satellite"] = transformed_data["satellite"].unsqueeze(0)
                    composed_data["street"] = transformed_data["street"].unsqueeze(0)
                    composed_data["polar"] = transformed_data["polar"].unsqueeze(0)
                else:
                    composed_data["satellite"] = torch.cat((composed_data["satellite"],  transformed_data["satellite"].unsqueeze(0)), 0)
                    composed_data["street"] = torch.cat((composed_data["street"],  transformed_data["street"].unsqueeze(0)), 0)
                    composed_data["polar"] = torch.cat((composed_data["polar"],  transformed_data["polar"].unsqueeze(0)), 0)
            
            # if next batch does not contain any more images or if we reach the last batch then training of an epoch is done
            if batch_grd is None or (iter+1) == break_iter or (iter+1) == 2:
                break
            
            # we transformed the data into: composed_data -> {"satellite": [B, 3, H, W], "street": [B, 3, H, W]}
            model_wrapper.set_input(composed_data)
            model_wrapper.optimize_parameters(epoch)

            fake_street_batches_t.append(model_wrapper.fake_street_out.cpu().data)
            street_batches_t.append(model_wrapper.street_out.cpu().data)
            epoch_retrieval_loss_t.append(model_wrapper.r_loss.item())
          
            if (iter + 1) % 40 == 0 or (iter + 2) == break_iter:
                fake_street_vec = torch.cat(fake_street_batches_t, dim=0)
                street_vec = torch.cat(street_batches_t, dim=0)
                dists = 2 - 2 * torch.matmul(fake_street_vec, street_vec.permute(1, 0))
                tp1 = model_wrapper.mutual_topk_acc(dists, topk=1)
                tp5 = model_wrapper.mutual_topk_acc(dists, topk=5)
                tp10 = model_wrapper.mutual_topk_acc(dists, topk=10)
                log_print('Batch:{} loss={:.3f} samples:{} tp1={tp1[0]:.2f}/{tp1[1]:.2f} ' \
                        'tp5={tp5[0]:.2f}/{tp5[1]:.2f} Time:{time:.2f}s'.format(iter + 1, np.mean(epoch_retrieval_loss_t),
                                                len(dists), tp1=tp1, tp5=tp5, time=time.time() - start_time_batches))
                start_time_batches = time.time()
                street_batches_t.clear()
                fake_street_batches_t.clear()

            iter+=1
        
        writer.add_scalar('Loss/Train', np.mean(epoch_retrieval_loss_t), epoch)

        model_wrapper.save_networks(epoch, os.path.dirname(log_file), best_acc=ret_best_acc,
                                        last_ckpt=True)  # Always save last ckpt

    # validation phase
    # this is very time consuming, so maybe don't execute it at the end of each epoch
        model_wrapper.retrieval.eval()
        with torch.no_grad():
            val_i = 0
            out = False
            while True:
                if val_i == 0:
                    start_time_batches = time.time()

                # we get here: batch_sat -> [B, H, W, 3], batch_grd -> [B, H, W, 3]
                batch_sat = dataloader.next_sat_scan(opt.batch_size*3)
                batch_grd = None

                 # this dataloader will run out first, so we are making sure, that it does not start over once its out
                if not out:
                    batch_grd = dataloader.next_grd_scan(opt.batch_size*2)
                
                if batch_grd is None:
                    out = True
                    
                # terminating the validation loop once the satellite images are out too
                # as we have evaluated all images and we can now compute the metrics on them
                if batch_sat is None: 
                    break
                
                composed_data = {
                    "satellite": None,
                    "street": None
                }

                # check which batch has more images in it
                loop_range = 0
                if batch_sat is None:
                    batch_sat_len = 0
                else:
                    batch_sat_len = len(batch_sat)

                if batch_grd is None:
                    batch_grd_len = 0
                else:
                    batch_grd_len = len(batch_grd)


                if batch_sat_len > batch_grd_len:
                    loop_range = batch_sat_len
                else:
                    loop_range = batch_grd_len

                for i in range(loop_range):

                    # check if there is any street image left in the batch
                    if batch_grd_len <= i:
                        current_grd = None
                    else:
                        current_grd = batch_grd[i]

                    # check if there is any sat image left in the batch
                    if batch_sat_len <= i:
                        current_sat = None
                    else:
                        current_sat = batch_sat[i]

                    data_to_be_transformed = {
                        "satellite": current_sat,
                        "street": current_grd
                    }
                    transformed_data = composed_transforms(data_to_be_transformed)
                    if i == 0:
                        if transformed_data["satellite"] is not None:
                            composed_data["satellite"] = transformed_data["satellite"].unsqueeze(0)
                        if transformed_data["street"] is not None:
                            composed_data["street"] = transformed_data["street"].unsqueeze(0)
                    else:
                        if transformed_data["satellite"] is not None:
                            composed_data["satellite"] = torch.cat((composed_data["satellite"],  transformed_data["satellite"].unsqueeze(0)), 0)
                        if transformed_data["street"] is not None:
                            composed_data["street"] = torch.cat((composed_data["street"],  transformed_data["street"].unsqueeze(0)), 0)

                composed_data["polar"] = None # we don't use this in validation, se we just set it to None

                # saving 8 of the original and transformed satellite images to tensorboard as visualizations
                # val_i == 11 is arbitrary, you can choose a different index where you find images you like
                if val_i == 11 and opt.polar is False:
                    images = composed_data["satellite"][:8].to(model_wrapper.device)
                    transformed_images = model_wrapper.retrieval.module.spatial_tr(images)
                    writer.add_images(f"Spatial Transformer/val_Inputs", convert_image_np_VIGOR(images.cpu()), dataformats='NHWC')
                    writer.add_images(f"Spatial Transformer/val_Outputs", convert_image_np_VIGOR(transformed_images.cpu()), global_step=epoch, dataformats='NHWC')

                # we transformed the data into: composed_data -> {"satellite": [B, 3, H, W], "street": [B, 3, H, W]}
                model_wrapper.set_input(composed_data)
                model_wrapper.eval_model()

                new_fake_street = model_wrapper.fake_street_out_val
                new_street = model_wrapper.street_out_val

                if new_fake_street is not None:
                    fake_street_batches_v.append(new_fake_street.cpu().data)
                if new_street is not None:
                    street_batches_v.append(new_street.cpu().data)

                val_i += 1

        fake_street_vec = torch.cat(fake_street_batches_v, dim=0).numpy() 
        street_vec = torch.cat(street_batches_v, dim=0).numpy()
        dist_array = 2 - 2 * np.matmul(street_vec, np.transpose(fake_street_vec))

        val_accuracy, val_accuracy_top1, val_accuracy_top5, val_accuracy_top10, hit_rate = model_wrapper.validate_top_VIGOR(dist_array, dataloader)
        
        street_batches_v.clear()
        fake_street_batches_v.clear()

        log_print('Evaluation epoch %d: accuracy = %.1f%% , top1: %.1f%%, top5: %.1f%%, top10: %.1f%%, hit_rate: %.1f%%' % (
            epoch, val_accuracy * 100.0, val_accuracy_top1 * 100.0, val_accuracy_top5 * 100.0, val_accuracy_top10 * 100.0, hit_rate * 100.0))
        
        # tensorboard eval logging
        # writer.add_scalar('Loss/Val', np.mean(epoch_retrieval_loss_v), epoch)
        writer.add_scalar('Recall/R@1', val_accuracy_top1*100, epoch)
        writer.add_scalar('Recall/R@5', val_accuracy_top5*100, epoch)
        writer.add_scalar('Recall/R@10', val_accuracy_top10*100, epoch)
        writer.add_scalar('Recall/R@1%',val_accuracy * 100, epoch)

        # Save the best model
        tp1_p2s_acc = val_accuracy_top1*100
        if tp1_p2s_acc > ret_best_acc:
            ret_best_acc = tp1_p2s_acc
            model_wrapper.save_networks(epoch, os.path.dirname(log_file), best_acc=ret_best_acc, is_best=True)
            log_print('>>Save best model: epoch={} best_acc(tp1_p2s):{:.2f}'.format(epoch + 1, tp1_p2s_acc))

        # Program statistics
        rss, vms, cpu_mem_percent_os, cpu_av_mem_os, cpu_mem_percent, cpu_available_mem_percent = get_sys_mem()
        log_print('Memory usage: rss={:.2f}GB vms={:.2f}GB Time:{:.2f}s\n'.format(rss, vms, time.time() - start_time))
        log_print('os mem percent: {} available mem percent: {} \n'.format(cpu_mem_percent_os, cpu_av_mem_os))
        log_print('psutil mem percent: {} available mem percent: {} \n'.format(cpu_mem_percent, cpu_available_mem_percent))