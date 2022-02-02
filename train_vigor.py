from data.custom_transforms import *
from data.cvusa_utils import CVUSA, convert_image_np, convert_image_np_VIGOR
from data.vigor_dataloader import DataLoader
from networks.c_gan import *
from torch.utils.tensorboard import SummaryWriter
from utils import rgan_wrapper, base_wrapper, parser
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
    # generator = define_G(netG=opt.g_model, gpu_ids=opt.gpu_ids)
    # log_print('Init {} as generator model'.format(opt.g_model))

    # discriminator = define_D(input_c=opt.input_c, output_c=opt.realout_c, ndf=opt.feature_c, netD=opt.d_model,
    #                             condition=opt.condition, n_layers_D=opt.n_layers, gpu_ids=opt.gpu_ids)
    # log_print('Init {} as discriminator model'.format(opt.d_model))

    retrieval = define_R(ret_method=opt.r_model, polar=opt.polar, gpu_ids=opt.gpu_ids)
    log_print('Init {} as retrieval model'.format(opt.r_model))

    rgan_wrapper = rgan_wrapper.RGANWrapper(opt, log_file, retrieval)
    total_params = sum(p.numel() for p in rgan_wrapper.retrieval.parameters() if p.requires_grad)
    log_print('No. of trainable parameters: {}'.format(total_params))

    # Configure data loader
    composed_transforms = transforms.Compose([RandomHorizontalFlip(),
                                                ToTensorVIGOR()])
    # train_dataset = CVUSA(root=opt.data_root, csv_file=opt.train_csv, use_polar=opt.polar, name=opt.name,
    #                     transform_op=composed_transforms, load_pickle=None)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    dataloader = DataLoader(mode=opt.vigor_mode, root=opt.vigor_root, dim=opt.vigor_dim, same_area=True if 'same' in opt.vigor_mode else False, logger=log_print)
    break_iter = int(dataloader.train_data_size / opt.batch_size)

    # val_dataset = CVUSA(root=opt.data_root, csv_file=opt.val_csv, use_polar=opt.polar, name=opt.name,
    #                     transform_op=ToTensor(), load_pickle=None)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    # log_print('Load datasets from {}: train_set={} val_set={}'.format(opt.data_root, len(train_dataset), len(val_dataset)))

    # for i, data in enumerate(val_loader):
    #     print(data)
    #     print(data["street"].shape, data["satellite"].shape)
    #     break

    ret_best_acc = rgan_wrapper.ret_best_acc
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
       
        rgan_wrapper.retrieval.train()
        iter = 0
        while True:
        # for i, data in enumerate(train_loader):  # inner loop within one epoch
            if iter == 0:
                start_time_batches = time.time()

            batch_sat, batch_grd, _, _ = dataloader.get_next_batch(opt.batch_size)
            # we get here: batch_sat -> (batch_size, 320, 320, 3), batch_grd -> (batch_size, 320, 640, 3)
            composed_data = {
                "satellite": None,
                "street": None
            }
            for i, elem in enumerate(batch_grd):
                data_to_be_transformed = {
                    "satellite": batch_sat[i],
                    "street": batch_grd[i]
                }
                transformed_data = composed_transforms(data_to_be_transformed)
                if i == 0:
                    composed_data["satellite"] = transformed_data["satellite"].unsqueeze(0)
                    composed_data["street"] = transformed_data["street"].unsqueeze(0)
                else:
                    composed_data["satellite"] = torch.cat((composed_data["satellite"],  transformed_data["satellite"].unsqueeze(0)), 0)
                    composed_data["street"] = torch.cat((composed_data["street"],  transformed_data["street"].unsqueeze(0)), 0)
            
            if batch_grd is None or (iter+1) == break_iter: # should we test for batch_grd or batch_sat???
                try:
                    print("batch_grd=", batch_grd.shape)
                except:
                    print("batch_grd=", batch_grd)
                print("breaking training loop iter=", iter)

                break

            # print(composed_data["satellite"].shape, composed_data["street"].shape)
            # writer.add_images(f"VIGOR/street", convert_image_np_VIGOR(composed_data["street"].cpu()), dataformats='NHWC') # NHWC
            # writer.add_images(f"VIGOR/sat", convert_image_np_VIGOR(composed_data["satellite"].cpu()), dataformats='NHWC') # NHWC
            # writer.close()

            # raise Exception ("interrupt execution for debugging")

            # we transformed the data into
            #       composed_data -> {"satellite": torch.Size([batch_size, 3, 320, 320]), "street": torch.Size([batch_size, 3, 320, 640])}
            rgan_wrapper.set_input(composed_data)
            rgan_wrapper.optimize_parameters(epoch)

            fake_street_batches_t.append(rgan_wrapper.fake_street_out.cpu().data)
            street_batches_t.append(rgan_wrapper.street_out.cpu().data)
            epoch_retrieval_loss_t.append(rgan_wrapper.r_loss.item())
           
            # if (iter + 1) % 40 == 0 or (iter + 1) == break_iter:
            #     print("batch=", iter+1)
            #     print("time=", time.time() - start_time_batches)
            #     start_time_batches = time.time()
            if (iter + 1) % 40 == 0 or (iter + 2) == break_iter:
                fake_street_vec = torch.cat(fake_street_batches_t, dim=0)
                street_vec = torch.cat(street_batches_t, dim=0)
                dists = 2 - 2 * torch.matmul(fake_street_vec, street_vec.permute(1, 0))
                tp1 = rgan_wrapper.mutual_topk_acc(dists, topk=1)
                tp5 = rgan_wrapper.mutual_topk_acc(dists, topk=5)
                tp10 = rgan_wrapper.mutual_topk_acc(dists, topk=10)
                log_print('Batch:{} loss={:.3f} samples:{} tp1={tp1[0]:.2f}/{tp1[1]:.2f} ' \
                        'tp5={tp5[0]:.2f}/{tp5[1]:.2f} Time:{time:.2f}s'.format(iter + 1, np.mean(epoch_retrieval_loss_t),
                                                len(dists), tp1=tp1, tp5=tp5, time=time.time() - start_time_batches))
                start_time_batches = time.time()
                street_batches_t.clear()
                fake_street_batches_t.clear()

            iter+=1
        
        writer.add_scalar('Loss/Train', np.mean(epoch_retrieval_loss_t), epoch) # tensorboard logging

        rgan_wrapper.save_networks(epoch, os.path.dirname(log_file), best_acc=ret_best_acc,
                                        last_ckpt=True)  # Always save last ckpt

        # Save model periodically
        # if (epoch + 1) % opt.save_step == 0:
        #     rgan_wrapper.save_networks(epoch, os.path.dirname(log_file), best_acc=ret_best_acc)

        # torch.cuda.empty_cache() # PyTorch thing
        # print(torch.cuda.memory_allocated())
        # print(torch.cuda.memory_reserved())

        rgan_wrapper.retrieval.eval()
        with torch.no_grad():
            # for i, data in enumerate(val_loader):
            val_i = 0
            out = False
            while True:
                if val_i == 0:
                    start_time_batches = time.time()

                # we get here: batch_sat -> (batch_size, 320, 320, 3), batch_grd -> (batch_size, 320, 640, 3)
                batch_sat = dataloader.next_sat_scan(opt.batch_size*3)
                batch_grd = None

                if not out:
                    batch_grd = dataloader.next_grd_scan(opt.batch_size*2)
                
                if batch_grd is None:
                    # print("**********batch_grd is out")
                    out = True
                    
                
                if batch_sat is None: # or val_i == 1
                    # print("^^^^^^^^^^^^^^^break after 40 iters")
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
                
                # print(composed_data)
                # print(composed_data["satellite"].shape, composed_data["street"].shape)

                # we transformed the data into
                    #       composed_data -> {"satellite": torch.Size([batch_size, 3, 320, 320]), "street": torch.Size([batch_size, 3, 320, 640])}
                rgan_wrapper.set_input(composed_data)
                rgan_wrapper.eval_model()

                new_fake_street = rgan_wrapper.fake_street_out_val
                new_street = rgan_wrapper.street_out_val

                if new_fake_street is not None:
                    fake_street_batches_v.append(new_fake_street.cpu().data)
                if new_street is not None:
                    street_batches_v.append(new_street.cpu().data)
                # epoch_retrieval_loss_v.append(rgan_wrapper.r_loss.item())

                if (val_i + 1) % 40 == 0:
                    print("val batch=", val_i+1)
                    print("val time=", time.time() - start_time_batches)
                    start_time_batches = time.time()

                val_i += 1

        rss, vms, cpu_mem_percent_os, cpu_av_mem_os, cpu_mem_percent, cpu_available_mem_percent = get_sys_mem()
        log_print(".........Before matmul...........")
        log_print('Memory usage: rss={:.2f}GB vms={:.2f}GB Time:{:.2f}s\n'.format(rss, vms, time.time() - start_time))
        log_print('os mem percent: {} available mem percent: {} \n'.format(cpu_mem_percent_os, cpu_av_mem_os))
        log_print('psutil mem percent: {} available mem percent: {} \n'.format(cpu_mem_percent, cpu_available_mem_percent))
        
        # torch tensors will not work with np.sum() on booleans, original code had tf tensors here
        fake_street_vec = torch.cat(fake_street_batches_v, dim=0).numpy() #.astype(np.float16)
        street_vec = torch.cat(street_batches_v, dim=0).numpy() #.astype(np.float16)
        # street_vec_permuted = street_vec.permute(1, 0) # we are in numpy now, we can just use transpose instead of permute
        dist_array = 2 - 2 * np.matmul(street_vec, np.transpose(fake_street_vec)) # just to check memory usage
        

        rss, vms, cpu_mem_percent_os, cpu_av_mem_os, cpu_mem_percent, cpu_available_mem_percent = get_sys_mem()
        log_print(".........After matmul...........")
        log_print('Memory usage: rss={:.2f}GB vms={:.2f}GB Time:{:.2f}s\n'.format(rss, vms, time.time() - start_time))
        log_print('os mem percent: {} available mem percent: {} \n'.format(cpu_mem_percent_os, cpu_av_mem_os))
        log_print('psutil mem percent: {} available mem percent: {} \n'.format(cpu_mem_percent, cpu_available_mem_percent))

        # tp1 = rgan_wrapper.mutual_topk_acc(dists, topk=1)
        # tp5 = rgan_wrapper.mutual_topk_acc(dists, topk=5)
        # tp10 = rgan_wrapper.mutual_topk_acc(dists, topk=10)
        print(street_vec.shape, fake_street_vec.shape)

        val_accuracy, val_accuracy_top1, val_accuracy_top5, val_accuracy_top10, hit_rate = rgan_wrapper.validate_top_VIGOR(dist_array, dataloader)
        
        street_batches_v.clear()
        fake_street_batches_v.clear()

        log_print('Evaluation epoch %d: accuracy = %.1f%% , top1: %.1f%%, top5: %.1f%%, top10: %.1f%%, hit_rate: %.1f%%' % (
            epoch, val_accuracy * 100.0, val_accuracy_top1 * 100.0, val_accuracy_top5 * 100.0, val_accuracy_top10 * 100.0, hit_rate * 100.0))

        # num = len(dists)
        # tp1p = rgan_wrapper.mutual_topk_acc(dists, topk=0.01 * num)
        # acc = Namespace(num=len(dists), tp1=tp1, tp5=tp5, tp10=tp10, tp1p=tp1p)

        # log_print('\nEvaluate Samples:{num:d}\nRecall(p2s/s2p) tp1:{tp1[0]:.2f}/{tp1[1]:.2f} ' \
        #             'tp5:{tp5[0]:.2f}/{tp5[1]:.2f} tp10:{tp10[0]:.2f}/{tp10[1]:.2f} ' \
        #             'tp1%:{tp1p[0]:.2f}/{tp1p[1]:.2f}'.format(epoch + 1, num=acc.num, tp1=acc.tp1,
        #                                                     tp5=acc.tp5, tp10=acc.tp10, tp1p=acc.tp1p))
        
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
            rgan_wrapper.save_networks(epoch, os.path.dirname(log_file), best_acc=ret_best_acc, is_best=True)
            log_print('>>Save best model: epoch={} best_acc(tp1_p2s):{:.2f}'.format(epoch + 1, tp1_p2s_acc))

        # Program statistics
        rss, vms, cpu_mem_percent_os, cpu_av_mem_os, cpu_mem_percent, cpu_available_mem_percent = get_sys_mem()
        log_print('Memory usage: rss={:.2f}GB vms={:.2f}GB Time:{:.2f}s\n'.format(rss, vms, time.time() - start_time))
        log_print('os mem percent: {} available mem percent: {} \n'.format(cpu_mem_percent_os, cpu_av_mem_os))
        log_print('psutil mem percent: {} available mem percent: {} \n'.format(cpu_mem_percent, cpu_available_mem_percent))

    # Visualize the STN transformation on some input batch
    # if opt.polar is False:
    #     with torch.no_grad():
    #         images = next(iter(val_loader))['satellite'][:16].to(rgan_wrapper.device)
    #         transformed_images = rgan_wrapper.retrieval.module.spatial_tr(images)
    #         writer.add_images(f"Spatial Transformer/Inputs", convert_image_np(images.cpu()), dataformats='NHWC')
    #         writer.add_images(f"Spatial Transformer/Outputs", convert_image_np(transformed_images.cpu()), dataformats='NHWC')
    #         writer.close()