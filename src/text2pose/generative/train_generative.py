##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

import sys, os
import shutil
from pathlib import Path
import time
import json
import numpy as np
import math
import roma
import csv
from datetime import datetime, timedelta
from human_body_prior.body_model.body_model import BodyModel

import torch 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import text2pose.config as config
from text2pose.option import get_args_parser, get_output_dir
from text2pose.data import PVCP2, collate_fn_general
from text2pose.loss import laplacian_nll, gaussian_nll
from text2pose.utils import MetricLogger, SmoothedValue, print_model_parameters
from text2pose.generative.model_generative import PPSG
from text2pose.generative.fid import FID

os.umask(0x0002)


def main(args):
    if os.path.isfile( os.path.join(args.output_dir, f'checkpoint_{args.epochs-1}.pth') ):
        print(f'✅ Training already done ({args.epochs} epochs: checkpoint_{args.epochs-1}.pth')
        return
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    # load data
    print('✅ Load training dataset')
    if args.text_encoder_name.split("_")[0] in ["glovebigru"]:
        collate_gen = None
    elif args.text_encoder_name.split("_")[0] in ["Qwen", "qwen"]:
        collate_gen = collate_fn_general
        
    dataset_train = PVCP2(split='train', text_encoder_name=args.text_encoder_name)
    print(dataset_train, len(dataset_train))
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=None, shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_gen
    )
    print('✅ Load validation dataset')
    dataset_val = PVCP2(split='val', text_encoder_name=args.text_encoder_name)
    print(dataset_val, len(dataset_val))
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=None, shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_gen
    )
    
    # set up models
    print('✅ Load model')
    model = PPSG(text_encoder_name=args.text_encoder_name, latentD=args.latentD)
    model.to(args.device)
    print_model_parameters(model)
    
    print('✅ Initialize body model')
    body_model = BodyModel(bm_fname = config.SMPL_NEUTRAL_BM, num_betas = config.SMPL_n_betas, model_type='smpl')
    body_model.eval()
    body_model.to(args.device)
    
    # optimizer 
    assert args.optimizer=='Adam'
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # prepare fid on the val set
    print('✅ Preparing FID')
    fid = FID((args.fid, args.seed), model, device=args.device)
    fid.extract_real_features(data_loader_val)
  
    # start or resume training
    ckpt_last_fname = os.path.join(args.output_dir, 'checkpoint_last.pth')

    if os.path.isfile(ckpt_last_fname): # resume training
        ckpt = torch.load(ckpt_last_fname, args.device)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if start_epoch == args.epochs:
            print(f'✅ Training already done {args.epochs} epochs: checkpoint_last.pth')
        else:
            print(f"⭐ Resume training from epoch {start_epoch}.")
    else:
        start_epoch = 0
        if args.pretrained: # load pretrained weights
            pretrained_path = (config.shortname_2_model_path[args.pretrained]).format(seed=args.seed)
            print(f'⭐ Loading pretrained model: {pretrained_path}')
            ckpt = torch.load(pretrained_path, args.device)
            model.load_state_dict(ckpt['model'])

    # tensorboard 
    log_writer = SummaryWriter(log_dir=args.output_dir)
    
    # training process
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs): 
        # training
        train_stats = one_epoch(
            model, body_model,
            is_training=True,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            log_writer=log_writer,
            args=args,
            fid=None,
        )

        # validation
        val_stats = one_epoch(
            model, body_model,
            is_training=False,
            data_loader=data_loader_val,
            optimizer=None,
            epoch=epoch,
            log_writer=log_writer,
            args=args,
            fid=fid,
        )        
        
        # save checkpoint
        tosave = {'epoch': epoch, 
                  'model': model.state_dict(), 
                  'optimizer': optimizer.state_dict(), 
                  'args': args,
                  }
        
        if epoch % args.saving_ckpt_step==0:
            print(f'save checkpoint [{os.path.basename(ckpt_last_fname)}] for resume...')
            ckpt_current_fname = os.path.join(args.output_dir, f'checkpoint_{epoch}.pth')
            torch.save(tosave, ckpt_current_fname)
            
        # last
        if epoch + 1 == args.epochs:
            print(f'save current checkpoint [{os.path.basename(ckpt_last_fname)}] for resume...')
            torch.save(tosave, ckpt_last_fname)
        
        # tensorboard
        log_writer.flush()

        # save logs
        log_stats = {'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                     'epoch': epoch, 
                     'lr': optimizer.param_groups[0]["lr"]}
        log_stats.update(**train_stats)
        log_stats.update(**val_stats)
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
        
        log_file_path = os.path.join(args.output_dir, "log.csv")
        is_new_file = not os.path.exists(log_file_path)
        with open(log_file_path, mode="a", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_stats.keys())

            if is_new_file:
                writer.writeheader()
                
            writer.writerow(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print('⏱️  Training time {}'.format(total_time_str))
    
    
def one_epoch(model, body_model, is_training, data_loader, optimizer, epoch, args, fid=None, log_writer=None):

    metric_logger = MetricLogger(delimiter="  ")
    if is_training: metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = '{}>>>gen-Epoch: [{}/{}]'.format('' if is_training else '[val] ', epoch, args.epochs)
    if fid is not None: fid.reset_gen_features()

    model.train(is_training)

    sstr = 'train_' if is_training else 'val_'
    
    for data_iter_step, item in enumerate(metric_logger.log_every(data_loader, args.log_step, header)):

        # get data
        poses = item['pose'].to(args.device) # [B,J,3]
        poses_class = item['pose_class'].to(args.device) # [B,4]
        caption_lengths = item['caption_lengths'].to(args.device)
        if isinstance(item['caption_tokens'][0], str):  # Qwen
            caption_tokens = item['caption_tokens']
        else:  # GloVe
            caption_tokens = item['caption_tokens'].to(args.device)
            caption_tokens = caption_tokens[:,:caption_lengths.max()] # truncate within the batch, based on the longest text 
        # forward
        with torch.set_grad_enabled(is_training):
            output = model(poses, caption_tokens, caption_lengths)  
            bm_rec = body_model(pose_body=output['pose_body_text'][:,1:22].flatten(1,2),   # text
                                root_orient=output['pose_body_text'][:,:1].flatten(1,2))
        with torch.no_grad():
            bm_orig = body_model(pose_body=poses[:,1:22].flatten(1,2),
                                root_orient=poses[:,:1].flatten(1,2))
            
        # compute losses 
        losses = {}
        
        # -- reconstruction losses
        losses[f'v2v'] = torch.mean(laplacian_nll(bm_orig.v, bm_rec.v, model.decsigma_v2v))
        losses[f'jts'] = torch.mean(laplacian_nll(bm_orig.Jtr, bm_rec.Jtr, model.decsigma_jts))
        losses[f'rot'] = torch.mean(gaussian_nll(output[f'pose_body_matrot_pose'].view(-1,3,3), roma.rotvec_to_rotmat(poses.view(-1, 3)), model.decsigma_rot))
                
        # -- KL losses
        losses['kldpt'] = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], output['t_z']), dim=[1])) # N(p,t)
        
        # -- KL regularization losses
        bs = poses.size(0)
        n_z = torch.distributions.normal.Normal(  # N(0,1)
            loc=torch.zeros((bs, model.latentD), device=args.device, requires_grad=False),
            scale=torch.ones((bs, model.latentD), device=args.device, requires_grad=False))
    
        # -- Pose class loss
        poses_class_idx = poses_class.argmax(dim=-1)   # [B]
        has_label = poses_class.sum(dim=-1) > 0
        if has_label.any():
            losses['cls'] = F.cross_entropy(
                output['pose_class_logits'][has_label], 
                poses_class_idx[has_label]
            )
        else:
            losses['cls'] = torch.tensor(0.0, device=args.device)
    
        # -- total loss
        loss = losses['v2v'] * args.wloss_v2v + \
               losses['jts'] * args.wloss_jts + \
               losses['rot'] * args.wloss_rot + \
               losses['kldpt'] * args.wloss_kld + \
               losses['cls'] * args.wloss_cls

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        
        # elbos (normalization is a bit different than for the losses)
        elbos = {}
        elbos[f'v2v'] = (-laplacian_nll(bm_orig.v, bm_rec.v, model.decsigma_v2v).sum()/2./bs - losses['kldpt']).detach().item()
        elbos[f'jts'] = (-laplacian_nll(bm_orig.Jtr, bm_rec.Jtr, model.decsigma_jts).sum()/2./bs - losses['kldpt']).detach().item()
        elbos[f'rot'] = (-gaussian_nll(output[f'pose_body_matrot_pose'].view(-1,3,3), roma.rotvec_to_rotmat(poses.view(-1, 3)), model.decsigma_rot).sum()/2./bs - losses['kldpt']).detach().item()
        
        # fid
        if fid is not None:
            fid.add_gen_features( output['pose_body_text'] )
       
        # step
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # loggers
        metric_logger.update( **{sstr+'loss': loss_value} )
        for k,v in losses.items():
            metric_logger.update( **{sstr+'loss_'+k: v.item()} )
        for k,v in elbos.items():
            metric_logger.update( **{sstr+'elbo_'+k: v} )
        if is_training: 
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
        # We use epoch_1000x as the x-axis in tensorboard.
        # This calibrates different curves when batch size changes.
        if log_writer and (is_training or data_iter_step==len(data_loader)-1):
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar(sstr.replace('_','/')+'loss', loss_value, epoch_1000x)
            for k, v in losses.items():
                log_writer.add_scalar(sstr.replace('_','/')+'loss_'+k, v.item(), epoch_1000x)
            for k, v in elbos.items():
                log_writer.add_scalar(sstr.replace('_','/')+'elbo_'+k, v, epoch_1000x)
            if is_training: 
                log_writer.add_scalar('lr', lr, epoch_1000x)
                log_writer.add_scalar('decsigma_v2v', model.decsigma_v2v.detach().item(), epoch_1000x)
                log_writer.add_scalar('decsigma_jt2', model.decsigma_jts.detach().item(), epoch_1000x)
                log_writer.add_scalar('decsigma_rot', model.decsigma_rot.detach().item(), epoch_1000x)

    if fid is not None:
        fid_value = fid.compute()
        fidstr = fid.sstr()
        metric_logger.add_meter(fidstr, SmoothedValue(window_size=1, fmt='{value:.3f}'))
        metric_logger.update(**{fidstr: fid_value})
        if log_writer: log_writer.add_scalar(sstr.replace('_','/')+fidstr, fid_value, epoch_1000x)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    

if __name__ == '__main__':
    argparser = get_args_parser()
    args = argparser.parse_args()
    
    # create path to saving directory
    if args.output_dir=='':
        args.output_dir = get_output_dir(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print('✅ save args:\n', json.dumps(vars(args), indent=4, ensure_ascii=False))
    
    args_path = Path(args.output_dir) / "args.json"
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False, default=str)
    
    main(args)