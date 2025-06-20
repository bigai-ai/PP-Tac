import os
import sys
sys.path.append('./')
from os.path import join as pjoin
import numpy as np
import shutil
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision

import yaml
from model import UNet, ConvMLP
from utils import parse_args, CompSet

sys.path.append(sys.path[0]+r"/../")
# sys.path.append(sys.path[0]+r"/models/")


from DataLoader import build_dataset_train, build_dataset_test





def build_models(opt):
    if opt.model=="UNet":
        if opt.concate_ref:
            model = UNet(1, ref=True)
        else:
            model = UNet(1, ref=False)
    if opt.model=="ConvMLP":
        if opt.concate_ref:
            model = ConvMLP(1, ref=True)
        else:
            model = ConvMLP(1, ref=False)
    else:
        raise KeyError('Model Does Not Exist')
    return model


# python test.py --config ./1.yml --test
if __name__ == '__main__':
    comp_opt = CompSet()
    opt = comp_opt.opt
    # device = torch.device(f"cuda:{opt.gpu_id}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device is {device}")

    model = build_models(opt)
    model.to(device)
    train_dataset, val_dataset = build_dataset_train(opt)

    START_EPOCH = 0
    if opt.rewake:
        REWAKE_CKPT_PATH = os.path.join(opt.save_ckpt_dir,'model_'+opt.rewake_ckpt)
        model.load_state_dict(torch.load(REWAKE_CKPT_PATH, map_location=device))
        START_EPOCH = int(opt.ckpt.split('_')[-2])

    # NOTE:
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i*10 for i in range(100)],gamma=0.9)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(opt.log_dir, 'TongTac{}'.format(timestamp)))

    sum_loss = 0.
    avg_loss = 0.
    iters = 0
    best_vloss = 1_000_000.
    STATEDICT = None
    ckpt_name_newest = None
    num_img = 0
    num_val_img = 0
    for epoch in range(START_EPOCH, opt.num_epochs):
        print(f'Starting epoch {epoch}')
        model.train(True)
        for idx, batch in enumerate(train_dataset):
            optimizer.zero_grad()
            img_gt, img_cap, img_gt_delta, mask ,ref_cap_img, cap_gray_delta, cap_gray_delta_mask= batch

            img_gt_delta = img_gt_delta.to(device)
            img_cap = img_cap.to(device)
            ref_cap_img = ref_cap_img.to(device)
            cap_gray_delta = cap_gray_delta.to(device)
            cap_gray_delta_mask = cap_gray_delta_mask.to(device)

            if opt.concate_ref:
                pred_depth_delta = model(img_cap,ref_cap_img)
            else:
                pred_depth_delta = model(cap_gray_delta)


            loss = criterion(img_gt_delta*cap_gray_delta_mask, pred_depth_delta*cap_gray_delta_mask)
            loss.backward()

            sum_loss = sum_loss + loss.item()
            optimizer.step()
            scheduler.step()
            
            num_img = num_img + img_gt.shape[0]
            if iters%10 ==9 :
                avg_loss = sum_loss / num_img
                print('//////////////////////////////////////////////////////////////\n\n')
                print('Epoch {}  Batch {} Loss: {}'.format(epoch, idx, avg_loss))
                writer.add_scalar('Loss/train', avg_loss, iters)
                sum_loss = 0
                num_img = 0
                print('//////////////////////////////////////////////////////////////\n\n')
                grid_capure = torchvision.utils.make_grid(img_cap)
                writer.add_image('capture_batch', grid_capure, iters)

                grid_capure_delta = torchvision.utils.make_grid(cap_gray_delta)
                writer.add_image('capture_delta_batch', grid_capure_delta, iters)


                grid_gt = torchvision.utils.make_grid(img_gt_delta)
                writer.add_image('gt_batch', grid_gt, iters)

                grid_pred = torchvision.utils.make_grid(pred_depth_delta)
                writer.add_image('pred_batch', grid_pred, iters)
                writer.flush()

            iters = iters +1

        # validation
        model.eval()
        val_sum_loss = 0.
        val_avg_loss = 0.
        with torch.no_grad():
            for i, val_batch in enumerate(val_dataset):
                img_gt, img_cap, img_gt_delta, mask, ref_cap_img, cap_gray_delta, cap_gray_delta_mask= val_batch
                img_gt_delta = img_gt_delta.to(device)
                img_cap = img_cap.to(device)
                ref_cap_img = ref_cap_img.to(device)
                cap_gray_delta = cap_gray_delta.to(device)
                cap_gray_delta_mask = cap_gray_delta_mask.to(device)

                if opt.concate_ref:
                    pred_depth_delta = model(img_cap,ref_cap_img)
                else:
                    pred_depth_delta = model(cap_gray_delta)
            
                vloss = criterion(img_gt_delta*cap_gray_delta_mask, pred_depth_delta*cap_gray_delta_mask)
                val_sum_loss = val_sum_loss + vloss.item()
                num_val_img = num_val_img + img_gt.shape[0]

        val_avg_loss = val_sum_loss / num_val_img
        num_val_img = 0

        print('Valuation: train loss{} valuation loss {}'.format(avg_loss, val_avg_loss))
        writer.add_scalar('Loss/validation', val_avg_loss, iters)

        grid_pred = torchvision.utils.make_grid(pred_depth_delta)
        writer.add_image('val_pred_batch', grid_pred, iters)

        grid_gt = torchvision.utils.make_grid(img_gt_delta)
        writer.add_image('val_gt_batch', grid_gt, iters)
        

        writer.flush()

        if val_avg_loss < best_vloss:
            best_vloss = val_avg_loss
            ckpt_name = 'model_{}'.format( epoch)
            STATEDICT = model.state_dict()

        if epoch == 0 :
            ckpt_name_newest = ckpt_name

        if epoch%3 == 0 and epoch>2:
            if ckpt_name!=ckpt_name_newest:
                torch.save(STATEDICT, os.path.join(opt.save_ckpt_dir, ckpt_name))
                ckpt_name_newest = ckpt_name
            # torch.save(STATEDICT, os.path.join(opt.save_ckpt_dir, ckpt_name))


