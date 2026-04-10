import numpy as np
import torch
import torch.optim as optim
import sys
from tqdm import trange
import os
from logger import Logger
from test import valid
from loss import MatchLoss
from utils import tocuda
from warmupMultiStepLR import WarmupMultiStepLR
from torch.utils.tensorboard import SummaryWriter

def train_step(step, optimizer, model, match_loss, data, scheduler):
    model.train()
    xs = data['xs']
    ys = data['ys'].squeeze(-1)
    logits, ys_ds, e_hat, y_hat = model(xs, ys)#return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat
    loss = 0
    loss_val = []
    #for i in range(len(y_hat)):
    loss, geo_loss, cla_loss, l2_loss, _, _ = match_loss.run(step, data, logits, ys_ds, e_hat, y_hat)
    #loss += loss_i
    loss_val += [geo_loss, cla_loss, l2_loss]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    return loss_val


def train(model, train_loader, valid_loader, config):
    model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr, weight_decay = config.weight_decay)
    scheduler = config.scheduler
    scheduler = WarmupMultiStepLR(optimizer, [200000, 400000], warmup_iters=100000,
                                  warmup_factor=0.01, warmup_method='linear')
    match_loss = MatchLoss(config)

    checkpoint_path = os.path.join(config.log_path, 'checkpoint.pth')
    config.resume = os.path.isfile(checkpoint_path)
    if config.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan', resume=True)
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan', resume=True)
    else:
        best_acc = -1
        start_epoch = 0
        logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan')
        logger_train.set_names(['Learning Rate'] + ['Geo Loss', 'Classfi Loss', 'L2 Loss']*(config.iter_num+1))
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan')
        logger_valid.set_names(['Valid Acc'] + ['Geo Loss', 'Clasfi Loss', 'L2 Loss'])

    # 初始化 TensorBoard
    tb_writer = SummaryWriter(log_dir=config.log_path)
    print(f"TensorBoard initialized, logs will be saved to: {config.log_path}")

    train_loader_iter = iter(train_loader)
    for step in trange(start_epoch, config.train_iter, ncols=config.tqdm_width):
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)
        train_data = tocuda(train_data)

        # run training
        cur_lr = optimizer.param_groups[0]['lr']
        # try:
        #     loss_vals = train_step(step, optimizer, model, match_loss, train_data)
        # except:
        #     print("Skip unstable step")
        #     continue

        loss_vals = train_step(step, optimizer, model, match_loss, train_data,scheduler)

        logger_train.append([cur_lr] + loss_vals)

        # TensorBoard 记录 - 每 100 步记录一次训练损失，避免 IO 开销
        if (step + 1) % 100 == 0:
            tb_writer.add_scalar('Train/Learning_Rate', cur_lr, step)
            tb_writer.add_scalar('Train/Geo_Loss', loss_vals[0], step)
            tb_writer.add_scalar('Train/Cla_Loss', loss_vals[1], step)
            tb_writer.add_scalar('Train/L2_Loss', loss_vals[2], step)

        # Check if we want to write validation
        b_save = ((step + 1) % config.save_intv) == 0
        b_validate = ((step + 1) % config.val_intv) == 0
        if b_validate:
            va_res, geo_loss, cla_loss, l2_loss,  _, _, _  = valid(valid_loader, model, step, config)
            logger_valid.append([va_res, geo_loss, cla_loss, l2_loss])

            # TensorBoard 记录验证结果
            tb_writer.add_scalar('Valid/Accuracy', va_res, step)
            tb_writer.add_scalar('Valid/Geo_Loss', geo_loss, step)
            tb_writer.add_scalar('Valid/Cla_Loss', cla_loss, step)
            tb_writer.add_scalar('Valid/L2_Loss', l2_loss, step)

            # 释放验证后的缓存显存，防止 OOM
            torch.cuda.empty_cache()

            if va_res > best_acc:
                print("Saving best model with va_res = {}".format(va_res))
                best_acc = va_res
                torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                }, os.path.join(config.log_path, 'model_best.pth'))
            if cla_loss < 1:
                print("Saving best model with cla_Loss = {}".format(cla_loss))
                # best_loss = cla_loss
                torch.save({
                    'epoch': step + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(config.log_path, 'model_best{}.pth'.format(cla_loss)))

        if b_save:
            torch.save({
            'epoch': step + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            }, checkpoint_path)

