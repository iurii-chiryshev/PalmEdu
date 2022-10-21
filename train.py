from __future__ import print_function, division
from __future__ import absolute_import

import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from cnn.ONet import ONet
from cnn.SSRNet import SSRNet
from utils.datasets import AngleDatasets
from utils import transforms
from utils import tb_logger


import os
import numpy as np


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        print(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print('Save checkpoint to {0:}'.format(filename))


def adjust_learning_rate(optimizer, initial_lr, step_index):
    lr = initial_lr * (0.1 ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate_warmup(optimizer, epoch, base_learning_rate, batch_id, burn_in=1000):
    lr = base_learning_rate
    if batch_id < burn_in:
        lr = base_learning_rate * pow(float(batch_id) / float(burn_in), 4)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        if epoch >= 60:  # 40 320
            lr /= 10
        if epoch >= 120:
            lr /= 10
        if epoch >= 267:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train(net, trainloader, criterion, optimizer, device):
    net.train()
    train_loss = []
    train_mae = []

    for imgs, labels in trainloader:
        optimizer.zero_grad()
        imgs, labels = imgs.to(device), labels.to(torch.float32).to(device)
        predicts = net(imgs)
        optimizer.zero_grad()
        loss = criterion(predicts,labels)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        mae = torch.mean(torch.abs(predicts.view(-1) - labels.view(-1)))
        train_mae.append(mae.item())

    return np.mean(train_loss), np.mean(train_mae)


def validate(net, valloader, criterion,device):
    net.eval()
    val_mae = []
    val_loss = []
    for img, labels in valloader:
        img = img.to(device)
        labels = labels.to(torch.float32).to(device)
        with torch.no_grad():
            predicts = net(img)
            loss = criterion(predicts, labels)
            val_loss.append(loss.item())

        mae = torch.mean(torch.abs(labels.view(-1) - predicts.view(-1)))
        val_mae.append(mae.item())
    return np.mean(val_loss), np.mean(val_mae)


def main(args):
    print_args(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Net
    #net = SSRNet(image_size=args.input_size,class_range=360)
    net = ONet()
    net.to(device)

    dt = datetime.now().strftime("%m%d%Y_%H%M")
    snapshot_path = os.path.join(args.snapshot, dt)
    tensorboard_path = os.path.join(args.tensorboard, dt)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    logger = tb_logger.Logger(tensorboard_path)
    # TODO tensorboard --logdir models/logs/09092022_1212


    # super params
    step_epoch = [int(x) for x in args.step.split(',')]

    criterion = torch.nn.MSELoss()
    loss_name = criterion.__class__.__name__
    criterion = criterion.to(device)
    cur_lr = args.base_lr
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay)

    # dataset
    train_transform = transforms.Compose([
        # transforms.VeritcalFlip(),
        transforms.Rotate(angle_range = (-90,90), prob = 1.0),
        transforms.Resize((args.input_size,args.input_size)),
        transforms.Normalize(mean=(127.5,127.5,127.5),std=(127.5,127.5,127.5)),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        # transforms.VeritcalFlip(),
        transforms.Rotate(angle_range=(-90, 90), prob=1.0),
        transforms.Resize((args.input_size, args.input_size)),
        transforms.Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)),
        transforms.ToTensor()
    ])

    train_datasets = AngleDatasets(args.train_path,transforms=train_transform)
    trainloader = DataLoader(
        train_datasets,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True)

    test_datasets = AngleDatasets(args.test_path,transforms=test_transform)
    valloader = DataLoader(
        test_datasets,
        batch_size=args.test_batchsize,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True)

    step_index = 0

    for epoch in range(args.start_epoch, args.end_epoch + 1):
        train_loss, train_MAE = train(net, trainloader,
                               criterion, optimizer, device)
        val_loss, val_MAE = validate(net,valloader,criterion,device)
        if epoch in step_epoch:
            step_index += 1
            cur_lr = adjust_learning_rate(optimizer, args.base_lr, step_index)

        print('Epoch: %d,  train_loss:%6.4f, val_loss:%8.6f, train_mae:%6.4f, val_mae:%8.6f, lr:%8.6f' % (epoch, train_loss, val_loss, train_MAE, val_MAE, cur_lr))
        filename = os.path.join(
            str(snapshot_path), "checkpoint_epoch_" + str(epoch) + '.pth')
        save_checkpoint(net.state_dict(), filename)

        info = {
            "{}/Train".format(loss_name): train_loss,
            "{}/Val".format(loss_name): val_loss,

            'MAE/Train': train_MAE,
            'MAE/Val': val_MAE,

            "Learning rate": cur_lr
        }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)


def parse_args():
    parser = argparse.ArgumentParser(description='Trainging Template')
    # general
    parser.add_argument('-j', '--workers', default=4, type=int)

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.002, type=float)
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--step', default="30,80,180", help="lr decay", type=str)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=200, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot', default='./models/checkpoint/snapshot/', type=str, metavar='PATH')
    parser.add_argument('--tensorboard', default="./models/checkpoint/tensorboard", type=str)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')  # TBD

    # --dataset
    parser.add_argument('--train_path', default='./', type=str, metavar='PATH')
    parser.add_argument('--test_path', default='./', type=str, metavar='PATH')
    parser.add_argument('--train_batchsize', default=64, type=int)
    parser.add_argument('--test_batchsize', default=32, type=int)
    parser.add_argument('--input_size', default=112, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
