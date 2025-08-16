# from dataloader import UISdataset_MM
import argparse
from torchvision.models import resnet50, resnext50_32x4d, densenet121

import torch
import torch.nn as nn
import os
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
import time
from torch import nn, Tensor
from torch.nn import functional as F
from tabulate import tabulate
# from model_fuse import LightSDNet
from torch.utils.data import DataLoader, Dataset
import time
from dataloader import OSdataset, Openspacedataset
# from make_dataset import ImgDataset, readfile, readfile_add, ImgDataset_train
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
# import pytorch_ssim # pytorch_ssim.SSIM()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Boundary-Aware Feature Propagation
# from rmi import RMILoss
import sklearn.utils.class_weight as class_weight
import numpy as np
from loss import FocalLossSemi
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_segformer import Segformer_baseline, Segformer_FaPN, UOFormer

def main(args):
    # Create model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    train_txt = "./data/train_patch_OS_GlobalCity.txt"
    val_txt = "./data/test_patch_OS_GlobalCity.txt"
    # test_txt = "./data/testSSOUISdataset_all.txt"

    train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()
                                                ])
    val_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # test_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = Openspacedataset(txt = train_txt,transform=train_transform)
    val_dataset = Openspacedataset(txt = val_txt,transform=val_transform)
    # test_dataset=UISdataset_MM(txt=test_txt,transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=1,pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8,pin_memory=True)

    print("Train numbers:{:d}".format(len(train_dataset)))
    print("val numbers:{:d}".format(len(val_dataset)))

    model2 = UOFormer(args.num_class)  # Segformer_baseline, LightSDNet

    print('model2 parameters:', sum(p.numel() for p in model2.parameters()))
    Trainable_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f'Trainable params: {Trainable_params/ 1e6}M')

    # model1 = model1.to(device)
    model2 = model2.to(device)

    cost1 = FocalLossSemi().to(device)
    # cost1 = nn.CrossEntropyLoss().to(device)  # [0.6601, 2.0615]

    # optimizer2 = torch.optim.AdamW([{'params': model2.backbone.parameters(), 'lr': 1e-4}, {'params': model2.decode_path.parameters()},
    # {'params': model2.SpatialPath.parameters()}, {'params': model2.FeatureFusionModule.parameters()}, {'params': model2.conv_fc.parameters()}, {'params': model2.conv_sp_fc.parameters()}], lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr)

    # best_acc_1 = 0.
    miou_max = 0.
    best_epoch = 0
    # best_acc_3 = 0.
    # alpha = 1
    for epoch in range(1, args.epochs + 1):
        # model1.train()
        model2.train()
        # model3.train()
        # start time
        start = time.time()
        index = 0
        for images, labels in train_loader:
            images = images.to(device)

            # mmdata = data[1].to(device)
            # print(images.shape)
            labels = labels.to(device, dtype=torch.int64).squeeze(1)
            # edges = data[2].to(device, dtype=torch.int64)
            # print(images.shape, labels.shape)
            # instance_label = instance_label.to(device, dtype=torch.int64)
            # mmdata = mmdata.clone().detach().float()
            images = images.clone().detach().float()
            # m = nn.Sigmoid()
            # labels = labels.clone().detach().Long()

            # Forward pass
            # outputs1 = model1(images)
            outputs2 = model2(images)  # out_0, out,
            # print(out_0.shape, out.shape, outputs2.shape)
            # outputs3 = model3(images)
            # loss1 = cost1(outputs1, labels)
            # print(outputs2.shape, labels.squeeze(1).shape, instance_out.shape, instance_label.squeeze(1).shape)
            loss = cost1(outputs2, labels)
            # loss = 0.1*cost1(out_0, labels) + 0.1*cost1(out, labels) + 0.8*cost1(outputs2, labels)

            # print(loss)
            # loss = loss.sum()
            # loss2 = 0.02*cost2(out_edge, edges.unsqueeze(1).float()) # torch.nn.functional.one_hot(edges)
            # loss = loss1 + loss2
            # loss3 = cost3(outputs3, labels)

            # if index % 10 == 0:
                # print (loss)
            # Backward and optimize
            # optimizer1.zero_grad()
            optimizer2.zero_grad()
            # optimizer3.zero_grad()
            # loss1.backward()
            loss.backward()
            # loss3.backward()
            # optimizer1.step()
            optimizer2.step()
            # optimizer3.step()
            index += 1

        # scheduler_poly_lr_decay.step()
        if epoch % 1 == 0:
            end = time.time()
            # print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss1.item(), (end-start) * 2))
            print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss.item(), (end-start) * 2))
            # print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss3.item(), (end-start) * 2))

            # model1.eval()
            model2.eval()
            # model3.eval()

            # classes = ('bareland', 'cropland', 'forest', 'impervious', 'shrub', 'water')
            # dic_type = {"城市公共空间":1,"公园":2,"户外运动场所":3,"交通场地空间":4,"景点":5,"非城市开放空间":7,"水体":6}
            # classes = ['Mask', '城市公共空间', '公园景点-户外运动场所-交通场地空间', '水体', '非城市开放空间'] # ('住宅区', '公共服务区域', '商业区', '城市绿地', '工业区')
            # {"城市公共空间": 1, "公园景点": 2, "户外运动场所": 3, "交通场地空间": 4, "水体": 5, "非城市开放空间": 6}
            # classes = ['Mask', '公园景点', '交通场地空间', '水体', '非城市开放空间']  # ['Mask', '城市公共空间', '公园景点', '户外运动场所', '交通场地空间', '水体', '非城市开放空间']
            classes = ['公园绿地', '户外运动场所', '交通场地空间', '水体', '非城市开放空间', 'Mask']  #  '户外运动场所',  {"其他":1,"公园绿地":2,"户外运动场所":3,"交通场地空间":4,"水体":5,"非城市开放空间":6}
            hist = torch.zeros(args.num_class, args.num_class).to(device)

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)

                    # mmdata = data[1].to(device)
                    # print(images.shape)
                    labels = labels.to(device, dtype=torch.int64).squeeze(1)
                    # instance_label = instance_label.to(device, dtype=torch.int64)
                    # mmdata = mmdata.clone().detach().float()
                    images = images.clone().detach().float()
                    # labels = labels.clone().detach().Long()

                    # Forward pass
                    # outputs1 = model1(images)
                    outputs2 = model2(images)  # out_0, out,
                    # print(labels.shape)

                    # Forward pass
                    # outputs1 = model1(images)
                    # outputs2 = model2(images, mmdata)
                    # outputs3 = model3(images)
                    # print(outputs2.shape)
                    # loss1 = cost1(outputs1, labels)
                    preds = outputs2[:, :5, :, :].softmax(dim=1).argmax(dim=1)
                    # print(preds.shape)

                    keep = labels != 5
                    hist += torch.bincount(labels[keep] * args.num_class + preds[keep], minlength=args.num_class**2).view(args.num_class, args.num_class)

            ious = hist.diag() / (hist.sum(0) + hist.sum(1) - hist.diag())
            miou = ious[~ious.isnan()].mean().item()
            ious = ious.cpu().numpy().tolist()
            miou = miou * 100

            Acc = hist.diag() / hist.sum(1)
            mOA = hist.diag().sum() / hist.sum() * 100

            # table = {
            #     'Class': classes[1:],
            #     'IoU': ious[1:],
            #     'Acc': Acc[1:],
            #     # 'mOA': mOA
            # }

            table = {
                'Class': classes[:5],
                'IoU': ious[:5],
                'Acc': Acc[:5],
                # 'mOA': mOA
            }

            print(tabulate(table, headers='keys'))
            print(f"\nOverall mIoU: {miou:.2f}")
            print(f"\nOverall mOA: {mOA:.2f}")

        if  miou > miou_max:
            print('save new best miou', miou)
            torch.save(model2, os.path.join(args.model_path, 'Openspace-GlobalCity-5class-UOFormer-round2.pth'))  # GVG-2.pth
            miou_max = miou
            best_epoch = epoch

        print('Current best iou', miou_max, best_epoch)
        print("-----------------------------------------")
    # print('save new best acc_3', best_acc_3)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=6, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    # parser.add_argument("--net", default='ResNet50', type=str)
    # parser.add_argument("--depth", default=50, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    # parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--model_name", default='', type=str)
    parser.add_argument("--model_path", default='./GlobalCity_OS_model', type=str)
    parser.add_argument("--pretrained", default=False, type=bool)
    args = parser.parse_args()

    main(args)
