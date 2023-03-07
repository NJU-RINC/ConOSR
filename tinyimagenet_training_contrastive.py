import time
import numpy as np
import torch
from torch.cuda.amp import autocast as autocast, GradScaler
from models.simCNN_contrastive import *
from models.ContrasiveLoss_SoftLabel import *
from evaluation import openset_eval_contrastive_logits
from utils import load_ImageNet200, load_ImageNet200_contrastive, get_smooth_labels
from mixup import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_root = './data/tinyimagenet/training/'
    TOTAL_CLASS_NUM = 200
    classid_all = [i for i in range(0, TOTAL_CLASS_NUM)]

    classid_training = [98, 36, 158, 177, 189, 157, 170, 191, 82, 196, 138, 166, 43, 13, 152, 11, 75, 174, 193, 190]
    classid_training.sort()

    batch_size = 128
    lr = 0.001
    num_contrastive_epochs = 600
    percentile = 5
    temperature = 0.1
    label_smoothing_coeff = 0.2
    feature_dim = 128

    model_folder_path = './saved_models/'

    print("==> Training Class: ", classid_training)

    train_loader_contrastive, train_classes = load_ImageNet200_contrastive([train_root], category_indexs=classid_training, batchSize=batch_size)
    train_loader_classifier, train_classes = load_ImageNet200([train_root], category_indexs=classid_training, train=True, batchSize=batch_size, useRandAugment=False)

    best_epoch = -1
    best_auc = 0

    feature_encoder = simCNN_contrastive(classid_list=classid_training, head='linear')
    feature_encoder.to(device)
    criterion = SupConLoss(temperature=temperature, base_temperature=temperature)
    criterion.to(device)

    optimizer = torch.optim.Adam(feature_encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min = lr * 1e-3)

    scaler = GradScaler()

    for epoch in range(1, num_contrastive_epochs+1):
        feature_encoder.train()
        time1 = time.time()
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader_contrastive):
            targets = get_smooth_labels(labels, classid_training, label_smoothing_coeff)
            images_mixup, targets_mixup, targets_a, targets_b, lam = mixup_data_contrastive(images, targets, alpha=1,
                                                                                            use_cuda=False)

            images = torch.cat([images[0], images[1]], dim=0)
            images = images.to(device)
            targets = targets.to(device)

            images_mixup = torch.cat([images_mixup[0], images_mixup[1]], dim=0)
            images_mixup = images_mixup.to(device)
            targets_mixup = targets_mixup.to(device)

            bsz = targets.shape[0]

            optimizer.zero_grad()
            with autocast():
                logits = feature_encoder(images)
                logits1, logits2 = torch.split(logits, [bsz, bsz], dim=0)
                logits = torch.cat([logits1.unsqueeze(1), logits2.unsqueeze(1)], dim=1)

                logits_mixup = feature_encoder(images_mixup)
                logits3, logits4 = torch.split(logits_mixup, [bsz, bsz], dim=0)
                logits_mixup = torch.cat([logits3.unsqueeze(1), logits4.unsqueeze(1)], dim=1)

                logits_combine = torch.cat([logits, logits_mixup], dim=0)
                targets_combine = torch.cat([targets, targets_mixup], dim=0)
                loss = criterion(logits_combine, targets_combine)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss

        print('epoch {}: contrastive_loss = {:.3f},  '.format(epoch, total_loss))
        time2 = time.time()
        scheduler.step()
        print('time for this epoch: {:.3f} minutes'.format((time2 - time1) / 60.0))

    torch.save(feature_encoder, model_folder_path + 'tinyimagenet_encoder.pt')

