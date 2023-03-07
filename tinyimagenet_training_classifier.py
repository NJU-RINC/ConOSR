import time
import numpy as np
import torch
from torch.cuda.amp import autocast as autocast, GradScaler
from models.simCNN_contrastive import *
from evaluation import openset_eval_contrastive, openset_eval_contrastive_logits
from utils import load_ImageNet200, get_smooth_labels
from mixup import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_root = './data/tinyimagenet/training/'

    TOTAL_CLASS_NUM = 200
    classid_all = [i for i in range(0, TOTAL_CLASS_NUM)]

    batch_size = 128
    lr = 0.0001
    num_classifier_epochs = 20
    percentile = 5
    label_smoothing_coeff = 0.1
    temperature = 1
    feature_dim = 128

    model_folder_path = './saved_models/'
    feature_encoder = torch.load(model_folder_path + 'tinyimagenet_encoder.pt')
    feature_encoder.to(device)
    classid_training = feature_encoder.classid_list
    classid_training.sort()
    print("==> Training Class: ", classid_training)

    train_loader_classifier, train_classes = load_ImageNet200([train_root], category_indexs=classid_training, train=True, batchSize=batch_size, useRandAugment=True)
    validation_loader, train_classes = load_ImageNet200([train_root], category_indexs=classid_training, train=False, batchSize=batch_size, useRandAugment=False)

    best_epoch = -1
    best_auc = 0
    scaler = GradScaler()

    feature_encoder.eval()
    classifier = MLPClassifier(classid_list=classid_training, feature_dim=feature_dim)
    classifier.to(device)
    optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_classifier, T_max=num_classifier_epochs)
    for classifier_epoch in range(num_classifier_epochs):
        classifier.train()
        time3 = time.time()
        total_classification_loss = 0
        for i, (images_classifier, labels_classifier) in enumerate(train_loader_classifier):
            images_classifier = images_classifier.to(device)
            labels_np = labels_classifier.numpy()
            labels_classifier = labels_classifier.to(device)
            targets = get_smooth_labels(labels_classifier, classid_training, smoothing_coeff=label_smoothing_coeff)

            optimizer_classifier.zero_grad()
            with autocast():
                with torch.no_grad():
                    features = feature_encoder.get_feature(images_classifier)
                logits = classifier(features)
                classification_loss = classifier.get_loss(logits, targets)

            scaler.scale(classification_loss).backward()
            scaler.step(optimizer_classifier)
            scaler.update()
            total_classification_loss += classification_loss

        scheduler_classifier.step()
        print('classifier_epoch {}: classification_loss = {:.3f}'.format(classifier_epoch, total_classification_loss))
        time4 = time.time()
        print('time for this epoch: {:.3f} minutes'.format((time4 - time3) / 60.0))

    with autocast():
        thresholds = classifier.estimate_threshold_logits(feature_encoder, validation_loader,percentile=percentile)
        print(thresholds)

    torch.save(classifier, model_folder_path + 'tinyimagenet_classifier.pt')
