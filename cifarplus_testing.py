import time
from torch.cuda.amp import autocast as autocast, GradScaler

from models.simCNN_contrastive import *
from models.ContrasiveLoss_SoftLabel import *
from evaluation import openset_eval_F1_contrastive
from utils import load_cifar, load_ImageNet_crop, load_ImageNet_resize, get_smooth_labels
from mixup import *


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_root = './data/CIFAR10_png/training/'
    test_root = './data/CIFAR10_png/testing/'
    test_root_unknown = './data/LSUN/testing/'
    classid_unknown = [0]
    # test_root_unknown = './data/tinyimagenet/testing/'
    # classid_unknown = [i for i in range(200)]

    percentile_min = 0
    percentile_max = 15


    model_folder_path = './saved_models/'
    encoder = torch.load(model_folder_path + 'cifar10_encoder.pt')
    classifier = torch.load(model_folder_path + 'cifar10_classifier.pt')
    encoder.to(device)
    classifier.to(device)
    classid_known = encoder.classid_list


    batch_size = 128
    validation_loader, _ = load_cifar([train_root], category_indexs=classid_known, train=False,
                                               batchSize=batch_size, shuffle=False)
    test_loader_known, _ = load_cifar([test_root], category_indexs=classid_known, train=False,
                                         batchSize=batch_size, shuffle=False)
    test_loader_unknown, _ = load_ImageNet_resize([test_root_unknown], category_indexs=classid_unknown, train=False,
                                                batchSize=batch_size, shuffle=False)

    print("test classes:", classid_unknown)

    for percentile in range(percentile_min, percentile_max+1):
        thresholds = classifier.estimate_validation_threshold_logits(encoder, validation_loader,
                                                                 percentile=percentile)

        accuracy_overall, accuracy_known, accuracy_unknown, f1_score = openset_eval_F1_contrastive(encoder,
                                                                                                classifier,
                                                                                                test_loader_known,
                                                                                                test_loader_unknown)
        print('percentile = {:.0f} - known acc = {:.3f}%, unknown acc = {:.3f}%, all acc = {:.3f}%, F1 = {:.3f} '.format(
        percentile, accuracy_known * 100, accuracy_unknown * 100, accuracy_overall * 100, f1_score))

