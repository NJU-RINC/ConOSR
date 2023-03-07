import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.ABN import MultiBatchNorm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('MultiBatchNorm') != -1:
        m.bns[0].weight.data.normal_(1.0, 0.02)
        m.bns[0].bias.data.fill_(0)
        m.bns[1].weight.data.normal_(1.0, 0.02)
        m.bns[1].bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class simCNN_contrastive(nn.Module):
    def __init__(self, classid_list, num_ABN=5, head='mlp', feature_dim=128):
        super(self.__class__, self).__init__()
        self.num_classes = len(classid_list)
        self.feature_dim = feature_dim
        self.logit_dim = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classid_list = classid_list

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, self.feature_dim, 3, 2, 1, bias=False)

        self.bn1 = MultiBatchNorm(64, num_ABN)
        self.bn2 = MultiBatchNorm(64, num_ABN)
        self.bn3 = MultiBatchNorm(128, num_ABN)
        self.bn4 = MultiBatchNorm(128, num_ABN)
        self.bn5 = MultiBatchNorm(128, num_ABN)
        self.bn6 = MultiBatchNorm(128, num_ABN)
        self.bn7 = MultiBatchNorm(128, num_ABN)
        self.bn8 = MultiBatchNorm(128, num_ABN)
        self.bn9 = MultiBatchNorm(self.feature_dim, num_ABN)
        self.bn10 = MultiBatchNorm(128, num_ABN)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        if head == 'linear':
            self.head = nn.utils.spectral_norm(nn.Linear(self.feature_dim, self.logit_dim))
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(self.feature_dim, self.feature_dim)),
                nn.ReLU(inplace=True),
                nn.utils.spectral_norm(nn.Linear(self.feature_dim, self.logit_dim))
            )

        self.apply(weights_init)
        self.to(self.device)

    def get_feature(self, x, bn_label=None):
        if bn_label is None:
            bn_label = 0 * torch.ones(x.shape[0], dtype=torch.long).cuda()

        x = self.conv1(x)
        x, _ = self.bn1(x, bn_label)
        x = F.relu_(x)

        x = self.conv2(x)
        x, _ = self.bn2(x, bn_label)
        x = F.relu_(x)

        x = self.conv3(x)
        x, _ = self.bn3(x, bn_label)
        x = F.relu_(x)

        x = self.conv4(x)
        x, _ = self.bn4(x, bn_label)
        x = F.relu_(x)

        x = self.conv5(x)
        x, _ = self.bn5(x, bn_label)
        x = F.relu_(x)

        x = self.conv6(x)
        x, _ = self.bn6(x, bn_label)
        x = F.relu_(x)

        x = self.conv7(x)
        x, _ = self.bn7(x, bn_label)
        x = F.relu_(x)

        x = self.conv8(x)
        x, _ = self.bn8(x, bn_label)
        x = F.relu_(x)

        x = self.conv9(x)
        x, _ = self.bn9(x, bn_label)
        x = F.relu_(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_output(self, features):
        logits = F.normalize(self.head(features), dim=1)
        return logits

    def forward(self, x):
        features = self.get_feature(x)
        logits = self.get_output(features)
        return logits

    def freeze_weight(self):
        for param in self.parameters():
            param.requires_grad = False


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, classid_list, feature_dim=128):
        super(LinearClassifier, self).__init__()
        self.num_classes = len(classid_list)
        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classid_list = classid_list
        self.fc = nn.utils.spectral_norm(nn.Linear(feature_dim, self.num_classes))

    def forward(self, features):
        logits = self.fc(features)
        return logits

    def get_prob(self, logits, temperature=1):
        probs = torch.softmax(logits / temperature, dim=1)
        return probs

    def get_loss(self, logits, targets, temperature=1):
        log_probs = torch.log_softmax(logits / temperature, dim=1)
        loss = - torch.sum(log_probs * targets)
        return loss

    def estimate_threshold(self, probs, labels, percentile=5):
        self.classwise_thresholds = []
        classwise_probs = []
        for i in range(self.num_classes):
            classwise_probs.append([])

        for i, val in enumerate(probs):
            if self.classid_list.count(labels[i]) > 0:
                id_index = self.classid_list.index(labels[i])
                maxProb = np.max(probs[i])
                if probs[i, id_index] == maxProb:
                    classwise_probs[id_index].append(probs[i, id_index])

        for val in classwise_probs:
            if len(val) == 0:
                self.classwise_thresholds.append(0)
            else:
                threshold = np.percentile(val, percentile)
                self.classwise_thresholds.append(threshold)

        return self.classwise_thresholds

    def estimate_threshold_logits(self, feature_encoder, validation_loader, percentile=5):
        self.eval()
        self.classwise_thresholds = []
        classwise_logits = []
        for i in range(self.num_classes):
            classwise_logits.append([])

        for i, (images, labels) in enumerate(validation_loader):
            images = images.to(self.device)
            with torch.no_grad():
                features = feature_encoder.get_feature(images)
                logits = self.forward(features)
                maxLogit, maxIndexes = torch.max(logits, 1)

            for j, label in enumerate(labels):
                id_index = self.classid_list.index(label)
                if maxIndexes[j] == id_index:
                    classwise_logits[id_index].append(logits[j, id_index].item())

        for val in classwise_logits:
            if len(val) == 0:
                self.classwise_thresholds.append(0)
            else:
                threshold = np.percentile(val, percentile)
                self.classwise_thresholds.append(threshold)
        return self.classwise_thresholds

    def predict_logits(self, features):
        logits = self.forward(features)

        maxLogits, maxIndexes = torch.max(logits, 1)
        prediction = torch.zeros([maxIndexes.shape[0]], requires_grad=False).to(self.device)

        for i in range(maxIndexes.shape[0]):
            prediction[i] = self.classid_list[maxIndexes[i]]
            if maxLogits[i] <= self.classwise_thresholds[maxIndexes[i]]:
                prediction[i] = -1

        return prediction.long(), logits.detach().cpu().numpy()

    def predict_closed(self, features):
        outs = self.forward(features)
        probs = torch.sigmoid(outs)

        maxProb, maxIndexes = torch.max(probs, 1)
        prediction = torch.zeros([maxIndexes.shape[0]], requires_grad=False).to(self.device)

        for i in range(maxIndexes.shape[0]):
            prediction[i] = self.classid_list[maxIndexes[i]]

        return prediction.long()

class MLPClassifier(LinearClassifier):
    """MLP classifier"""
    def __init__(self, classid_list, feature_dim=128):
        super(MLPClassifier, self).__init__(classid_list, feature_dim)

        self.fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.feature_dim, self.feature_dim)),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Linear(self.feature_dim, self.num_classes))
        )
