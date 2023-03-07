import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import auc, roc_auc_score, f1_score

def closedset_eval(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            prediction = model.predict_closed(images)
            correct += (prediction == labels).sum().item()
            total += len(images)

    Accuracy = correct / total
    return Accuracy

def openset_eval_contrastive(model, classifier, known_test_loader, unknown_test_loader, temperature=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    classifier.eval()
    total_known = 0
    total_unknown = 0
    correct_known = 0
    correct_unknown = 0

    label_binary = []
    label_prob = []
    known_prob = []
    unknown_prob = []


    for images, labels in known_test_loader:
        images = images.to(device)

        labels = labels.to(device)

        features = model.get_feature(images)
        prediction, probabilities = classifier.predict(features, temperature)
        for i in range(len(images)):
            total_known += 1
            if prediction[i] == labels[i]:
                correct_known += 1
            label_prob.append(np.max(probabilities[i, :]))
            known_prob.append(np.max(probabilities[i, :]))
            label_binary.append(1)

    print('mean prob of known classes:{:.3f}, std:{:.3f}'.format(np.mean(known_prob), np.std(known_prob)))

    for images, labels in unknown_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        features = model.get_feature(images)
        prediction, probabilities = classifier.predict(features, temperature)

        for i in range(len(images)):
            total_unknown += 1
            if prediction[i] == -1:
                correct_unknown += 1
            label_prob.append(np.max(probabilities[i, :]))
            unknown_prob.append(np.max(probabilities[i, :]))
            label_binary.append(0)
    print('mean prob of unknown classes:{:.3f}, std:{:.3f}'.format(np.mean(unknown_prob), np.std(unknown_prob)))

    AUC = roc_auc_score(label_binary, label_prob)
    Accuracy_known = correct_known / total_known
    Accuracy_unknown = correct_unknown / total_unknown
    Accuracy = (correct_known + correct_unknown) / (total_known + total_unknown)
    return Accuracy, Accuracy_known, Accuracy_unknown, AUC

def openset_eval_contrastive_logits(model, classifier, known_test_loader, unknown_test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    classifier.eval()
    total_known = 0
    total_unknown = 0
    correct_known = 0
    correct_unknown = 0

    label_binary = []
    label_logits = []
    known_logits = []
    unknown_logits = []


    for images, labels in known_test_loader:
        images = images.to(device)
        labels = labels.to(device)

        features = model.get_feature(images)
        prediction, logits = classifier.predict_logits(features)
        for i in range(len(images)):
            total_known += 1
            if prediction[i] == labels[i]:
                correct_known += 1
            label_logits.append(np.max(logits[i, :]))
            known_logits.append(np.max(logits[i, :]))
            label_binary.append(1)

    print('mean logits of known classes:{:.3f}, std:{:.3f}'.format(np.mean(known_logits), np.std(known_logits)))

    for images, labels in unknown_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        features = model.get_feature(images)
        prediction, logits = classifier.predict_logits(features)

        for i in range(len(images)):
            total_unknown += 1
            if prediction[i] == -1:
                correct_unknown += 1
            label_logits.append(np.max(logits[i, :]))
            unknown_logits.append(np.max(logits[i, :]))
            label_binary.append(0)
    print('mean logits of unknown classes:{:.3f}, std:{:.3f}'.format(np.mean(unknown_logits), np.std(unknown_logits)))

    AUC = roc_auc_score(label_binary, label_logits)

    Accuracy_known = correct_known / total_known
    Accuracy_unknown = correct_unknown / total_unknown
    Accuracy = (correct_known + correct_unknown) / (total_known + total_unknown)
    return Accuracy, Accuracy_known, Accuracy_unknown, AUC

def openset_eval_F1(model, known_test_loader, unknown_test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_known = 0
    total_unknown = 0
    correct_known = 0
    correct_unknown = 0

    labels_all = np.asarray([])
    predictions_all = np.asarray([])

    with torch.no_grad():
        for images, labels in known_test_loader:
            images = images.to(device)

            labels_np = labels.numpy()
            labels = labels.to(device)
            labels_all = np.concatenate((labels_all, labels_np))

            prediction, probilities, dists = model.predict(images)
            for i in range(len(images)):
                total_known += 1
                if prediction[i] == labels[i]:
                    correct_known += 1
            predictions_all = np.concatenate((predictions_all, prediction.cpu().detach().numpy()))


        for images, labels in unknown_test_loader:
            images = images.to(device)
            labels_np = -1 * np.ones(len(labels))
            labels_all = np.concatenate((labels_all, labels_np))

            prediction, probilities, dists = model.predict(images)

            for i in range(len(images)):
                total_unknown += 1
                if prediction[i] == -1:
                    correct_unknown += 1

            predictions_all = np.concatenate((predictions_all, prediction.cpu().detach().numpy()))

    F1Score = f1_score(labels_all, predictions_all, average='macro')

    Accuracy_known = correct_known / total_known
    Accuracy_unknown = correct_unknown / total_unknown
    Accuracy = (correct_known + correct_unknown) / (total_known + total_unknown)
    return Accuracy, Accuracy_known, Accuracy_unknown, F1Score

def openset_eval_F1_contrastive(encoder, classifier, known_test_loader, unknown_test_loader, disturb_rate=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.eval()
    classifier.eval()
    total_known = 0
    total_unknown = 0
    correct_known = 0
    correct_unknown = 0

    label_binary = []
    label_logits = []
    known_logits = []
    unknown_logits = []

    labels_all = np.asarray([])
    predictions_all = np.asarray([])

    for images, labels in known_test_loader:
        images = images.to(device)

        labels_np = labels.numpy()
        labels = labels.to(device)
        labels_all = np.concatenate((labels_all, labels_np))

        features = encoder.get_feature(images)
        prediction, logits = classifier.predict_logits(features)
        for i in range(len(images)):
            total_known += 1
            if prediction[i] == labels[i]:
                correct_known += 1
            label_logits.append(np.max(logits[i, :]))
            known_logits.append(np.max(logits[i, :]))
            label_binary.append(1)
        predictions_all = np.concatenate((predictions_all, prediction.cpu().detach().numpy()))

    for images, labels in unknown_test_loader:
        images = images.to(device)
        labels_np = -1 * np.ones(len(labels))
        labels_all = np.concatenate((labels_all, labels_np))
        features = encoder.get_feature(images)
        prediction, logits = classifier.predict_logits(features)

        for i in range(len(images)):
            total_unknown += 1
            if prediction[i] == -1:
                correct_unknown += 1
            label_logits.append(np.max(logits[i, :]))
            unknown_logits.append(np.max(logits[i, :]))
            label_binary.append(0)

        predictions_all = np.concatenate((predictions_all, prediction.cpu().detach().numpy()))

    F1Score = f1_score(labels_all, predictions_all, average='macro')

    Accuracy_known = correct_known / total_known
    Accuracy_unknown = correct_unknown / total_unknown
    Accuracy = (correct_known + correct_unknown) / (total_known + total_unknown)
    return Accuracy, Accuracy_known, Accuracy_unknown, F1Score