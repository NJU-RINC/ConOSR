from evaluation import openset_eval_contrastive_logits
from utils import load_ImageNet200, get_smooth_labels
from mixup import *




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_root = './data/tinyimagenet/testing/'

    TOTAL_CLASS_NUM = 200
    classid_all = [i for i in range(0, TOTAL_CLASS_NUM)]

    batch_size = 128

    model_folder_path = './saved_models/'
    encoder = torch.load(model_folder_path + 'tinyimagenet_encoder.pt')
    classifier = torch.load(model_folder_path + 'tinyimagenet_classifier.pt')

    classid_known = encoder.classid_list
    classid_unknown = list(set(classid_all) - set(classid_known))

    test_loader_known, _ = load_ImageNet200([test_root], category_indexs=classid_known, train=False, batchSize=batch_size, shuffle=False)
    test_loader_unknown, _ = load_ImageNet200([test_root], category_indexs=classid_unknown, train=False, batchSize=batch_size, shuffle=False)

    _, _, _, AUROC = openset_eval_contrastive_logits(encoder, classifier, test_loader_known, test_loader_unknown)

    print("==> Known Class: ", classid_known)
    print("==> Unknown Class: ", classid_unknown)
    print('unknown detection AUC = {:.3f}%'.format(AUROC * 100))

