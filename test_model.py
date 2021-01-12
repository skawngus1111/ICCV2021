import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from Dataset import CustomDataset
from Classifier import Classifier

import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    normalize
])

model_list = ['resnet18', 'resnet50', 'densenet121']
augment_list = ['ours']

def evaluate(device, model, dataloader) :
    print('\n======= Testing... =======\n')
    model.eval()
    with torch.no_grad():
        y_true = list()
        y_pred = list()
        for batch_idx, (img, target) in enumerate(dataloader):
            if (batch_idx + 1) % 250 == 0:
                print("{}/{}({}%) COMPLETE".format(
                    batch_idx + 1, len(dataloader), (batch_idx + 1) / len(dataloader) * 100))
            img, target = img.to(device), target.to(device)
            out = model(img)
            y_true.append(target.item())
            y_pred.append(torch.argmax(out, dim=1).item())
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        y_true_b, y_pred_b = label_binarize(y_true, classes=[0, 1, 2, 3]), label_binarize(y_pred, classes=[0, 1, 2, 3])

        auc = np.mean(roc_auc_score(y_true_b, y_pred_b,  average=None))
        precision = np.mean(precision_score(y_true_b, y_pred_b, average=None))
        recall = np.mean(recall_score(y_true_b, y_pred_b, average=None))
        f1_score = 2 * (precision * recall) / (precision + recall)
        total_correct = (y_true == y_pred).sum()
        test_acc = np.round(total_correct / float(len(dataloader)), 4)

        return (test_acc, precision, recall, f1_score, auc)

def reporter(results) :
    for augment in results.keys() :
        print('\n============ TEST REPORT | {} ============\n'.format(augment))
        for model_name in results[augment] :
            print('\n++++++++++++++ {} ++++++++++++++\n'.format(model_name))
            print("accuracy mean = {}, std = {}".format(results[augment][model_name]['accuracy'][0], results[augment][model_name]['accuracy'][1]))
            print("precision mean = {}, std = {}".format(results[augment][model_name]['precision'][0], results[augment][model_name]['precision'][1]))
            print("recall mean = {}, std = {}".format(results[augment][model_name]['recall'][0], results[augment][model_name]['recall'][1]))
            print("f1_score mean = {}, std = {}".format(results[augment][model_name]['f1_score'][0], results[augment][model_name]['f1_score'][1]))
            print("auc mean = {}, std = {}\n".format(results[augment][model_name]['auc'][0], results[augment][model_name]['auc'][1]))
            print('\n++++++++++++++ {} ++++++++++++++\n'.format(model_name))
        print('\n============ TEST REPORT | {} ============\n'.format(augment))

def main(device) :
    dataset = CustomDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = dict()
    for augment in augment_list:
        if augment not in results.keys() :
            results[augment] = dict()
        for model_name in model_list :
            if model_name not in results[augment].keys():
                results[augment][model_name] = dict()
            accuracies, precisions, recalls, f1_scores, aucs = list(), list(), list(), list(), list()
            for fold in range(1, 6) :
                print("Fold = {}".format(fold))
                model_path = os.path.join('./model_save', augment, model_name, '0.01_030(030)(fold={}).pth'.format(fold))
                check_point = torch.load(model_path)
                del check_point['optimizer']
                del check_point['model_name']
                del check_point['learning_rate']
                del check_point['epochs']
                del check_point['losses_dict']
                del check_point['acc_dict']
                # try :
                #     torch.save(check_point, os.path.join('model_save5/', augment, model_name,
                #                                          '0.01_030(030)(fold={}).pth'.format(fold)))
                # except FileNotFoundError :
                #     os.makedirs(os.path.join('model_save5/', augment, model_name))
                #     torch.save(check_point, os.path.join('model_save5/', augment, model_name,
                #                                          '0.01_030(030)(fold={}).pth'.format(fold)))
                model = check_point['model']
                # model = Classifier(model_name=model_name, num_of_classes=4)
                model.load_state_dict(check_point['model_state_dict'])
                accuracy, precision, recall, f1_score, auc = evaluate(device, model, dataloader)

                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1_score)
                aucs.append(auc)
            print('\n============ TEST REPORT ============\n')
            print("test accuracy = ", accuracies)
            print("test precision = ", precisions)
            print("test recall = ", recalls)
            print("test f1_score = ", f1_scores)
            print("test auc = ", aucs)

            print("\ntest accuracy mean = {}, std = {}".format(np.round(np.mean(accuracies), 4), np.round(np.std(accuracies), 4)))
            print("test precision mean = {}, std = {}".format(np.round(np.mean(precisions), 4), np.round(np.std(precisions), 4)))
            print("test recall mean = {}, std = {}".format(np.round(np.mean(recalls), 4), np.round(np.std(recalls), 4)))
            print("test f1 score mean = {}, std = {}".format(np.round(np.mean(f1_scores), 4), np.round(np.std(f1_scores), 4)))
            print("test auc mean = {}, std = {}".format(np.round(np.mean(aucs), 4), np.round(np.std(aucs), 4)))

            results[augment][model_name]['accuracy'] = [np.round(np.mean(accuracies), 4), np.round(np.std(accuracies), 4)]
            results[augment][model_name]['precision'] = [np.round(np.mean(precisions), 4), np.round(np.std(precisions), 4)]
            results[augment][model_name]['recall'] = [np.round(np.mean(recalls), 4), np.round(np.std(recalls), 4)]
            results[augment][model_name]['f1_score'] = [np.round(np.mean(f1_scores), 4), np.round(np.std(f1_scores), 4)]
            results[augment][model_name]['auc'] = [np.round(np.mean(aucs), 4), np.round(np.std(aucs), 4)]
            print('\n============ TEST REPORT ============\n')

    reporter(results)

    return results

if __name__=="__main__" :
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    results = main(device)
    print(results)