import torch
import os
import numpy as np


def get_gpu_info():
    gpuinfolist = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free').readlines()
    freemem = [int(gpuinfo.split()[2]) for gpuinfo in gpuinfolist]
    gpuidx = len(freemem) - 1 - np.argmax(list(reversed(freemem)))
    return f'cuda:{gpuidx}'


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size()).fill_(-1)
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


def compute_per_class_acc(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(
            test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()


def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package):
    device = in_package['device']
    per_class_accuracies = torch.zeros(
        target_classes.size()[0]).float().to(device).detach()
    predicted_label = predicted_label.to(device)
    for i in range(target_classes.size()[0]):
        is_class = test_label == target_classes[i]
        per_class_accuracies[i] = torch.div(
            (predicted_label[is_class] == test_label[is_class]).sum().float(), 
            is_class.sum().float())
    return per_class_accuracies.mean().item()


def val_gzsl(test_seen_loader, target_classes, in_package, bias=0):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    test_label = []
    predicted_label = []
    with torch.no_grad():
        for batch, (imgs, labels) in enumerate(test_seen_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            out_package = model(imgs)
            output = out_package['embed']
            output[:, target_classes] = output[:, target_classes]+bias
            predicted_label.append(torch.argmax(output.data, 1))
            test_label.append(labels)
    test_label = torch.cat(test_label, dim=0)
    predicted_label = torch.cat(predicted_label, dim=0)
    acc = compute_per_class_acc_gzsl(
        test_label, predicted_label, target_classes, in_package)
    return acc


def val_zs_gzsl(test_unseen_loader, unseen_classes, in_package, bias=0):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    test_label = []
    predicted_label_gzsl = []
    predicted_label_zsl = []
    predicted_label_zsl_t = []
    with torch.no_grad():
        for batch, (imgs, labels) in enumerate(test_unseen_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            out_package = model(imgs)
            output = out_package['embed']
            output_t = output.clone()
            output_t[:, unseen_classes] = output_t[:,
                                                   unseen_classes] + torch.max(output) + 1
            predicted_label_zsl.append(torch.argmax(output_t.data, 1))
            predicted_label_zsl_t.append(
                torch.argmax(output.data[:, unseen_classes], 1))
            output[:, unseen_classes] = output[:, unseen_classes]+bias
            predicted_label_gzsl.append(torch.argmax(output.data, 1))
            test_label.append(labels)
    test_label = torch.cat(test_label, dim=0)
    predicted_label_gzsl = torch.cat(predicted_label_gzsl, dim=0)
    predicted_label_zsl = torch.cat(predicted_label_zsl, dim=0)
    predicted_label_zsl_t = torch.cat(predicted_label_zsl_t, dim=0)
    acc_gzsl = compute_per_class_acc_gzsl(
        test_label, predicted_label_gzsl, unseen_classes, in_package)
    acc_zs = compute_per_class_acc_gzsl(
        test_label, predicted_label_zsl, unseen_classes, in_package)
    acc_zs_t = compute_per_class_acc(map_label(test_label, unseen_classes).to(
        device), predicted_label_zsl_t, unseen_classes.size(0))
    return acc_gzsl, acc_zs_t


def eval_zs_gzsl(config, dataloader, model, bias_seen=0, bias_unseen=0):
    model.eval()
    test_seen_loader = dataloader.test_seen_loader
    test_unseen_loader = dataloader.test_unseen_loader
    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses
    batch_size = config.batch_size
    in_package = {'model': model, 'device': config.device, 'batch_size': batch_size}
    if config.zsl_task == 'CZSL':
        with torch.no_grad():
            _, acc_zs = val_zs_gzsl(
                test_unseen_loader, unseenclasses, in_package, bias=bias_unseen)
        return acc_zs
    elif config.zsl_task == 'GZSL':
        with torch.no_grad():
            acc_seen = val_gzsl(test_seen_loader, seenclasses,
                                in_package, bias=bias_seen)
            acc_novel, _ = val_zs_gzsl(
                test_unseen_loader, unseenclasses, in_package, bias=bias_unseen)
        if (acc_seen+acc_novel) > 0:
            H = (2*acc_seen*acc_novel) / (acc_seen+acc_novel)
        else:
            H = 0
        return acc_seen, acc_novel, H


def evaluation(config, dataloader, model):
    if config.zsl_task == 'CZSL':
        acc_zs = eval_zs_gzsl(config, dataloader, model)
        print('Results: Acc_ZSL={:.3f}'.format(acc_zs))
    elif config.zsl_task == 'GZSL':
        acc_seen, acc_novel, H = eval_zs_gzsl(config, dataloader, model)
        print('Results: Acc_Unseen={:.3f}, Acc_Seen={:.3f}, H={:.3f}'.format(
            acc_novel, acc_seen, H))
    return
