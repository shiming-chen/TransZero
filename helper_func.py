import torch
import torch.nn.functional as F


def val_gzsl(test_X, test_label, target_classes,in_package,bias = 0):

    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)
            
            out_package = model(input)
            
            output = out_package['S_pp']
            output[:,target_classes] = output[:,target_classes]+bias
            predicted_label[start:end] = torch.argmax(output.data, 1)

            start = end

        acc = compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package)
        return acc


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size()).fill_(-1)
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label


def val_zs_gzsl(test_X, test_label, unseen_classes,in_package,bias = 0):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label_gzsl = torch.LongTensor(test_label.size())
        predicted_label_zsl = torch.LongTensor(test_label.size())
        predicted_label_zsl_t = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)
            
            out_package = model(input)
            output = out_package['S_pp']
            
            output_t = output.clone()
            output_t[:,unseen_classes] = output_t[:,unseen_classes]+torch.max(output)+1
            predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
            predicted_label_zsl_t[start:end] = torch.argmax(output.data[:,unseen_classes], 1) 
            
            output[:,unseen_classes] = output[:,unseen_classes]+bias
            predicted_label_gzsl[start:end] = torch.argmax(output.data, 1)
            
            
            start = end
        acc_gzsl = compute_per_class_acc_gzsl(test_label, predicted_label_gzsl, unseen_classes, in_package)
        acc_zs = compute_per_class_acc_gzsl(test_label, predicted_label_zsl, unseen_classes, in_package)
        acc_zs_t = compute_per_class_acc(map_label(test_label, unseen_classes), predicted_label_zsl_t, unseen_classes.size(0))
        
        return acc_gzsl,acc_zs_t


def compute_per_class_acc(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()
    

def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package):

    device = in_package['device']
    per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(device).detach()

    predicted_label = predicted_label.to(device)

    for i in range(target_classes.size()[0]):

        is_class = test_label == target_classes[i]

        per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(),is_class.sum().float())
    return per_class_accuracies.mean().item()


def eval_zs_gzsl(dataloader,model,device,bias_seen=0, bias_unseen=0, batch_size=50):
    model.eval()
    # print('bias_seen {} bias_unseen {}'.format(bias_seen,bias_unseen))
    test_seen_feature = dataloader.data['test_seen']['resnet_features']
    test_seen_label = dataloader.data['test_seen']['labels'].to(device)
    
    test_unseen_feature = dataloader.data['test_unseen']['resnet_features']
    test_unseen_label = dataloader.data['test_unseen']['labels'].to(device)
    
    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses
    
    batch_size = batch_size
    
    in_package = {'model':model,'device':device, 'batch_size':batch_size}
    
    with torch.no_grad():
        acc_seen = val_gzsl(test_seen_feature, test_seen_label, seenclasses, in_package,bias=bias_seen)
        acc_novel,acc_zs = val_zs_gzsl(test_unseen_feature, test_unseen_label, unseenclasses, in_package,bias = bias_unseen)

    if (acc_seen+acc_novel)>0:
        H = (2*acc_seen*acc_novel) / (acc_seen+acc_novel)
    else:
        H = 0
        
    return acc_seen, acc_novel, H, acc_zs
    

def val_gzsl_k(k,test_X, test_label, target_classes,in_package,bias = 0,is_detect=False):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    n_classes = in_package["num_class"]
    
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        test_label = F.one_hot(test_label, num_classes=n_classes)
        predicted_label = torch.LongTensor(test_label.size()).fill_(0).to(test_label.device)
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)
            
            out_package = model(input)
            
            output = out_package['S_pp']
            output[:,target_classes] = output[:,target_classes]+bias
            _,idx_k = torch.topk(output,k,dim=1)
            if is_detect:
                assert k == 1
                detection_mask=in_package["detection_mask"]
                predicted_label[start:end] = detection_mask[torch.argmax(output.data, 1)]
            else:
                predicted_label[start:end] = predicted_label[start:end].scatter_(1,idx_k,1)
            start = end
        
        acc = compute_per_class_acc_gzsl_k(test_label, predicted_label, target_classes, in_package)
        return acc


def val_zs_gzsl_k(k,test_X, test_label, unseen_classes,in_package,bias = 0,is_detect=False):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    n_classes = in_package["num_class"]
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        
        test_label_gzsl = F.one_hot(test_label, num_classes=n_classes)
        predicted_label_gzsl = torch.LongTensor(test_label_gzsl.size()).fill_(0).to(test_label.device)
        
        predicted_label_zsl = torch.LongTensor(test_label.size())
        predicted_label_zsl_t = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)
            
            out_package = model(input)
            output = out_package['S_pp']
            
            output_t = output.clone()
            output_t[:,unseen_classes] = output_t[:,unseen_classes]+torch.max(output)+1
            predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
            predicted_label_zsl_t[start:end] = torch.argmax(output.data[:,unseen_classes], 1) 
            
            output[:,unseen_classes] = output[:,unseen_classes]+bias
            _,idx_k = torch.topk(output,k,dim=1)
            if is_detect:
                assert k == 1
                detection_mask=in_package["detection_mask"]
                predicted_label_gzsl[start:end] = detection_mask[torch.argmax(output.data, 1)]
            else:
                predicted_label_gzsl[start:end] = predicted_label_gzsl[start:end].scatter_(1,idx_k,1)
            
            start = end
        
        acc_gzsl = compute_per_class_acc_gzsl_k(test_label_gzsl, predicted_label_gzsl, unseen_classes, in_package)
        #print('acc_zs: {} acc_zs_t: {}'.format(acc_zs,acc_zs_t))
        return acc_gzsl,-1


def compute_per_class_acc_k(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()
    

def compute_per_class_acc_gzsl_k(test_label, predicted_label, target_classes, in_package):
    device = in_package['device']
    per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(device).detach()

    predicted_label = predicted_label.to(device)
    
    hit = test_label*predicted_label
    for i in range(target_classes.size()[0]):

        target = target_classes[i]
        n_pos = torch.sum(hit[:,target])
        n_gt = torch.sum(test_label[:,target])
        per_class_accuracies[i] = torch.div(n_pos.float(),n_gt.float())
        #pdb.set_trace()
    return per_class_accuracies.mean().item()


def eval_zs_gzsl_k(k,dataloader,model,device,bias_seen,bias_unseen,is_detect=False):
    model.eval()
    print('bias_seen {} bias_unseen {}'.format(bias_seen,bias_unseen))
    test_seen_feature = dataloader.data['test_seen']['resnet_features']
    test_seen_label = dataloader.data['test_seen']['labels'].to(device)
    
    test_unseen_feature = dataloader.data['test_unseen']['resnet_features']
    test_unseen_label = dataloader.data['test_unseen']['labels'].to(device)
    
    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses
    
    batch_size = 100
    n_classes = dataloader.ntrain_class+dataloader.ntest_class
    in_package = {'model':model,'device':device, 'batch_size':batch_size,'num_class':n_classes}
    
    if is_detect:
        print("Measure novelty detection k: {}".format(k))
        
        detection_mask = torch.zeros((n_classes,n_classes)).long().to(dataloader.device)
        detect_label = torch.zeros(n_classes).long().to(dataloader.device)
        detect_label[seenclasses]=1
        detection_mask[seenclasses,:] = detect_label
        
        detect_label = torch.zeros(n_classes).long().to(dataloader.device)
        detect_label[unseenclasses]=1
        detection_mask[unseenclasses,:]=detect_label
        in_package["detection_mask"]=detection_mask
    
    with torch.no_grad():
        acc_seen = val_gzsl_k(k,test_seen_feature, test_seen_label, seenclasses, in_package,bias=bias_seen,is_detect=is_detect)
        acc_novel,acc_zs = val_zs_gzsl_k(k,test_unseen_feature, test_unseen_label, unseenclasses, in_package,bias = bias_unseen,is_detect=is_detect)

    if (acc_seen+acc_novel)>0:
        H = (2*acc_seen*acc_novel) / (acc_seen+acc_novel)
    else:
        H = 0
        
    return acc_seen, acc_novel, H, acc_zs
