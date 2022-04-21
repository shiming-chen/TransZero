import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import TransZero
from dataset import SUNDataLoader
from  helper_func import eval_zs_gzsl

# init wandb from config file
wandb.init(project='TransZero', config='wandb_config/sun_gzsl.yaml')
config = wandb.config
print('Config file from wandb:', config)

# load dataset
dataloader = SUNDataLoader('.', config.device)

# set random seed
seed = config.random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# TransZero model
model = TransZero(config, dataloader.att, dataloader.w2v_att,
                  dataloader.seenclasses, dataloader.unseenclasses).to(config.device)
optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.0001, momentum=0.9)

# main loop
niters = dataloader.ntrain * config.epochs//config.batch_size
report_interval = niters//config.epochs
best_performance = [0, 0, 0, 0]
best_performance_zsl = 0
for i in range(0, niters):
    model.train()
    optimizer.zero_grad()
    
    batch_label, batch_feature, batch_att = dataloader.next_batch(config.batch_size)
    out_package = model(batch_feature)
    
    in_package = out_package
    in_package['batch_label'] = batch_label
    
    out_package=model.compute_loss(in_package)
    loss, loss_CE, loss_cal, loss_reg = out_package['loss'], out_package[
        'loss_CE'], out_package['loss_cal'], out_package['loss_reg']
    
    loss.backward()
    optimizer.step()

    # report result
    if i % report_interval==0:
        print('-'*30)
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
            dataloader, model, config.device, bias_seen=0, bias_unseen=0)
        
        if H > best_performance[2]:
            best_performance = [acc_novel, acc_seen, H, acc_zs]
        if acc_zs > best_performance_zsl:
            best_performance_zsl = acc_zs

        print('iter/epoch=%d/%d | loss=%.3f, loss_CE=%.3f, loss_cal=%.3f, '
              'loss_reg=%.3f | acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | '
              'acc_zs=%.3f' % (
                  i, int(i//report_interval),
                  loss.item(), loss_CE.item(), loss_cal.item(),
                  loss_reg.item(),
                  best_performance[0], best_performance[1],
                  best_performance[2], best_performance_zsl))

        wandb.log({
            'iter': i,
            'loss': loss.item(),
            'loss_CE': loss_CE.item(),
            'loss_cal': loss_cal.item(),
            'loss_reg': loss_reg.item(),
            'acc_unseen': acc_novel,
            'acc_seen': acc_seen,
            'H': H,
            'acc_zs': acc_zs,
            'best_acc_unseen': best_performance[0],
            'best_acc_seen': best_performance[1],
            'best_H': best_performance[2],
            'best_acc_zs': best_performance_zsl
        })
