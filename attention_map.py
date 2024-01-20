import os
import torch
import torch.optim as optim
import torch.nn as nn
from core.DAZLE_plot import DAZLE
from core.CUBDataLoader import CUBDataLoader
from core.helper_func import eval_zs_gzsl
from global_setting import NFS_path
import numpy as np
import wandb
from get_gpu_info import get_gpu_info
from PIL import Image
import matplotlib.pyplot as plt
import skimage
from sklearn.manifold import TSNE
from torchvision import transforms


data_transforms = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor()])


def dazle_visualize_attention_np_global_448(img_ids,alphas_1,alphas_2,attr_name,save_path=None):
    #  alphas_1: [bir]     alphas_2: [bi]
    n = img_ids.shape[0]
    image_size = 448          #one side of the img
    assert alphas_1.shape[1] == alphas_2.shape[1] == len(attr_name)
    r = alphas_1.shape[2]
    h = w =  int(np.sqrt(r))
    for i in range(n):
        fig=plt.figure(i,figsize=(33, 5))
        file_path=img_ids[i]#.decode('utf-8')
        img_name = file_path.split("/")[-1]
        alpha_1 = alphas_1[i]           #[ir]
        alpha_2 = alphas_2[i]           #[i]
        # score = S[i]
        # Plot original image
        image = Image.open(file_path)
        if image.mode == 'L':
            image=image.convert('RGB')
        image = data_transforms(image)
        image = image.permute(1,2,0) #[224,244,3] <== [3,224,224] 
        idx = 1
        ax = plt.subplot(1, 11, 1)
        idx += 1
        plt.imshow(image)
        # ax.set_title(os.path.splitext(img_name)[0],{'fontsize': 13})
        plt.axis('off')
        
        idxs_top_p=np.argsort(-alpha_2)[:10]
        idxs_top_g=np.argsort(-alpha_2)[:200]
        # idxs_top_n=np.argsort(alpha_2)[:3]
            
        #pdb.set_trace()
        for idx_ctxt,idx_attr in enumerate(idxs_top_p):
            ax=plt.subplot(1, 11, idx)
            idx += 1
            plt.imshow(image)
            alp_curr = alpha_1[idx_attr,:].reshape(14,14)
            alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=image_size/h, sigma=10,multichannel=False)
            plt.imshow(alp_img, alpha=0.5, cmap='jet')
            # ax.set_title("{}\n{}\n{}-{}".format(attr_name[idx_attr],alpha_2[idx_attr],score[idx_attr],attr[idx_attr]),{'fontsize': 10})
            # ax.set_title("{}\n(Score = {:.2f})".format(attr_name[idx_attr].title().replace(
            #     ' ', ''), alpha_2[idx_attr]), {'fontsize': 19})
            ax.set_title("{}\n(Score = {:.1f})".format(' '.join(attr_name[idx_attr].split()[:2]).title(
            ) + '\n' + ' '.join(attr_name[idx_attr].split()[2:]).title(), alpha_2[idx_attr]), {'fontsize': 25})

            plt.axis('off')

        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path+img_name,dpi=200)
            plt.close()


def dazle_visualize_attention_np_global_448_small(img_ids,alphas_1,alphas_2,attr_name,save_path=None):
    #  alphas_1: [bir]     alphas_2: [bi]
    n = img_ids.shape[0]
    image_size = 448          #one side of the img
    assert alphas_1.shape[1] == alphas_2.shape[1] == len(attr_name)
    r = alphas_1.shape[2]
    h = w =  int(np.sqrt(r))
    for i in range(n):
        fig=plt.figure(i,figsize=(33, 4))
        file_path=img_ids[i]#.decode('utf-8')
        img_name = file_path.split("/")[-1]
        alpha_1 = alphas_1[i]           #[ir]
        alpha_2 = alphas_2[i]           #[i]
        # score = S[i]
        # Plot original image
        image = Image.open(file_path)
        if image.mode == 'L':
            image=image.convert('RGB')
        image = data_transforms(image)
        image = image.permute(1,2,0) #[224,244,3] <== [3,224,224] 
        idx = 1
        ax = plt.subplot(1, 11, 1)
        idx += 1
        plt.imshow(image)
        # ax.set_title(os.path.splitext(img_name)[0],{'fontsize': 13})
        plt.axis('off')
        
        idxs_top_p=np.argsort(-alpha_2)[:10]
        idxs_top_g=np.argsort(-alpha_2)[:200]
        # idxs_top_n=np.argsort(alpha_2)[:3]
            
        #pdb.set_trace()
        for idx_ctxt,idx_attr in enumerate(idxs_top_p):
            ax=plt.subplot(1, 11, idx)
            idx += 1
            plt.imshow(image)
            alp_curr = alpha_1[idx_attr,:].reshape(14,14)
            alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=image_size/h, sigma=10,multichannel=False)
            plt.imshow(alp_img, alpha=0.5, cmap='jet')
            # ax.set_title("{}\n{}\n{}-{}".format(attr_name[idx_attr],alpha_2[idx_attr],score[idx_attr],attr[idx_attr]),{'fontsize': 10})
            ax.set_title("{}\n(Score = {:.2f})".format(attr_name[idx_attr].title().replace(
                ' ', ''), alpha_2[idx_attr]), {'fontsize': 18})
            # ax.set_title("{}\n(Score = {:.1f})".format(' '.join(attr_name[idx_attr].split()[:2]).title(
            # ) + '\n' + ' '.join(attr_name[idx_attr].split()[2:]).title(), alpha_2[idx_attr]), {'fontsize': 20})

            plt.axis('off')


        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path+img_name,dpi=200)
            plt.close()


def plot_att(config):
    model_path = 'saved_model/CUB_weights_H-0.688.pth'

    config.dataset = 'CUB'
    config.num_class = 200
    config.num_attribute = 312
    if config.img_size == 224: config.resnet_region = 49
    elif config.img_size == 448: config.resnet_region = 196

    print('Config file from wandb:', config)

    if config.device == 'auto':
        device = get_gpu_info()
    else:
        device = config.device
    dataloader = CUBDataLoader(NFS_path, device,
                            is_unsupervised_attr=False, is_balance=False,
                            img_size=config.img_size, use_unzip=config.use_unzip)
    dataloader.augment_img_path()
    torch.backends.cudnn.benchmark = True

    def get_lr(optimizer):
        lr = []
        for param_group in optimizer.param_groups:
            lr.append(param_group['lr'])
        return lr

    seed = config.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    batch_size = config.batch_size
    nepoches = config.epochs
    niters = dataloader.ntrain * nepoches//batch_size
    dim_f = 2048
    dim_v = 300
    init_w2v_att = dataloader.w2v_att
    att = dataloader.att
    normalize_att = dataloader.normalize_att

    trainable_w2v = config.trainable_w2v
    # CE loss和cal loss的超参数
    lambda_ = config.lambda_
    bias = 0
    prob_prune = 0
    # uniform DAZLE attention的选项
    uniform_att_1 = False
    uniform_att_2 = False

    seenclass = dataloader.seenclasses
    unseenclass = dataloader.unseenclasses
    desired_mass = 1
    report_interval = niters//nepoches

    model = DAZLE(config, dim_f,dim_v,init_w2v_att,att,normalize_att,
                seenclass,unseenclass,
                lambda_,
                trainable_w2v,normalize_V=False,normalize_F=True,is_conservative=True,
                uniform_att_1=uniform_att_1,uniform_att_2=uniform_att_2,
                prob_prune=prob_prune,desired_mass=desired_mass, is_conv=False,
                is_bias=config.is_bias)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    num_parameters = sum([p.numel() for p in model.parameters()]) * 1e-6
    print('model parameters: %.3fM' % num_parameters)

    file_list = [
        'Acadian_Flycatcher_0008_795599',
        'American_Goldfinch_0092_32910',
        'Canada_Warbler_0117_162394',
        'Carolina_Wren_0006_186742',
        'Vesper_Sparrow_0090_125690',
        'Western_Gull_0058_53882',
        'White_Throated_Sparrow_0128_128956',
        'Winter_Wren_0118_189805',
        'Yellow_Breasted_Chat_0044_22106',
        'Elegant_Tern_0085_151091',
        'European_Goldfinch_0025_794647',
        'Florida_Jay_0008_64482',
        'Fox_Sparrow_0025_114555',
        'Grasshopper_Sparrow_0053_115991',
        'Grasshopper_Sparrow_0107_116286',
        'Gray_Crowned_Rosy_Finch_0036_797287'
    ]
    
    for filename in file_list:
        for i, id in enumerate(dataloader.seenclasses):
            # if i == 5:
            #     raise Exception
            id = id.item()
            (batch_label, batch_feature, batch_files, batch_att) = dataloader.next_batch_img(
                batch_size=10, class_id=id, is_trainset=False)

            if filename not in str(batch_files):
                continue

            idx = [filename in str(f) for f in batch_files]
            batch_feature = batch_feature[idx]
            batch_files = batch_files[idx]

            model.eval()
            with torch.no_grad():
                out_package = model(batch_feature)

            # attention map of DAZLE
            dazle_visualize_attention_np_global_448_small(batch_files,
                                out_package['att'].cpu().numpy(),
                                out_package['dazle_embed'].cpu().numpy(),
                                dataloader.attr_name,
                                'plot/atten_fig/')



if __name__ == '__main__':

    wandb.init(project='ZSL_DALZE_Transformer_GA', config='config_cub.yaml', allow_val_change=True)
    config = wandb.config
    plot_att(config)


