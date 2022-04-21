import torch
from utils import evaluation
from model import TransZero
from dataset import UNIDataloader
import argparse
import json


def run_test(config):
    print(f'Dataset: {config.dataset}\nSetting: {config.zsl_task}')
    # dataset
    dataloader = UNIDataloader(config)
    # model
    model = TransZero(config)
    # load parameters
    model_dict = model.state_dict()
    saved_dict = torch.load(config.saved_model)
    saved_dict = {k: v for k, v in saved_dict.items() if k in model_dict}
    model_dict.update(saved_dict)
    model.load_state_dict(model_dict)
    model.to(config.device)
    # evaluation
    evaluation(config, dataloader, model)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/CUB_CZSL.json')
    config = parser.parse_args()
    with open(config.config, 'r') as f:
        config.__dict__ = json.load(f)
    run_test(config)
