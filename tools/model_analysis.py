from glob import glob
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Model')
    # currently only support plot curve and calculate average train time
    parser.add_argument(
        '--model_dir', 
        type=str, 
        help='path of models')
    parser.add_argument(
        '--key_name', 
        type=str,
        default='alpha', 
        help='name of param')
    args = parser.parse_args()
    return args

def main():
    # get args
    args = parse_args()

    paths = glob(os.path.join(args.model_dir, 'epoch_*.pth'))
    keys = []

    # get keys
    path = os.path.join(args.model_dir, 'epoch_{}.pth'.format(1))
    data = torch.load(path, map_location='cpu')['state_dict']
    for key in data.keys():
        if args.key_name in key:
            keys.append(key)

    res = np.zeros((len(paths)+1, len(keys)))
    for id in tqdm(range(1, len(paths)+1, 1)):
        path = os.path.join(args.model_dir, 'epoch_{}.pth'.format(id))
        data = torch.load(path, map_location='cpu')['state_dict']
        for j,keyx in enumerate(keys):
            res[id,j] = data[keyx].numpy()
        del data

    np.savetxt(os.path.join(args.model_dir, 'param_alpha.txt'), res)

    plt.figure(figsize=(10,5))
    colors = ['darkorange', 'limegreen', 'lightcoral', 'deeppink', 'royalbluw']
    for i in range(len(keys)):
        stage = keys[i].split('lateral_convs.')[1][:1]
        plt.plot(res[1:,i], '-', color=colors[i], linewidth=2, label=r'FPN stage{}: $\alpha$'.format(stage))
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel(r'$\alpha$', fontsize=18)
    plt.xticks(range(1,len(paths)//5*5+6,5))

    plt.title(r'Learning Process of Attention Bypass Weight Param: {}'.format(r'$\alpha$'), fontsize=18)
    plt.legend(fontsize=18)

    plt.savefig(os.path.join(args.model_dir, 'plot_alpha.png'))

if __name__ == '__main__':
    main()