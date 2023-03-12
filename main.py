import torch
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision.models import resnet50, ResNet50_Weights
from clip_main import clip
from PIL import Image
import matplotlib.pyplot as plt

import os
import argparse

import dissection


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', dest='k', required=True, help="k-th neuron", type=int)
    parser.add_argument('-ord', dest='ord', required=False, help="Order of the distance norm")
    parser.add_argument('-concept_set', dest='S', required=False, default='data/20k.txt', help="Concept set file")
    parser.add_argument('-D_probe', dest='probing_dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('-network', dest='network', default='resnet50', choices=['resnet50'])
    args = parser.parse_args()
    return args



def main():

    args = argument_parser()

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Download the CIFAR100 dataset (contains 60000 images from 100 classes)
    if args.probing_dataset == 'cifar100':
        D_probe = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    elif args.probing_dataset =='cifar10':
        D_probe = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)
    else:
        print('Choose a valid probing dataset.')

    if args.network == 'resnet50': # ResNet-50
        weights=ResNet50_Weights.DEFAULT
        probed_network = resnet50(weights=weights)
        probed_network.eval()
        preprocess_probed = weights.transforms()

    else:
        print('Choose a valid probed network.')

    #print(probed_network)
    #print(model.visual)


    label = dissection.neuron_label(k=args.k, 
                                    concept_set=args.S, 
                                    probing_dataset=D_probe, 
                                    probed_network=probed_network,
                                    preprocess_probed=preprocess_probed,
                                    model=model, 
                                    preprocess=preprocess, 
                                    device=device, 
                                    ord=args.ord,
                                    M=10000, 
                                    N=1000)
    
    print("Label of neuron {} : {}".format(args.k, label))


if __name__ == '__main__':
    main()