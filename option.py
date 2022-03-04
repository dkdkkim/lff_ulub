# -*- coding: utf-8 -*-
import argparse, os, json
from xmlrpc.client import boolean

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=0, type=int, help="GPU number")
parser.add_argument('--batch', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr_init', type=float, default=0.001)
parser.add_argument('--exp_name', type=str, default='exp_test', help='title of experiment')
parser.add_argument('--cls_type', type=str, help='gender or skintone')
parser.add_argument('--data_path', type=str, help='path of dataset')
parser.add_argument('--log_path', type=str, help='path of log')
parser.add_argument('--extractor_freeze', type=boolean, default=True, help='whether feature extractor freeze or not')

def get_option():
    option = parser.parse_args()
    return option

def save_option(save_dir, option):
    option_path = os.path.join(save_dir, "options.json")

    with open(option_path, 'w') as fp:
        json.dump(option.__dict__, fp, indent=4, sort_keys=True)