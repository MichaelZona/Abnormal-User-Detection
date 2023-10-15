import pickle
import json
import time
import torch
import random
import argparse
import numpy as np
import sys
import pandas as pd


np.set_printoptions(threshold = sys.maxsize)

parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--cuda', type=int, default=-1, help='Which GPU to run on (-1 for using CPU, 9 for not specifying which GPU to use.)')
parser.add_argument('--dataSet', type=str, default='weibo_s')
parser.add_argument('--file_paths', type=str, default='file_paths.json')
parser.add_argument('--config_dir', type=str, default='../configs')      #改变路径到上级文件夹
parser.add_argument('--logs_dir', type=str, default='./logs')
parser.add_argument('--out_dir', default='../results')
parser.add_argument('--name', type=str, default='debug')

parser.add_argument('--seed', type=int, default=2019)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--b_sz', type=int, default=100)
parser.add_argument('--n_gnnlayer', type=int, default=2)
parser.add_argument('--out_emb_size', type=int, default=128)
parser.add_argument('--tvt_split', type=int, default=0, help='Which of the 5 presets of train-validation-test data splits to use. (0~4), used for bitcoin dataset.')
parser.add_argument('--C', type=float, default=20)
parser.add_argument('--n_block', type=int, default=-1)
parser.add_argument('--thresh', type=float, default=-1)
parser.add_argument('--a_loss_weight', type=float, default=4)
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--simi_func', type=str, default='cos')
parser.add_argument('--learn_method', type=str, default='bigal')
parser.add_argument('--loss', type=str, default='1010')
parser.add_argument('--a_loss', type=str, default='none')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--over_sample', type=str, default='none')
parser.add_argument('--feature', type=str, default='all')
parser.add_argument('--biased_rw', action='store_true')
parser.add_argument('--cluster_aloss', action='store_true')
parser.add_argument('--best_rw', action='store_true')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--gat', action='store_true')
parser.add_argument('--no_save_embs', action='store_true')
parser.add_argument('--batch_output', action='store_true')
parser.add_argument('--nognn', action='store_true')
parser.add_argument('--noblock', action='store_true')
args = parser.parse_args()
args.argv = sys.argv
args.name = f'{args.name}_{args.dataSet}_{args.learn_method}_{args.feature}_loss{args.loss}_{args.n_gnnlayer}layers_simi-{args.simi_func}_{args.a_loss}_{time.strftime("%m-%d_%H-%M")}'
args.out_path  =  args.out_dir + '/' + args.name

print(f'{args.config_dir}/{args.file_paths}')


def text_creat(name, msg):
    data_txt_file = 'E:\\GAL\Graph-Anomaly-Loss-master\data_txt\\'
    print("Craet sucessfully")
    full_path = data_txt_file + name + '.txt'
    file  =open(full_path, 'w')
    print(full_path)
    file.write(msg)


ds = args.dataSet

file_paths = json.load(open(f'{args.config_dir}/{args.file_paths}'))       #存储文件data路径
# 读取数组中的目标文件路径
graph_u2p_file = file_paths[ds]['graph_u2p']
labels_file = file_paths[ds]['labels']
features_bow_file = file_paths[ds]['features_bow']
features_loc_file = file_paths[ds]['features_loc']
clusters_files = file_paths[ds]["lockinfer"]["a_cluster"]
labels_a_files = file_paths[ds]["lockinfer"]["a_label"]


graph_u2p = pickle.load(open(graph_u2p_file, 'rb'), encoding='latin1')
#graph_u2p = str(graph_u2p)

'''
print(type(graph_u2p))
print(graph_u2p[1, : ].nonzero()[0])
print(graph_u2p[1, : ].nonzero()[1])
'''
#for j in range():
#   print(graph_u2p[)
'''
labels = pickle.load(open(labels_file, 'rb'), encoding='latin1')
print(np.shape(labels))
feat_bow = pickle.load(open(features_bow_file, 'rb'), encoding='latin1')
print(np.shape(feat_bow))
feat_loc = pickle.load(open(features_loc_file, 'rb'), encoding='latin1')
print(np.shape(feat_loc)
'''
clusters_neighbor = pickle.load(open(clusters_files, 'rb'))
print('*clusters_neighbor[:-1]', *clusters_neighbor[:-1])
print(np.shape(clusters_neighbor[:-1]))
#print(type(*clusters_neighbor))
print('clusters_neighbor[-1]', )
print(type(clusters_neighbor[-1]))
#print(np.shape(clusters_neighbor))
print('tyep:', type(clusters_neighbor))
print(len(clusters_neighbor))
labels_a_data = pickle.load(open(labels_a_files, 'rb'), encoding = 'latin1')
print('type:', type(labels_a_data))
print(np.shape(labels_a_data))
#print(np.shape(labels_a_data))
'''
graph_u2p = str(graph_u2p)
labels = str(labels)
feat_bow = str(feat_bow)
feat_loc = str(feat_loc)
'''

'''
clusters_neighbor = str(clusters_neighbor)
print('tyep:', type(clusters_neighbor))
print(np.shape(clusters_neighbor))
labels_a_data = str(labels_a_data)
print('type:', type(labels_a_data))
print(np.shape(labels_a_data))
'''
'''
graph_u2p_file_txt = graph_u2p_file.split("/")[-1].split(".")[0]
print(labels_file)
labels_file_txt = labels_file.split("/")[-1].split(".")[0]
feat_bow_file_txt = features_bow_file.split("/")[-1].split(".")[0]
feat_loc_file_txt = features_loc_file.split("/")[-1].split(".")[0]
clusters_files_txt = clusters_files.split("/")[-1].split(".")[0]
labels_a_files_txt = labels_a_files.split("/")[-1].split(".")[0]
'''
#print(graph_u2p_file_txt)

'''
text_creat(graph_u2p_file_txt , graph_u2p)
text_creat(labels_file_txt , labels)
text_creat(feat_bow_file_txt , feat_bow)
text_creat(feat_loc_file_txt , feat_loc)
text_creat(clusters_files_txt, clusters_neighbor)
text_creat(labels_a_files_txt, labels_a_data)
'''

'''label_a_file = "./data/weibo_s/weibo_s_labels_fraudar.pkl"
labels_a = pickle.load(open(label_a_file, 'rb')).astype(int)
print(labels_a)
'''

