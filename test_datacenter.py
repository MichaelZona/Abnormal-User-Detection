import os
import sys

sys.path.insert(0, '../')       #添加目录优先被其他目录检查

import argparse
import copy
import time
import math
import torch
import pickle
import pathlib
import powerlaw
import numpy as np
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter

from src.utils import *


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
args.embedding_ready_methods = set(['feature', 'rand', 'rand2', 'svdgnn', 'lr'])
if os.path.isdir(args.out_path) is False:
    os.mkdir(args.out_path)
print('learn_method: ', args.learn_method)

def text_creat(name, msg):
    data_txt_file = 'E:\\GAL\Graph-Anomaly-Loss-master\data_txt\\'
    print("Craet sucessfully")
    full_path = data_txt_file + name + '.txt'
    file  =open(full_path, 'w')
    print(full_path)
    file.write(msg)


class DataCenter(object):
    """docstring for DataCenter"""
    def __init__(self, args):
        self.args = args
        self.file_paths = json.load(open(f'{args.config_dir}/{args.file_paths}'))       #存储文件data路径
        self.logger = getLogger(args.name, args.out_path, args.config_dir)      #创建日志

        self.load_dataSet(args.dataSet)

    def load_dataSet(self, dataSet):
        self.logger.info(f'Dataset: {dataSet}')     #level:info

        ###判断数据库的挑选
        if dataSet.startswith('bitcoin'):
            if self.args.learn_method == 'gnn':
                ds = dataSet
                #读取数组中的目标文件路径
                graph_u2p_file = self.file_paths[ds]['graph_u2p']
                labels_file = self.file_paths[ds]['labels']
                features_file = self.file_paths[ds]['features']
                data_split_file = self.file_paths[ds]['trainvalitest_indexes']

                graph_u2p = pickle.load(open(graph_u2p_file, 'rb'))
                labels = pickle.load(open(labels_file, 'rb'))
                feat_data = pickle.load(open(features_file, 'rb'))

                m, n = np.shape(graph_u2p)
                adj_lists = {}
                for i in range(m):
                    adj_lists[i] = set(m + graph_u2p[i,:].nonzero()[1])
                for j in range(n):
                    adj_lists[j+m] = set(graph_u2p[:,j].nonzero()[0])

                feat_data = np.concatenate((feat_data, np.zeros((n, np.shape(feat_data)[1]))), 0)
                assert np.shape(feat_data)[0] == m+n

                train_indexs = np.arange(len(labels))

                if os.path.isfile(data_split_file):
                    self.logger.info('loaded train-vali-test data splits.')
                    data_splits = pickle.load(open(data_split_file, 'rb'))
                else:
                    node_w_labels = np.where(labels>=0)[0]
                    data_splits = self._split_data_limited(node_w_labels)
                    pickle.dump(data_splits, open(data_split_file, "wb"))
                    self.logger.info('generated train-vali-test data splits and saved for future use.')

                self.logger.info(f'using data split {self.args.tvt_split}')
                test_indexs, val_indexs, train_indexs_cls = data_splits[self.args.tvt_split]

                assert len(feat_data) == len(labels)+n == len(adj_lists)
                train_indexs = np.arange(len(labels))

                user_id_max = len(labels)
                graph_simi = np.ones((m+n, m+n))

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)
                setattr(self, dataSet+'_train_cls', train_indexs_cls)
                setattr(self, dataSet+'_trainable', train_indexs)
                setattr(self, dataSet+'_useridmax', user_id_max)

                setattr(self, dataSet+'_feats', feat_data)
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adj_lists)
                setattr(self, dataSet+'_best_adj_lists', adj_lists)
                setattr(self, dataSet+'_simis', graph_simi)
            else:
                ds = dataSet
                graph_u2p_file = self.file_paths[ds]['graph_u2p']
                graph_u2u_file = self.file_paths[ds]['graph_u2u']
                labels_file = self.file_paths[ds]['labels']
                features_file = self.file_paths[ds]['features']
                data_split_file = self.file_paths[ds]['trainvalitest_indexes']

                if self.args.simi_func == 'pcc' or self.args.simi_func == 'rpcc':
                    _get_simi = getPCC
                    graph_simi_file = self.file_paths[ds]['graph_u2u_pcc']
                elif self.args.simi_func == 'acos':
                    _get_simi = getACOS
                    graph_simi_file = self.file_paths[ds]['graph_u2u_acos']
                elif self.args.simi_func == 'cos':
                    _get_simi = getCOS
                    graph_simi_file = self.file_paths[ds]['graph_u2u_cos']
                else:
                    self.logger.error('Invalid user-user similarity function.')

                graph_u2p = pickle.load(open(graph_u2p_file, 'rb'))
                graph_u2u = pickle.load(open(graph_u2u_file, 'rb'))
                labels = pickle.load(open(labels_file, 'rb'))
                feat_data = pickle.load(open(features_file, 'rb'))

                if self.args.feature == 'deepwalk':
                    embs_file = self.file_paths[ds]['deepwalk']
                    feat_data = pickle.load(open(embs_file, 'rb'))
                elif self.args.feature == 'bine':
                    embs_file = self.file_paths[ds]['bine']
                    feat_data = pickle.load(open(embs_file, 'rb'))
                elif self.args.feature == 'line':
                    embs_file = self.file_paths[ds]['line']
                    feat_data = pickle.load(open(embs_file, 'rb'))
                elif self.args.feature.startswith('n2v'):
                    n2v_index = self.args.feature.split('_')[-1]
                    embs_file = self.file_paths[ds]['n2v'][n2v_index]
                    feat_data = pickle.load(open(embs_file, 'rb'))

                graph_simi = copy.deepcopy(graph_u2u)
                if self.args.learn_method not in self.args.embedding_ready_methods:
                    self.logger.info(f'Using {self.args.simi_func} user-user similarity.')
                    if pathlib.Path(graph_simi_file).is_file():
                        graph_simi = pickle.load(open(graph_simi_file, 'rb'))
                        self.logger.info('Loaded user-user similarity graph.')
                    else:
                        nz_entries = np.asarray(graph_u2u.nonzero()).T
                        self.logger.info(f'Calculating user-user {self.args.simi_func} similarity graph, {len(nz_entries)} edges to go...')
                        sz = 1000
                        n_batch = math.ceil(len(nz_entries) / sz)
                        batches = np.array_split(nz_entries, n_batch)
                        pool = Pool()
                        results = pool.map(get_simi_single_iter, [(entries_batch, graph_u2p, _get_simi) for entries_batch in batches])
                        results = list(zip(*results))
                        row = np.concatenate(results[0])
                        col = np.concatenate(results[1])
                        dat = np.concatenate(results[2])
                        graph_simi = csr_matrix((dat, (row, col)), shape=np.shape(graph_u2u))
                        if np.max(graph_simi) > 1:
                            graph_simi = graph_simi / np.max(graph_simi)
                        pickle.dump(graph_simi, open(graph_simi_file, "wb"))
                        self.logger.info('Calculated user-user similarity and saved it for catch.')

                if self.args.simi_func == 'rpcc':
                    graph_simi[graph_simi<0] = 0

                # get the user-user adjacency list
                adj_lists = {}
                for i in range(np.shape(graph_u2u)[0]):
                    adj_lists[i] = set(graph_u2u[i,:].nonzero()[1])

                # best neighbors list
                best_adj_lists = {}
                for i in range(np.shape(graph_u2u)[0]):
                    if len(adj_lists[i]) <= 15:
                        best_adj_lists[i] = adj_lists[i]
                    else:
                        adjs = graph_u2u[i].toarray()[0]
                        best_adj_lists[i] = set(np.argpartition(adjs, -15)[-15:])

                if self.args.a_loss != 'none':
                    self.logger.info(f'using {self.args.a_loss} anomaly loss.')
                    label_a_file = self.file_paths[ds][self.args.a_loss]
                    labels_a = pickle.load(open(label_a_file, 'rb'))
                    clusters_a = labels_a
                    u2size_of_cluster = {}
                    for u in range(np.shape(graph_u2p)[0]):
                        u2size_of_cluster[u] = np.sum(labels_a == labels_a[u])

                assert len(feat_data) == len(labels) == len(adj_lists)
                train_indexs = np.arange(len(labels))

                if os.path.isfile(data_split_file):
                    self.logger.info('loaded train-vali-test data splits.')
                    data_splits = pickle.load(open(data_split_file, 'rb'))
                else:
                    node_w_labels = np.where(labels>=0)[0]
                    data_splits = self._split_data_limited(node_w_labels)
                    pickle.dump(data_splits, open(data_split_file, "wb"))
                    self.logger.info('generated train-vali-test data splits and saved for future use.')

                self.logger.info(f'using data split {self.args.tvt_split}')
                test_indexs, val_indexs, train_indexs_cls = data_splits[self.args.tvt_split]

                user_id_max = len(labels)
                self.logger.info(f'distr: train {np.sum(labels[train_indexs_cls])}/{len(train_indexs_cls)}, vali: {np.sum(labels[val_indexs])}/{len(val_indexs)}, test: {np.sum(labels[test_indexs])}/{len(test_indexs)}.')
                self.logger.info(f'shape of user-user graph: {np.shape(graph_u2u)}, with {np.sum(graph_u2u > 0)} nnz entries ({100*np.sum(graph_u2u > 0)/(np.shape(graph_u2u)[0]**2):.4f}%).')

                if self.args.learn_method == 'rand':
                    self.logger.info('using randomized features.')
                    feat_data = torch.rand(np.shape(feat_data))

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)
                setattr(self, dataSet+'_train_cls', train_indexs_cls)
                setattr(self, dataSet+'_trainable', train_indexs)
                setattr(self, dataSet+'_useridmax', user_id_max)

                setattr(self, dataSet+'_feats', feat_data)
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adj_lists)
                setattr(self, dataSet+'_best_adj_lists', best_adj_lists)
                setattr(self, dataSet+'_simis', graph_simi)
                if self.args.a_loss != 'none':
                    setattr(self, dataSet+'_clusters_a', clusters_a)
                    setattr(self, dataSet+'_labels_a', labels_a)
                    setattr(self, dataSet+'_u2size_of_cluster', u2size_of_cluster)

        elif dataSet.startswith('weibo'):
            if self.args.learn_method == 'gnn':
                ds = dataSet
                graph_u2p_file = self.file_paths[ds]['graph_u2p']
                labels_file = self.file_paths[ds]['labels']
                features_bow_file = self.file_paths[ds]['features_bow']
                features_loc_file = self.file_paths[ds]['features_loc']

                graph_u2p = pickle.load(open(graph_u2p_file, 'rb'))
                labels = pickle.load(open(labels_file, 'rb'))
                feat_bow = pickle.load(open(features_bow_file, 'rb'))
                feat_loc = pickle.load(open(features_loc_file, 'rb'))

                if self.args.feature == 'all':
                    feat_data = np.concatenate((feat_loc, feat_bow), axis=1)
                elif self.args.feature == 'bow':
                    feat_data = feat_bow
                elif self.args.feature == 'loc':
                    feat_data = feat_loc

                m, n = np.shape(graph_u2p)      #shape = (8405, 61964)
                adj_lists = {}
                feat_p = np.zeros((n, np.shape(feat_data)[1]))      #shape.(label) = (8405,) shape(bow) = (8405, 300), shape(loc) = (8405, 100)
                for i in range(m):
                    adj_lists[i] = set(m + graph_u2p[i,:].nonzero()[1])         #m+第i行中非零元素的纵索引列表,广播机制每个元素都加了m
                for j in range(n):
                    adj_lists[j+m] = set(graph_u2p[:,j].nonzero()[0])       #表示第j列中非零元素的横坐标索引值，     ？表明和j相连的元素序号？
                    feat_j = feat_data[graph_u2p[:,j].nonzero()[0]]     #输出对应的行向量               ？输出对应点的特征值
                    feat_j = np.mean(feat_j, 0)     #mean(a, axis = 0)在axis=0上求平均值，仅保留列数，即为4*3->1*3     ？与点j相连的所有点的各个特征值的平均值
                    feat_p[j] = feat_j

                feat_data = np.concatenate((feat_data, feat_p), 0)      #axis=0方向上拼接矩阵
                # feat_data = np.concatenate((feat_data, np.zeros((n, np.shape(feat_data)[1]))), 0)
                assert np.shape(feat_data)[0] == m+n        #相当于try

                assert len(feat_data) == len(labels)+n == len(adj_lists)
                ###分割训练集、测试集、验证集
                test_indexs, val_indexs, train_indexs = self._split_data(len(labels))
                user_id_max = len(labels)
                train_indexs_cls = train_indexs
                train_indexs = np.arange(len(labels))

                user_id_max = len(labels)
                graph_simi = np.ones((10, 10))

                # get labels for anomaly losses if needed
                if self.args.a_loss != 'none':
                    self.logger.info(f'using {self.args.a_loss} anomaly loss.')
                    label_a_file = self.file_paths[ds][self.args.a_loss]["a_label"]
                    labels_a = pickle.load(open(label_a_file, 'rb')).astype(int)
                    # get clusters for anomaly losses if needed
                    if self.args.cluster_aloss:
                        clusters_file = self.file_paths[ds][self.args.a_loss]["a_cluster"]
                        clusters = pickle.load(open(clusters_file, 'rb'))
                        u2cluster = defaultdict(list)
                        for i in range(len(clusters)):
                            for u in clusters[i]:
                                u2cluster[u].append(i)
                        cluster_neighbors = defaultdict(set)
                        for u in range(np.shape(graph_u2p)[0]):
                            for clus_i in u2cluster[u]:
                                cluster_neighbors[u] |= clusters[clus_i]
                        for u in range(np.shape(graph_u2p)[0]):
                            cluster_neighbors[u] = cluster_neighbors[u] - set([u])
                            assert len(cluster_neighbors[u]) >= 1
                        u2size_of_cluster = {}
                        for u, neighbors in cluster_neighbors.items():
                            u2size_of_cluster[u] = len(neighbors)
                    else:
                        u2size_of_cluster = {}
                        for u in range(np.shape(graph_u2p)[0]):
                            u2size_of_cluster[u] = np.sum(labels_a == labels_a[u])

                setattr(self, dataSet+'_test', test_indexs)     #设置类的属性
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)
                setattr(self, dataSet+'_train_cls', train_indexs_cls)
                setattr(self, dataSet+'_trainable', train_indexs)
                setattr(self, dataSet+'_useridmax', user_id_max)

                setattr(self, dataSet+'_feats', feat_data)
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adj_lists)
                setattr(self, dataSet+'_best_adj_lists', adj_lists)
                setattr(self, dataSet+'_simis', graph_simi)
                if self.args.a_loss != 'none':
                    setattr(self, dataSet+'_labels_a', labels_a)
                    setattr(self, dataSet+'_u2size_of_cluster', u2size_of_cluster)
                if self.args.cluster_aloss:
                    setattr(self, dataSet+'_cluster_neighbors', cluster_neighbors)
            ###实际执行的模块      args.learn_method = bigal, args_simi_func = cos
            else:
                ds = dataSet
                graph_u2p_file = self.file_paths[ds]['graph_u2p']
                graph_u2u_file = self.file_paths[ds]['graph_u2u']
                labels_file = self.file_paths[ds]['labels']
                features_bow_file = self.file_paths[ds]['features_bow']
                features_loc_file = self.file_paths[ds]['features_loc']

                if self.args.simi_func == 'pcc' or self.args.simi_func == 'rpcc':
                    _get_simi = getPCC
                    graph_simi_file = self.file_paths[ds]['graph_u2u_pcc']
                elif self.args.simi_func == 'acos':
                    _get_simi = getACOS
                    graph_simi_file = self.file_paths[ds]['graph_u2u_acos']
                elif self.args.simi_func == 'cos':
                    _get_simi = getCOS
                    graph_simi_file = self.file_paths[ds]['graph_u2u_cos']
                    '''
                    ###自己加上去的
                    graph_simi_data_content = str(pickle.load(open(graph_simi_file, 'rb')))
                    graph_simi_data_content_txt_path = graph_simi_file.split("/")[-1].split(".")[0]
                    print('path: ',graph_simi_data_content_txt_path)
                    text_creat(graph_simi_data_content_txt_path, graph_simi_data_content)
                    '''
                else:
                    self.logger.error('Invalid user-user similarity function.')


                graph_u2p = pickle.load(open(graph_u2p_file, 'rb'))
                graph_u2u = pickle.load(open(graph_u2u_file, 'rb'))
                '''
                graph_u2u_content_path_txt = graph_u2u_file.split("/")[-1].split(".")[0]
                graph_u2u_content = str(graph_u2u)
                text_creat(graph_u2u_content_path_txt, graph_u2u_content)
                '''
                labels = pickle.load(open(labels_file, 'rb'))
                feat_bow = pickle.load(open(features_bow_file, 'rb'))
                feat_loc = pickle.load(open(features_loc_file, 'rb'))

                ###默认为args.feature = 'all'
                if self.args.feature == 'all':
                    feat_data = np.concatenate((feat_loc, feat_bow), axis=1)        #feat_bow邻居节点属性

                elif self.args.feature == 'bow':
                    feat_data = feat_bow
                elif self.args.feature == 'loc':
                    feat_data = feat_loc
                elif self.args.feature == 'deepwalk':
                    embs_file = self.file_paths[ds]['deepwalk']
                    feat_data = pickle.load(open(embs_file, 'rb'))
                elif self.args.feature == 'bine':
                    embs_file = self.file_paths[ds]['bine']
                    feat_data = pickle.load(open(embs_file, 'rb'))
                elif self.args.feature == 'line':
                    embs_file = self.file_paths[ds]['line']
                    feat_data = pickle.load(open(embs_file, 'rb'))
                elif self.args.feature.startswith('n2v'):
                    n2v_index = self.args.feature.split('_')[-1]
                    embs_file = self.file_paths[ds]['n2v'][n2v_index]
                    feat_data = pickle.load(open(embs_file, 'rb'))
                else:
                    self.logger.error('Invalid features.')
                    sys.exit(1)

                # Compute or load the similarity graph when needed
                graph_simi = copy.deepcopy(graph_u2u)       #两个部分相互独立
                graph_simi[graph_simi > 0] = 1          #相似度为正的设置为1
                ###执行该if语句，default为args.learn_method = bigal
                if self.args.learn_method not in self.args.embedding_ready_methods:
                    self.logger.info(f'Using {self.args.simi_func} user-user similarity.')      #输出日志
                    if pathlib.Path(graph_simi_file).is_file():
                        graph_simi = pickle.load(open(graph_simi_file, 'rb'))
                        self.logger.info('Loaded user-user similarity graph.')

                    #暂时不予考虑……
                    else:
                        nz_entries = np.asarray(graph_u2u.nonzero()).T
                        self.logger.info(f'Calculating user-user {self.args.simi_func} similarity graph, {len(nz_entries)} edges to go...')
                        sz = 1000
                        n_batch = math.ceil(len(nz_entries) / sz)
                        batches = np.array_split(nz_entries, n_batch)
                        pool = Pool()
                        if self.args.simi_func == 'pcc' or self.args.simi_func == 'acos' or self.args.simi_func == 'cos':
                            # when calculating pcc or acos similarity, directly use the adjacency matrix
                            results = pool.map(get_simi_single_iter, [(entries_batch, graph_u2p, _get_simi) for entries_batch in batches])
                        elif self.args.simi_func == 'man' or self.args.simi_func == 'euc':
                            # when calculating manhattan or euclidean ditance, normalize the edge weights by power low distribution
                            _row, _col = graph_u2p.nonzero()
                            _data = np.asarray(graph_u2p[_row, _col]).squeeze()
                            pl_fit = powerlaw.Fit(_data,discrete=True,xmin=1)
                            alpha = pl_fit.power_law.alpha
                            self.logger.info(f'normalized the edge weight with log base {alpha:.2f}')
                            _data = np.log1p(_data) / np.log(alpha)
                            graph_u2p_normalized = csr_matrix((_data, (_row, _col)), shape=np.shape(graph_u2p))
                            results = pool.map(get_simi_single_iter, [(entries_batch, graph_u2p_normalized, _get_simi) for entries_batch in batches])
                        results = list(zip(*results))
                        row = np.concatenate(results[0])
                        col = np.concatenate(results[1])
                        dat = np.concatenate(results[2])
                        graph_simi = csr_matrix((dat, (row, col)), shape=np.shape(graph_u2u))
                        if np.max(graph_simi) > 1:
                            graph_simi = graph_simi / np.max(graph_simi)
                        pickle.dump(graph_simi, open(graph_simi_file, "wb"))
                        self.logger.info('Calculated user-user similarity and saved it for catch.')

                if self.args.simi_func == 'rpcc':
                    graph_simi[graph_simi<0] = 0

                # get the user-user adjacency list
                adj_lists = {}
                for i in range(np.shape(graph_u2u)[0]):
                    #按行获取每个点的邻接点，即为graph_u2u矩阵一行中的非零元素，对于点i的邻接点及其重要性存入adj-list[i]，其中adj_llist[i]的重要性中各个元素的值表示该节点对于节点i的重要性
                    adj_lists[i] = set(graph_u2u[i,:].nonzero()[1])
                    assert len(adj_lists[i]) > 0

                #得到前15大的adjst_list
                # best neighbors list
                best_adj_lists = {}
                for i in range(np.shape(graph_u2u)[0]):
                    if len(adj_lists[i]) <= 15:
                        best_adj_lists[i] = adj_lists[i]
                    else:
                        adjs = graph_u2u[i].toarray()[0]        #将csr_matrix转化为array
                        best_adj_lists[i] = set(np.argpartition(adjs, -15)[-15:])       #得到前15大的adj_list，相当于S=15

                #default情况下，args.a_loss默认为none
                # get labels for anomaly losses if needed
                if self.args.a_loss != 'none':
                    self.logger.info(f'using {self.args.a_loss} anomaly loss.')
                    label_a_file = self.file_paths[ds][self.args.a_loss]["a_label"]         ##ds = Weibo's      #labels_a: <type numpy.ndarray>, shpe = 8405
                    labels_a = pickle.load(open(label_a_file, 'rb')).astype(int)
                    # get clusters for anomaly losses if needed
                    if self.args.cluster_aloss:
                        clusters_file = self.file_paths[ds][self.args.a_loss]["a_cluster"]
                        clusters = pickle.load(open(clusters_file, 'rb'))
                        if self.args.noblock:
                            c = [set.union(*clusters[:-1]), clusters[-1]]           #clusters: <type list>, len(clusters) = 171, cluster是list of set
                                                                                    ###???????????????????????????????????????????#########带星号表示*clusters[:-1])是分立的170个set，
                                                                                    ###？？？？？？？？？？？？？？？？？？？？？？?#########不带星号表示clusters[:-1])是整个list
                            clusters = c
                        u2cluster = defaultdict(list)       #使用'defaultdict类'的方法，创建u2cluster字典并为<type list>类型赋予默认值，即为default{<type list>,{u1:i1, u2:i2},<type set>,{u1: cluster_neighbors1}}
                        for i in range(len(clusters)):
                            for u in clusters[i]:
                                u2cluster[u].append(i)              #第i个点的cluster_neighbor
                        cluster_neighbors = defaultdict(set)
                        for u in range(np.shape(graph_u2p)[0]):
                            for clus_i in u2cluster[u]:
                                cluster_neighbors[u] |= clusters[clus_i]        #更新集合为二者并集          cluter_neighbor[u]表示节点u和他的cluster_neighbors，有neighbors的在u2cluster[u]不为空
                        for u in range(np.shape(graph_u2p)[0]):
                            cluster_neighbors[u] = cluster_neighbors[u] - set([u])      #除去自身
                            assert len(cluster_neighbors[u]) >= 1       #cluster_neighbors中每个点都有不止一个neighbor
                        u2size_of_cluster = {}
                        for u, neighbors in cluster_neighbors.items():
                            u2size_of_cluster[u] = len(neighbors)       ##每个cluster_neighbor的个数
                    else:
                        u2size_of_cluster = {}
                        for u in range(np.shape(graph_u2p)[0]):
                            u2size_of_cluster[u] = np.sum(labels_a == labels_a[u])

                assert len(feat_data) == len(labels) == len(adj_lists)      #每个节点都有对应的属性和邻接点
                test_indexs, val_indexs, train_indexs = self._split_data(len(labels))       #_split_data按照1/3，1/6的比例分别划分出训练集，测试集，验证集
                user_id_max = len(labels)
                train_indexs_cls = train_indexs     #训练样本
                train_indexs = np.arange(len(labels))

                user_id_max = len(labels)
                self.logger.info(f'distr: train {np.sum(labels[train_indexs_cls])}/{len(train_indexs_cls)}, vali: {np.sum(labels[val_indexs])}/{len(val_indexs)}, test: {np.sum(labels[test_indexs])}/{len(test_indexs)}.')
                self.logger.info(f'shape of user-user graph: {np.shape(graph_u2u)}, with {np.sum(graph_u2u > 0)} nnz entries ({100*np.sum(graph_u2u > 0)/(np.shape(graph_u2u)[0]**2):.4f}%).')

                #默认情况learn_method为bigal
                if self.args.learn_method == 'rand':
                    self.logger.info('using randomized features.')
                    feat_data = torch.rand(np.shape(feat_data))

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)
                setattr(self, dataSet+'_train_cls', train_indexs_cls)
                setattr(self, dataSet+'_trainable', train_indexs)
                setattr(self, dataSet+'_useridmax', user_id_max)

                setattr(self, dataSet+'_feats', feat_data)
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adj_lists)          #对应Class GNN中self.adj_lists
                setattr(self, dataSet+'_best_adj_lists', best_adj_lists)
                setattr(self, dataSet+'_simis', graph_simi)
                if self.args.a_loss != 'none':
                    setattr(self, dataSet+'_labels_a', labels_a)
                    setattr(self, dataSet+'_u2size_of_cluster', u2size_of_cluster)
                #default为args.cluster_aloss = True,?????此处属性不存在应该？？？？
                if self.args.cluster_aloss:
                    setattr(self, dataSet+'_cluster_neighbors', cluster_neighbors)


    def _split_data(self, num_nodes, test_split = 3, val_split = 6):
        rand_indices = np.random.permutation(num_nodes)

        test_size = num_nodes // test_split
        val_size = num_nodes // val_split
        train_size = num_nodes - (test_size + val_size)

        test_indexs = rand_indices[:test_size]
        val_indexs = rand_indices[test_size:(test_size+val_size)]
        train_indexs = rand_indices[(test_size+val_size):]

        return test_indexs, val_indexs, train_indexs

    def _split_data_limited(self, nodes, test_split = 4, val_split = 4):
        np.random.shuffle(nodes)

        test_size = len(nodes) // test_split
        val_size = len(nodes) // val_split
        train_size = len(nodes) - (test_size + val_size)

        val_indexs = nodes[:test_size]
        test_indexs_init = nodes[test_size:(test_size+val_size)]
        train_indexs_init = nodes[(test_size+val_size):]
        data_splits = []

        for _ in range(5):
            np.random.shuffle(test_indexs_init)
            np.random.shuffle(train_indexs_init)
            test_split = np.split(test_indexs_init, [test_size-10, test_size])
            train_split = np.split(train_indexs_init, [train_size-10, train_size])
            test_indexes = np.concatenate((test_split[0], train_split[1]))
            train_indexes = np.concatenate((train_split[0], test_split[1]))
            assert len(test_indexes) == test_size
            assert len(train_indexes) == train_size
            data_splits.append((test_indexes, val_indexs, train_indexes))

        return data_splits


Dc = DataCenter(args)
print('next')
print('Dc:', Dc)
adj_lists = getattr(Dc, Dc.args.dataSet+'_adj_lists')       #继承Dc中的
print('adj_lists: ',adj_lists)