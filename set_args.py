#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/8 21:30
# @Author  : zhixiuma
# @File    : set_args.py
# @Project : FedE-master
# @Software: PyCharm
import argparse

parser = argparse.ArgumentParser()

# parser.add_argument('--state_dir', '-state_dir', default='./state_clean', type=str,
#                     help='directory for saving model state dict')
# parser.add_argument('--log_dir', '-log_dir', default='./log_clean', type=str, help='directory for saving log')
# parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log_clean', type=str,
#                     help='directory for saving tensorboard log')
parser.add_argument('--state_dir', '-state_dir', default='state_attack_cpa', type=str,
                    help='directory for saving model state dict')
parser.add_argument('--log_dir', '-log_dir', default='log_attack_cpa', type=str, help='directory for saving log')
parser.add_argument('--tb_log_dir', '-tb_log_dir', default='tb_log_attack_cpa', type=str,
                    help='directory for saving tensorboard log')

# parser.add_argument('--setting', default='DPA_S', choices=['FedE',
#                                                           'DPA_S',
#                                                           'FMPA_S',
#                                                           'CPA',
#                                                           'FedE_CPA',
#                                                           ],
#                     help='setting for current experiment')
parser.add_argument('--setting', default='DPA_S_gradient_attack', choices=['FedE',
                                                          'DPA_S',
                                                          'DPA_S_gradient_attack',
                                                          'FMPA_S',
                                                          'CPA',
                                                          'FedE_CPA',
                                                          ],
                    help='setting for current experiment')

parser.add_argument('--mode', default='train', choices=['train', 'test'], help='model training or testing')
parser.add_argument('--one_client_idx', default=0, type=int, help='the client index on Isolation or Collection setting')
parser.add_argument('--max_epoch', default=10000, type=int,
                    help='the max training epoch on Isolation or Collection setting')
parser.add_argument('--log_per_epoch', default=1, type=int,
                    help='take log per epoch on Isolation or Collection setting')
parser.add_argument('--check_per_epoch', default=10, type=int,
                    help='do validation per epoch on Isolation or Collection setting')
parser.add_argument('--isolation_name_list', default=None, type=list,
                    help='list with names for experiments on isolation training of a dataset')

parser.add_argument('--batch_size', default=512, type=int,
                    help='batch size for training KGE on FedE, Isolation or Collection,')
parser.add_argument('--test_batch_size', default=512, type=int,
                    help='batch size for training KGE on FedE, Isolation or Collection,')
parser.add_argument('--num_neg', default=256, type=int,
                    help='number of negative sample for training KGE on FedE, Isolation or Collection,')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate for training KGE on FedE, Isolation or Collection,')

# hyper parameter for FedE
# parser.add_argument('--max_round', default=10000, type=int, help='the max training round on FedE')
parser.add_argument('--max_round', default=200, type=int, help='the max training round on FedE')
# parser.add_argument('--local_epoch', default=3, help='number of local training epochs on FedE')
parser.add_argument('--local_epoch', default=1, help='number of local training epochs on FedE')
parser.add_argument('--fraction', default=1, type=float, help='client selection fraction each round on FedE setting')
parser.add_argument('--log_per_round', default=1, type=int, help='take log per epoch on FedE setting')
parser.add_argument('--check_per_round', default=5, type=int, help='do validation per epoch on FedE setting')

parser.add_argument('--early_stop_patience', default=5, type=int, help='early stop patience for training')
parser.add_argument('--gamma', default=10.0, type=float, help='gamma in self-adversarial loss')
parser.add_argument('--epsilon', default=2.0, type=float)
parser.add_argument('--hidden_dim', default=128, type=int)
# parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--gpu', default='cpu', type=str)
parser.add_argument('--num_cpu', default=10, type=int)
parser.add_argument('--adversarial_temperature', default=1.0, type=float)
parser.add_argument('--seed', default=12345, type=int)
parser.add_argument('--num_client', default=3, type=int, help='no need to specifiy')
parser.add_argument('--save_client_path', default='./client_data/', type=str, help='client dataset save path')
parser.add_argument('--dataset_name', default='FB15k237', help='dataset', 
                    choices = [ 'NELL995', 'WNRR', 'FB15k237', 'CoDEx-M' ])
# parser.add_argument('--num_train', default=100, help='the number of train triples in client')
# parser.add_argument('--num_test', default=10, help='the number of test triples in client')
# parser.add_argument('--num_val', default=10, help='the number of valid triples in client')
parser.add_argument('--server_model', default='TransE', choices=['TransE', 'RotatE', 'DistMult', 'ComplEx'],
                    help='specific KGE method for training KGE')
parser.add_argument('--client_model', default='TransE', choices=['TransE', 'RotatE', 'DistMult', 'ComplEx'],
                    help='specific KGE method for training KGE')
parser.add_argument('--attack_entity_ratio', default=10, help='attack entity ratio')
parser.add_argument('--poisoned_triples_path', default='./poisoned_triples_path/', help='poisoned triples path')
parser.add_argument('--victim_client', default=1, help='poisoned triples path')
parser.add_argument('--poisoning_attack_type', default='random', choices=['random', 'inference_based'], help='poisoning_attack_type')


args = parser.parse_args()