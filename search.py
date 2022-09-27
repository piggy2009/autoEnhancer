import os
import sys
import time
import glob
import numpy as np
import pickle
import torch
import logging
import argparse
import torch
import random


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = False

from underwater_model.model_SPOS import Water

from tester_water import get_cand_err
# from flops import get_cand_flops
from config import saving_path
import sys
sys.setrecursionlimit(10000)
import argparse

import functools
print = functools.partial(print, flush=True)

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))

device_id = 0
torch.cuda.set_device(device_id)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = saving_path

exp_name = 'WaterEnhance_2022-06-24 10:46:22'

args = {
    'snapshot': '200000',  # your snapshot filename (exclude extension name)
    'choice': 9,
    'layers': 12,
    'en_channels': [64, 128, 256],
    'dim': 48,
    'log_dir': 'log',
    'max_epochs': 40,
    'select_num': 10,
    'population_num': 40,
    'top_k': 20,
    'm_prob': 0.1,
    'crossover_num': 40,
    'mutation_num': 40,
    'flops_limit': 330 * 1e6,
    'max_train_iters': 20,
    'train_batch_size': 5,
    'image_size': 430,
    'crop_size': 380,
    # 'image_path': '/mnt/hdd/data/ty2/input_test',
    # 'depth_path': '/mnt/hdd/data/ty2/depth_test',
    # 'gt_path': '/mnt/hdd/data/ty2/gt_test',
    'image_path': 'dataset image path',
    'gt_path': 'dataset gt path',
    'dataset': 'dataset name',
}


class EvolutionSearcher(object):

    def __init__(self):
        self.args = args
        # print(args['flops-limit'])

        self.max_epochs = args['max_epochs']
        self.select_num = args['select_num']
        self.top_k = args['top_k']
        self.population_num = args['population_num']
        self.m_prob = args['m_prob']
        self.crossover_num = args['crossover_num']
        self.mutation_num = args['mutation_num']
        self.flops_limit = args['flops_limit']

        self.model = Water(dim=args['dim'])
        # self.model = torch.nn.DataParallel(self.model).cuda()
        # supernet_state_dict = torch.load(
        #     '../Supernet/models/checkpoint-latest.pth.tar')['state_dict']
        self.model.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),
                                   map_location='cuda:' + str(device_id)))

        self.log_dir = args['log_dir']
        self.checkpoint_name = os.path.join(self.log_dir, 'checkpoint.pth.tar')

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], self.top_k: []}
        self.epoch = 0
        self.candidates = []

        self.nr_layer = args['layers']
        self.nr_state = args['choice']

    def save_checkpoint(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        info = {}
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        torch.save(info, self.checkpoint_name)
        print('save checkpoint to', self.checkpoint_name)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info = torch.load(self.checkpoint_name)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_name)
        # print('top k:', info.keys())
        # print('infor message:', info)
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple) and len(cand) == self.nr_layer
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False

        # if 'flops' not in info:
        #     info['flops'] = get_cand_flops(cand)

        print(cand)

        # if info['flops'] > self.flops_limit:
        #     print('flops limit exceed')
        #     return False

        info['err'] = get_cand_err(self.model, cand, self.args)

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(
            lambda: tuple(np.random.randint(self.nr_state) for i in range(self.nr_layer)))
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            for i in range(self.nr_layer):
                if np.random.random_sample() < m_prob:
                    cand[i] = np.random.randint(self.nr_state)
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            return tuple(choice([i, j]) for i, j in zip(p1, p2))
        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        print('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['err'], reverse=True)
            self.update_top_k(
                self.candidates, k=self.top_k, key=lambda x: self.vis_dict[x]['err'], reverse=True)

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[self.top_k])))
            for i, cand in enumerate(self.keep_top_k[self.top_k]):
                print('No.{} {} Top-1 err = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['err']))
                ops = [i for i in cand]
                print('ops:', ops)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

        self.save_checkpoint()


def main():
    # print(args['max-epochs'])
    t = time.time()

    searcher = EvolutionSearcher()

    searcher.search()

    print('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))

if __name__ == '__main__':
    main()
