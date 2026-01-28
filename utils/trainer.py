import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from utils.grad import get_pcgrad, get_gcsgrad
from utils.data import MALDataset, TRAIN_DATA, TEST_DATA
from utils.utils import log_loss, compute_auc_gauc, print_config


class Trainer:
    def __init__(self, args, model, optimizer, lr_scheduler, writer):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = writer    

        self.steps = 0
        self.all_p = list(self.model.parameters())

    def run(self):
        for i in range(len(TRAIN_DATA)):
            self.train(i)
        self.test()

    def train(self, idx):
        self.model.train()
        data_name = (TRAIN_DATA[idx]).replace('.parquet', '.pkl')
        data_path = os.path.join(self.args.path, 'data', 'cached', data_name)
        with open(data_path, 'rb') as f:
            data_train = pickle.load(f)
        dataset_train = MALDataset(data_train)
        dataloader_train = DataLoader(dataset_train, batch_size=self.args.batch_size, num_workers=8, shuffle=False)
        for batch in dataloader_train:
            user_indices, ad_indices, buy_mm_scores, buy_seq_indices = [batch[x].to('cuda') for x in ['user', 'ad', 'mm', 'seq']]
            labels = {k: v.to('cuda', non_blocking=True) for k, v in batch['label'].items()}
            probs = self.model(user_indices[:, 1:], ad_indices, buy_mm_scores, buy_seq_indices)
            main_loss, aux_loss_list = log_loss(probs, labels, self.args.main_view, self.args.aux_views)
            total_loss = main_loss + sum([self.args.lamda * x for x in aux_loss_list])
            self.optimizer.zero_grad()

            if self.args.gcsgrad:
                gcsgrad = get_gcsgrad(main_loss, aux_loss_list, self.all_p)
                for p, g in zip(self.all_p, gcsgrad):
                    p.grad = g
            elif self.args.pcgrad:
                pcgrad = get_pcgrad(main_loss, aux_loss_list, self.all_p)
                for p, g in zip(self.all_p, pcgrad):
                    p.grad = g
            else:
                total_loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            self.writer.add_scalar('Training/Loss', main_loss.item(), self.steps)
            self.steps += 1    

    def test(self):
        self.model.eval()
        cur_views = list(set([self.args.main_view] + self.args.aux_views) - {'cartesian'})
        all_labels,  all_probs, all_user = {view:[] for view in cur_views}, {view:[] for view in cur_views}, []
        with torch.no_grad():
            for data_name in TEST_DATA:
                data_name = data_name.replace('.parquet', '.pkl')
                data_path = os.path.join(self.args.path, 'data', 'cached', data_name)
                with open(data_path, 'rb') as f:
                    data_test = pickle.load(f)
                dataset_test = MALDataset(data_test)
                dataloader_test = DataLoader(dataset_test, batch_size=self.args.batch_size, num_workers=8, shuffle=False)
                for batch in dataloader_test:
                    user_indices, ad_indices, buy_mm_scores, buy_seq_indices = [batch[x].to('cuda') for x in ['user', 'ad', 'mm', 'seq']]
                    labels = {k: v.to('cuda', non_blocking=True) for k, v in batch['label'].items()}
                    probs = self.model(user_indices[:, 1:], ad_indices, buy_mm_scores, buy_seq_indices)

                    all_user.append(user_indices[:, 0].cpu().numpy())
                    for view in cur_views:
                        all_labels[view].append(labels[view].cpu().numpy())
                        all_probs[view].append(probs[view][:, 1].cpu().numpy())

        all_user = np.concatenate(all_user)
        for view in cur_views:
            all_probs[view] = np.concatenate(all_probs[view])
            all_labels[view] = np.concatenate(all_labels[view])
        print_config(self.args)
        for view in cur_views:
            auc, gauc, valid_user_cnt = compute_auc_gauc(all_user, all_labels[view], all_probs[view])
            print(f'Testing/View: {view:<8}, Testing/AUC: {auc:.5f}, Testing/GAUC: {gauc:.5f}, Valid Users: {valid_user_cnt}')
