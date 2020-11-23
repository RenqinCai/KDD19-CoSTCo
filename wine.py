import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import argparse
import copy
from collections import Counter 
from nltk.tokenize import TweetTokenizer
import gensim
import random

class WINE(Dataset):
    def __init__(self, args, vocab_obj, df):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_max_seq_len = args.max_seq_length
        self.m_batch_size = args.batch_size
    
        self.m_sos_id = vocab_obj.sos_idx
        self.m_eos_id = vocab_obj.eos_idx
        self.m_pad_id = vocab_obj.pad_idx
        self.m_vocab_size = vocab_obj.vocab_size
        self.m_vocab = vocab_obj
    
        self.m_sample_num = len(df)
        print("sample num", self.m_sample_num)

        # self.m_batch_num = int(self.m_sample_num/self.m_batch_size)
        
        # if (self.m_sample_num/self.m_batch_size - self.m_batch_num) > 0:
        #     self.m_batch_num += 1

        # print("batch num", self.m_batch_num)

        ###get length
    
        self.m_item_batch_list = []
        self.m_user_batch_list = []
        
        # self.m_pos_target_list = []
        # self.m_pos_len_list = []

        # self.m_neg_target_list = []
        # self.m_neg_len_list = []

        self.m_attr_list = []
        self.m_target_list = []

        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        # review_list = df.review.tolist()
        # tokens_list = df.token_idxs.tolist()
        pos_attr_list = df.pos_attr.tolist()

        whole_attr_list = [i for i in range(self.m_vocab_size)]
        neg_sample_num = 300

        for sample_index in range(self.m_sample_num):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]

            pos_attrlist_i = list(pos_attr_list[sample_index])
            pos_attrlist_i = [int(j) for j in pos_attrlist_i] 
            
            full_neg_attrlist_i = set(pos_attrlist_i)^set(whole_attr_list)
            full_neg_attrlist_i = list(full_neg_attrlist_i)
            random.shuffle(full_neg_attrlist_i)
            neg_attrlist_i = full_neg_attrlist_i[:neg_sample_num]
            neg_attrlist_i = [int(j) for j in neg_attrlist_i]

            for j in range(len(pos_attrlist_i)):
                self.m_user_batch_list.append(user_id)
                self.m_item_batch_list.append(item_id)

                self.m_attr_list.append(pos_attrlist_i[j])
                self.m_target_list.append([1])

            for j in range(len(neg_attrlist_i)):
                self.m_user_batch_list.append(user_id)
                self.m_item_batch_list.append(item_id)
                self.m_attr_list.append(neg_attrlist_i[j])
                self.m_target_list.append([0])

        print("... load train data ...", len(self.m_item_batch_list))

        self.m_batch_num = len(self.m_item_batch_list)/self.m_batch_size
        print("batch num", self.m_batch_num)
        # exit()

    def __len__(self):
        return len(self.m_item_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        
        item_i = self.m_item_batch_list[i]

        user_i = self.m_user_batch_list[i]

        attr_i = self.m_attr_list[i]

        target_i = self.m_target_list[i]

        sample_i = { "item": item_i,  "user": user_i,  "attr": attr_i, "target": target_i}

        return sample_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        user_iter = []

        item_iter = []

        attr_iter = []

        target_iter = []

        for i in range(batch_size):
            sample_i = batch[i]

            user_i = sample_i["user"]
            user_iter.append(user_i)

            item_i = sample_i["item"]
            item_iter.append(item_i)
            
            attr_i = copy.deepcopy(sample_i["attr"])

            target_i = copy.deepcopy(sample_i["target"])
        
            attr_iter.append(attr_i)
            target_iter.append(target_i)

        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()
    
        attr_iter_tensor = torch.from_numpy(np.array(attr_iter)).long()
        target_iter_tensor = torch.from_numpy(np.array(target_iter)).long()
       
        return user_iter_tensor, item_iter_tensor, attr_iter_tensor, target_iter_tensor

class WINE_TEST(Dataset):
    def __init__(self, args, vocab_obj, df):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_max_seq_len = args.max_seq_length
        self.m_batch_size = args.batch_size
    
        self.m_sos_id = vocab_obj.sos_idx
        self.m_eos_id = vocab_obj.eos_idx
        self.m_pad_id = vocab_obj.pad_idx
        self.m_vocab_size = vocab_obj.vocab_size
        self.m_vocab = vocab_obj
    
        self.m_sample_num = len(df)
        print("sample num", self.m_sample_num)

        # self.m_batch_num = int(self.m_sample_num/self.m_batch_size)
        # print("batch num", self.m_batch_num)

        # if (self.m_sample_num/self.m_batch_size - self.m_batch_num) > 0:
        #     self.m_batch_num += 1

        ###get length
    
        self.m_item_batch_list = []
        self.m_user_batch_list = []
        
        self.m_target_list = []
        self.m_target_len_list = []

        userid_list = df.userid.tolist()
        itemid_list = df.itemid.tolist()
        attr_list = df.attr.tolist()
        
        for sample_index in range(self.m_sample_num):
            user_id = userid_list[sample_index]
            item_id = itemid_list[sample_index]
            attrlist_i = list(attr_list[sample_index])
            attrlist_i = [int(j) for j in attrlist_i]

            self.m_item_batch_list.append(item_id)
        
            self.m_user_batch_list.append(user_id)

            self.m_target_list.append(attrlist_i)
            self.m_target_len_list.append(len(attrlist_i))

        print("... load train data ...", len(self.m_item_batch_list))

        self.m_batch_num = len(self.m_item_batch_list)/self.m_batch_size
        print("batch num", self.m_batch_num)
        

    def __len__(self):
        return len(self.m_item_batch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        
        item_i = self.m_item_batch_list[i]
        user_i = self.m_user_batch_list[i]

        target_i = self.m_target_list[i]
        target_len_i = self.m_target_len_list[i]

        sample_i = {"item": item_i, "user": user_i, "target": target_i, "target_len": target_len_i, "vocab_size":self.m_vocab_size}

        return sample_i
    
    @staticmethod
    def collate(batch):
        batch_size = len(batch)

        item_iter = []
        user_iter = []

        attr_iter = []

        target_iter = []
        target_len_iter = []
        target_mask_iter = []

        for i in range(batch_size):
            sample_i = batch[i]
            
            target_len_i = sample_i["target_len"]
            target_len_iter.append(target_len_i)

        max_targetlen_iter = max(target_len_iter)

        # freq_pad_id = float('-inf')
        freq_pad_id = float(0)
        pad_id = 0

        for i in range(batch_size):
            sample_i = batch[i]

            item_i = sample_i["item"]
            item_iter.append(item_i)

            user_i = sample_i["user"]
            user_iter.append(user_i)

            vocab_size = sample_i["vocab_size"]
            attr_i = [j for j in range(vocab_size)]
            attr_iter.append(attr_i)

            target_i = copy.deepcopy(sample_i["target"])
            target_len_i = sample_i["target_len"]
            target_i.extend([0]*(max_targetlen_iter-target_len_i))
            target_iter.append(target_i)

            target_mask_iter.append([1]*target_len_i+[0]*(max_targetlen_iter-target_len_i))
            
        user_iter_tensor = torch.from_numpy(np.array(user_iter)).long()
        item_iter_tensor = torch.from_numpy(np.array(item_iter)).long()
        attr_iter_tensor = torch.from_numpy(np.array(attr_iter)).long()

        target_iter_tensor = torch.from_numpy(np.array(target_iter)).long()
        target_mask_iter_tensor = torch.from_numpy(np.array(target_mask_iter)).long()

        return  user_iter_tensor, item_iter_tensor, attr_iter_tensor, target_iter_tensor, target_mask_iter_tensor
