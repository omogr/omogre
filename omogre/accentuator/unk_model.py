# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import sys
import numpy as np
import torch

from .bert import BertForTokenClassification

from .unk_reader import InfBatchFromSentenceList
from .unk_vocab import UnkVocab

 
def norm_word(tword):
    return tword.casefold().replace('ё', 'е').replace(' ', '!')


def check_ee(tword):
    acc_pos = tword.find('ё')
    if acc_pos >= 0:
        return acc_pos

    acc_pos = tword.find('Ё')
    if acc_pos >= 0:
        return acc_pos
    return -1

    
class UnkModel:
    def __init__(self, data_path, device_name=None):
        model_data_path = os.path.join(data_path, 'unk_model')
        
        if device_name is None:           
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)
            
        self.model = BertForTokenClassification.from_pretrained(model_data_path, num_labels=1, cache_dir=None)
        assert self.model
        self.unk_vocab = UnkVocab(data_path)
        self.model.eval()
        self.model.to(self.device)
        self.error_counter = 0

    def seed(self, seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)

    def process_model_batch(self, input_ids, attention_mask, batch_text, all_res):
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            logits = self.model(input_ids, attention_mask=attention_mask)
            logits = logits.squeeze(-1)
            logits = logits.detach().cpu().tolist() # cpu().
            input_ids = input_ids.detach().cpu().tolist() # cpu().

            for batch_indx, (t_logits, t_input_ids) in enumerate(zip(logits, input_ids)):
                word_text = batch_text[batch_indx]
                if len(word_text) < 2:
                    all_res.append(word_text)
                    continue

                max_pos = 1
                max_logit = t_logits[max_pos]
                for token_indx in range(len(word_text)):
                    
                    if t_logits[token_indx+1] > max_logit:
                        max_pos = token_indx
                        max_logit = t_logits[token_indx+1]
                ct = (word_text, max_pos)
                all_res.append(ct)
    
    def get_acc_pos(self, unk_word):
        acc_pos = self.unk_vocab.get_acc_pos(unk_word)
        if acc_pos >= 0:
            return acc_pos
        
        file_reader = InfBatchFromSentenceList([unk_word])
        
        batch = file_reader.get_next_batch()
        if batch is None:
            return -1
        (all_input_ids, all_attention_mask, sentence_data) = batch
        sum_batch = []
        self.process_model_batch(all_input_ids, all_attention_mask, sentence_data, sum_batch)
        if len(sum_batch) != 1:
            return -1
        (word_text, max_pos) = sum_batch[0]
        if word_text != unk_word:
            return -1
        return max_pos

