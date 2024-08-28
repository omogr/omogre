
import torch
import collections
import random

pad_token_id = 0
bos_token_id = 1
eos_token_id = 2
alp_token_id = 3
alphabet = ' абвгдеёжзийклмнопрстуфхцчшщъыьэюя' #АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'

TokenSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "TokenSpan", ["text", "label", "punct", "span"])

def get_tokens(sentence):
    tokens = [bos_token_id]

    err_cnt = 0
    for tc in sentence: 
        if tc == '+':
            continue
        else:
            tid = alphabet.find(tc)
            if tid < 0:
                err_cnt += 1
                tid = pad_token_id
            else:
                tid += alp_token_id
            tokens.append(tid)
    tokens.append(eos_token_id)
    return tokens, err_cnt
    
    
class InfBatchFromSentenceList:
    def __init__(self, sentence_list): # ='text/forms_g2p_prep.txt'
        self.all_entries = []

        for line_indx, sentence in enumerate(sentence_list):
            tokens, err_cnt = get_tokens(sentence.casefold())
            if err_cnt == 0:                
                ct = (tokens, sentence)
                self.all_entries.append(ct)
                
        self.file_pos = -1        
        self.iter = 0
        assert len(self.all_entries) > 0

    def is_first_iter(self):
        if self.iter > 0:
            return False
        if (self.file_pos + 1) >= len(self.all_entries):
            return False
        return True
        
    def get_next_pos(self):
        self.file_pos += 1
        if self.file_pos >= len(self.all_entries):
            self.iter += 1            
            self.file_pos = 0
            
    def get_next_batch(self, is_test=True):
        if is_test:
            assert self.is_first_iter()

        num_tokens = 250 * 24
        max_length = 1
        batch_data = []
        sentence_data = []
        while True:
            self.get_next_pos()
            if is_test:
                if self.iter > 0:
                    break
            sentence_pos = self.file_pos
            input_ids, sentence = self.all_entries[sentence_pos]
            len_input_ids = len(input_ids)
            max_length = max(max_length, len_input_ids)
            
            token_cnt = max_length * (1 + len(sentence_data))
            if token_cnt > num_tokens:
                break
            batch_data.append((input_ids))
            sentence_data.append(sentence)
        
        if len(batch_data) < 1:
            return None
            
        all_input_ids = []
        all_attention_mask = []
                   
        for input_ids in batch_data:
            len_input_ids = len(input_ids)
            attention_mask = [1] * len_input_ids
            assert len(input_ids) <= max_length
            if len_input_ids < max_length:
                # Pad input_ids and attention_mask to max length
                padding_length = max_length - len_input_ids
                input_ids += [pad_token_id] * padding_length
                attention_mask += [0] * padding_length

            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
    
        all_input_ids = torch.stack(all_input_ids)
        all_attention_mask = torch.stack(all_attention_mask)
        batch = (all_input_ids, all_attention_mask, sentence_data)
        return batch


if __name__ == '__main__':
    fr = InfBatchFromSentenceList(['квазисублимирующие'])
    
    x = fr.get_next_batch()
    print('get_next_batch', x)
  