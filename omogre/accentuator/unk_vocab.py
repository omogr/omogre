
import bisect
import pickle
import os

def common_left(str1: str, str2: str) -> int:
    if len(str1) < len(str2):
        for index, tc in enumerate(str1):
           if str2[index] != tc:
               return index
        return len(str1)

    for index, tc in enumerate(str2):
       if str1[index] != tc:
           return index
    return len(str2)    


def get_neib_pos(key_pos_list: list, key: str, pos: int, min_len: int = 3) -> list:    
    result = [pos]
    len1 = 0
    len2 = 0
    len3 = common_left(key, key_pos_list[pos][0])
    max_len = len3
    
    tp1 = pos - 1
    if tp1 >= 0:
        len1 = common_left(key, key_pos_list[tp1][0])
        max_len = max(max_len, len1)
        result.append(tp1)
        
    tp2 = pos + 1
    if tp2 < len(key_pos_list):
        len2 = common_left(key, key_pos_list[tp2][0])
        max_len = max(max_len, len2)
        result.append(tp2)

    if max_len < min_len:
        return result
        
    if len1 >= max_len:
        result.append(tp1)
        tp = pos - 2
        while tp >= 0:
            len1 = common_left(key, key_pos_list[tp][0])
            if len1 < max_len:
                break
            result.append(tp)
            tp -= 1
                
    if len2 >= max_len:
        result.append(tp2)
        tp = pos + 2
        while tp < len(key_pos_list):            
            len1 = common_left(key, key_pos_list[tp][0])
            if len1 < max_len:
                break
            result.append(tp)
            tp += 1            
    return result        


class UnkVocab:
    def __init__(self, data_path: str, encoding: str = 'utf-8'):
        unk_file = os.path.join(data_path, 'unk_vocab.pickle')

        with open(unk_file, 'rb') as finp:
            self.acc_vocab, self.all_tails = pickle.load(finp)

    def cmp_form_norm(self, form: str, tpos: int, res: list):       
        norm = self.acc_vocab[tpos][0]
        num_equ_chars: int = common_left(form, norm)
        key = (form[num_equ_chars:], norm[num_equ_chars:])

        if key in self.all_tails:
            res.append( (num_equ_chars, self.acc_vocab[tpos]) )
    
    def search_neibs(self, text: str) -> list:
        key = (text, '')

        all_len = len(self.acc_vocab)
        if all_len < 1:
            return []
        if key < self.acc_vocab[0]:
            return get_neib_pos(self.acc_vocab, text, 0)
            
        try_pos = bisect.bisect_left(self.acc_vocab, key)
            
        return get_neib_pos(self.acc_vocab, text, try_pos)

    def get_neibs(self, text: str) -> list:
        res = []
        for tpos in self.search_neibs(text):
            self.cmp_form_norm(text, tpos, res)
        return res
    
    def get_acc_pos(self, text: str) -> int:
        result = self.get_neibs(text)
        if len(result) < 1:
            return -1
        result.sort()
        cl, key_acc = result[-1]
        acc_pos_list = [int(tp) for tp in key_acc[1].split(',')]
        if len(acc_pos_list) < 1:
            return -1
        acc_pos = acc_pos_list[-1]
        if acc_pos < 0:
            return -1
        if acc_pos >= cl:
            return -1
        return acc_pos

