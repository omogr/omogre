
import bisect
import pickle
import os

def common_left(str1, str2):
    if len(str1) < len(str2):
        for index, tc in enumerate(str1):
           if str2[index] != tc:
               return index
        return len(str1)

    for index, tc in enumerate(str2):
       if str1[index] != tc:
           return index
    return len(str2)    


def get_neib_pos(key_pos_list, key, pos, min_len=3):    
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
    def __init__(self, data_path, encoding='utf-8'):
        unk_file = os.path.join(data_path, 'unk_vocab.pickle')

        with open(unk_file, 'rb') as finp:
            self.search_key, self.all_tails = pickle.load(finp)

    def cmp_form_norm(self, form, tp, res):       
        norm = self.search_key[tp][0]
        cl = common_left(form, norm)
        key = (form[cl:], norm[cl:])

        if key in self.all_tails:
            res.append( (cl, self.search_key[tp]) )
    
    def search_neibs(self, text):
        key = (text, '')

        all_len = len(self.search_key)
        if all_len < 1:
            return []
        if key < self.search_key[0]:
            return get_neib_pos(self.search_key, text, 0)
            
        try_pos = bisect.bisect_left(self.search_key, key)
            
        return get_neib_pos(self.search_key, text, try_pos)

    def get_neibs(self, text):
        res = []
        for tpos in self.search_neibs(text):
            self.cmp_form_norm(text, tpos, res)
        return res
    
    def get_acc_pos(self, text):
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


if __name__ == "__main__":        
    pass  
