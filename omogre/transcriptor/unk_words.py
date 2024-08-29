# coding=utf-8

import os
import sys
import numpy as np
import pickle

def invert_vocab(vocab):
    inv = {}
    for key, value in vocab.items():
        inv[value] = key        
    return inv


def clean_russian_g2p_trascription(text: str) -> str:
    
    result = text.replace("<DELETE>", "").replace("+", "").replace("~", "").replace("ʑ", "ɕ:")
    return result.replace("ɣ", "x").replace(":", "ː").replace("'", "`").replace("_", "")
    

def get_vocab_from_list(alphabet):
    vocab = {}
    for indx, tc in enumerate(alphabet):
        vocab[tc] = indx
    return vocab


def kaikki_id_search(ids, kaikki_ids):
    tids = tuple(ids)
    first_pos = 0
    last_pos = len(kaikki_ids)
    while True:
        mid_search = int( (last_pos - first_pos) / 2)
        if mid_search < 1:
            break
            
        mid_pos = first_pos + mid_search
        if kaikki_ids[mid_pos][0] > tids:
            last_pos = mid_pos
            continue
        first_pos = mid_pos
        if kaikki_ids[mid_pos][0] == tids:
            break            
    return first_pos    


class UnkWords:
    def __init__(self, data_path):
        transcriptor_data_path = os.path.join(data_path, 'gram_model.pickle')
        with open(transcriptor_data_path, 'rb') as finp:
            (
                self.pattern_alphabet,
                self.gram_prob,
                self.bi_eval,
                self.pt,
                self.pt2,
                self.kaikki_ids0,
                self.kaikki_ids1
            ) = pickle.load(finp)

        self.inv_pattern_alphabet = invert_vocab(self.pattern_alphabet)
        self.ru_alphabet = ' <>-`абвгдеёжзийклмнопрстуфхцчшщъыьэюя?'
        self.input_token_vocab = get_vocab_from_list(self.ru_alphabet)
        self.stress_sign = "`"
        
        self.bos_sign = '<BOW>'
        self.stress_indx = self.input_token_vocab.get(self.stress_sign)
        self.delete_indx = self.pattern_alphabet.get('<DELETE>', -1)
        self.pattern_stress_indx = self.pattern_alphabet.get("'", -1)

        assert self.delete_indx >= 0
        assert self.pattern_stress_indx >= 0
        assert self.stress_indx == 4

    def get_input_ids(self, input_text):
        input_letters = [tc for tc in input_text]
        input_letters_plus = ['<', '<'] + input_letters + ['>', '>']
        input_ids = [self.input_token_vocab.get(tc, 0) for tc in input_letters_plus]           
        return input_ids

    def search_step(self, best_path, all_char_eval, window_size=7):
        new_best = {}
        cf = 1.0
        for char_indx, t_char_eval in enumerate(all_char_eval):
            if t_char_eval < -99.0:
                continue
            tc = self.inv_pattern_alphabet[char_indx]
            
            for path_indx, (path_eval, _, phoneme1, phoneme2) in enumerate(best_path):
                key = (phoneme1, phoneme2, tc)
                if key in self.gram_prob:
                    curgr_eval = self.gram_prob[key]
                else:
                    curgr_eval = -1.31 * self.bi_eval.get((phoneme1, phoneme2), 20)
                    
                cur_eval = path_eval + cf * curgr_eval + t_char_eval
                new_key = (phoneme2, tc)
                if new_key not in new_best:
                    new_best[new_key] = (cur_eval, path_indx)
                else:
                    tv, _ = new_best[new_key]
                    if cur_eval > tv:
                        new_best[new_key] = (cur_eval, path_indx)
                        
        best_list = sorted(            [
                (cur_eval, path_indx, phoneme1, phoneme2)
                    for (phoneme1, phoneme2), (cur_eval, path_indx) in new_best.items()
            ], reverse=True)
        return best_list[:window_size]

    def transcribe(self, input_word_text):
        if not input_word_text:
            return ""
            
        word_text = input_word_text.casefold().replace('+', self.stress_sign)    
        input_ids = self.get_input_ids(word_text)
        best_path = {}
        pattern_alphabet_len = len(self.pattern_alphabet)
        word_len = len(input_ids)

        bin_pos0 = kaikki_id_search(input_ids, self.kaikki_ids0)
        pattern_len = -1
        for indx, tid in enumerate(self.kaikki_ids0[bin_pos0][0]):
            if indx >= len(input_ids):
                break
            if tid != input_ids[indx]:
                break

            pattern_indx = self.kaikki_ids0[bin_pos0][1][indx]
            if pattern_indx in [self.delete_indx, self.pattern_stress_indx]:
                continue
            pattern_len = indx
       
        tt = list(reversed(input_ids))
        bin_pos1 = kaikki_id_search(tt, self.kaikki_ids1)
        pattern_len1 = -1
        for indx, tid in enumerate(self.kaikki_ids1[bin_pos1][0]):
            if indx >= len(input_ids):
                break
            if tid != tt[indx]:
                break
            pattern_indx = self.kaikki_ids1[bin_pos1][1][indx]
            if pattern_indx in [self.delete_indx, self.pattern_stress_indx]:
                continue
            pattern_len1 = indx
              
        best_path[1] = [ (0.0, 0, self.bos_sign, self.bos_sign) ]
        indx = 2
        while indx < word_len:
            outputs = np.ones(pattern_alphabet_len, dtype=np.float32) * -100.0            
            indx1 = word_len - indx - 1            
            is_empty = True
            if indx < pattern_len:
                t_kaikki_id = self.kaikki_ids0[bin_pos0][1][indx]
                outputs[t_kaikki_id] = 0.0
                is_empty = False
            if indx1 < pattern_len1:
                t_kaikki_id = self.kaikki_ids1[bin_pos1][1][indx1]
                outputs[t_kaikki_id] = 0.0
                is_empty = False
            if is_empty and (indx < word_len-1):
                i1 = indx - 1
                i2 = indx + 1
                if self.stress_indx == input_ids[i1]:
                    if i1 > 0:
                        i1 -= 1
                if self.stress_indx == input_ids[i2]:
                    if i2 < word_len-1:
                        i2 += 1
                src = tuple(input_ids[i1:i2+1])
                if src in self.pt2:
                    is_empty = False
                    for key, value in self.pt2[src].items():
                        outputs[key] = value
            if is_empty:
                src = input_ids[indx]
                if src in self.pt:
                    for key, value in self.pt[src].items():
                        outputs[key] = value
            
            best_path[indx] = self.search_step(best_path[indx-1], outputs)
            indx += 1
                   
        indx = word_len-1
        prev = 0
         
        res = []
        
        while indx > 0 and len(best_path[indx]) > 0:
            path_eval, prev, phoneme1, phoneme2 = best_path[indx][prev]
            res.append(phoneme1)
            indx -= 1

        res.reverse()
        
        if len(res) > 3:
            # strip '<BOW>' '<BOW>' ... '<EOW>'            
            return clean_russian_g2p_trascription(''.join(res[2:-1]))
        return ""


if __name__ == "__main__":
    pass
