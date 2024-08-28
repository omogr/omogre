# coding=utf-8

import os
import pickle
from .unk_words import UnkWords

vowels = 'аеёиоуыэюяАЕЁИОУЫЭЮЯ'

def get_single_vowel(src_str):
    pos = None
    for indx, tc in enumerate(src_str):
        if tc in vowels:
            if pos is not None:
                return None
            pos = indx
    return pos        


def normalize_accent(src_str):
    # .casefold() ?
    if '+' in src_str:
        return src_str
    if 'ё' in src_str:
        return src_str.replace('ё', '+ё')

    vowel_pos = get_single_vowel(src_str)
    if vowel_pos is None:
        return src_str
    return src_str[:vowel_pos] + '+' + src_str[vowel_pos:]


def get_g2p_without_accent(grapheme_to_phoneme_vocab):
    grapheme_phoneme = {}
    grapheme_freq = {}
    for grapheme_with_accent, phoneme_vars in grapheme_to_phoneme_vocab.items():        
        grapheme = grapheme_with_accent.replace('+', '')
        for phoneme, freq in phoneme_vars.items():
            if (grapheme_freq.get(grapheme, 0) < freq):
                grapheme_phoneme[grapheme] = phoneme
                grapheme_freq[grapheme] = freq        
    return grapheme_phoneme

    
def get_max_freq_phoneme(key, vocab):   
    if key not in vocab:
        return None
    max_freq = -1
    max_phoneme = None

    for t_phoneme, freq in vocab[key].items():
        if freq > max_freq:
            max_freq = freq
            max_phoneme = t_phoneme
    return max_phoneme


class Transcriptor:
    def __init__(self, data_path):
        transcriptor_data_path = os.path.join(data_path, 'word_vocab.pickle')
        with open(transcriptor_data_path, "rb") as finp:
            self.grapheme_to_phoneme_vocab = pickle.load(finp)

        self.g2p_without_accent = get_g2p_without_accent(self.grapheme_to_phoneme_vocab)
        self.unk_words = UnkWords(data_path=data_path)

    def transcribe(self, src_str): 
        word_str = src_str.casefold()
        word_str_norm = normalize_accent(word_str)
        if '+' in word_str_norm:
            max_phoneme = get_max_freq_phoneme(word_str_norm, self.grapheme_to_phoneme_vocab)
            if max_phoneme is not None:
                return max_phoneme
        
        if word_str in self.g2p_without_accent:
            return self.g2p_without_accent[word_str]
       
        return self.unk_words.transcribe(word_str)
        

if __name__ == "__main__":
    pass
    
