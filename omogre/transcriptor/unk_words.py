# coding=utf-8

import os
import sys
import numpy as np
import pickle
import json

INPUT_TEXT_ALPHABET = ' <>-`абвгдеёжзийклмнопрстуфхцчшщъыьэюя?'
BOS_SYMBOL = '<BOW>'
DELETE_SYMBOL = '<DELETE>'
UNK_PATH_EVAL = -100.0
UNK_PATH_THRESHOLD = -99.0

AUXILIARY_SYMBOL_REPLACEMENTS = [
        (DELETE_SYMBOL, ""),
        ("+", ""),
        ("~", ""),
        ("ʑ", "ɕ:"),
        ("ɣ", "x"),
        (":", "ː"),
        ("'", "`"),
        ("_", "")
    ]

class UnkWordsConfig(object):
    """Configuration class to store the configuration of a `UnkWords`.
    """
    def __init__(self,
                 window_size=7,
                 unk_bigram_eval=20,
                 kind_of_stupid_backoff=-1.31,
                 gram_data_name='gram_model.pickle'):
        
            self.window_size = window_size
            self.unk_bigram_eval = unk_bigram_eval
            self.kind_of_stupid_backoff = kind_of_stupid_backoff
            self.gram_data_name = gram_data_name

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


def invert_vocab(vocab: dict) -> dict:
    """Inverts a dictionary, swapping keys and values.
    Args:
       vocab (dict): The dictionary to invert.

    Returns:
       dict: The inverted dictionary.
    """

    inv = {}
    for key, value in vocab.items():
        inv[value] = key        
    return inv


def replace_auxiliary_symbols(text: str) -> str:
    result = text
    for src, dst in AUXILIARY_SYMBOL_REPLACEMENTS:
        if result.find(src) >= 0:
            result = result.replace(src, dst)
    return result
    

def get_input_token_vocab() -> dict:
    # vocab for input text tokenizer
    vocab = {}
    for indx, tc in enumerate(INPUT_TEXT_ALPHABET):
        vocab[tc] = indx
    return vocab


def token_lower_bound(raw_search_key: list, id_list: list) -> int:
    # binary search (a kind of)
    search_key = tuple(raw_search_key)
    first_pos = 0
    last_pos = len(id_list)
    while True:
        half_range = (last_pos - first_pos) // 2
        if half_range < 1:
            break
            
        mid_pos = first_pos + half_range
        if id_list[mid_pos][0] > search_key:
            last_pos = mid_pos
            continue
        first_pos = mid_pos
        if id_list[mid_pos][0] == search_key:
            break            
    return first_pos    


class UnkWords:
    def __init__(self, data_path: str, config=None):
        if config is None:
            self.config = UnkWordsConfig()
        else:
            self.config = config
        transcriptor_data_path = os.path.join(data_path, self.config.gram_data_name)
        with open(transcriptor_data_path, 'rb') as finp:
            (
                self.dst_alphabet,
                self.gram_prob,
                self.bigram_eval,
                self.char_phrase_table,
                self.gram_phrase_table,
                self.head_transcriptions,
                self.tail_transcriptions,
            ) = pickle.load(finp)

        self.inv_dst_alphabet = invert_vocab(self.dst_alphabet)
        
        self.input_token_vocab = get_input_token_vocab()
        self.src_stress_sign = "`"
        
        self.src_stress_indx = self.input_token_vocab.get(self.src_stress_sign)
        self.delete_indx = self.dst_alphabet.get(DELETE_SYMBOL, -1)
        self.dst_stress_indx = self.dst_alphabet.get("'", -1)

        assert self.delete_indx >= 0
        assert self.dst_stress_indx >= 0
        assert self.src_stress_indx == 4

    def tokenize(self, input_text: str) -> list:
        bos_input_letters_eos = ['<', '<'] + [tc for tc in input_text] + ['>', '>']        
        return [self.input_token_vocab.get(tc, 0) for tc in bos_input_letters_eos]

    def viterbi_step(self, best_path: list, emission_logprobs: np.ndarray) -> list:
        """Performs a single step in the Viterbi search algorithm.

        Args:
            best_path (list): The best paths from the previous step.
            emission_logprobs (np.ndarray): Character emission probabilities.

        Returns:
            list: The top `config.window_size` best paths for the current step.
        """
        
        new_best_paths: dict = {}
        for char_indx, t_char_eval in enumerate(emission_logprobs):
            if t_char_eval < UNK_PATH_THRESHOLD:
                continue
            next_dst_char = self.inv_dst_alphabet[char_indx]
            
            for path_indx, (prev_path_eval, _, phoneme1, phoneme2) in enumerate(best_path):
                key = (phoneme1, phoneme2, next_dst_char)
                if key in self.gram_prob:
                    current_gram_eval = self.gram_prob[key]
                else:
                    current_gram_eval = self.config.kind_of_stupid_backoff * self.bigram_eval.get(
                        (phoneme1, phoneme2), self.config.unk_bigram_eval)
                    
                new_path_eval = prev_path_eval + current_gram_eval + t_char_eval
                new_key = (phoneme2, next_dst_char)
                if new_key not in new_best_paths:
                    new_best_paths[new_key] = (new_path_eval, path_indx)
                else:
                    best_so_far, _ = new_best_paths[new_key]
                    if new_path_eval > best_so_far:
                        new_best_paths[new_key] = (new_path_eval, path_indx)
                        
        best_list = sorted(            [
                (path_eval, path_indx, phoneme1, phoneme2)
                    for (phoneme1, phoneme2), (path_eval, path_indx) in new_best_paths.items()
            ], reverse=True)
        return best_list[:self.config.window_size]

    def transcribe(self, input_word_text: str) -> str:
        """
        Receives a word as input, returns its transcription
        """
        if not input_word_text:
            return ""
        
        # in the input text of the word, the stress can be indicated by a plus,
        # we replace it with the symbol that is used in n-grams...
        
        word_text = input_word_text.casefold().replace('+', self.src_stress_sign)    
        input_ids = self.tokenize(word_text)
        dst_alphabet_len = len(self.dst_alphabet)
        word_len = len(input_ids)

        head_lower_bound = token_lower_bound(input_ids, self.head_transcriptions)
        head_len = -1
        for indx, tid in enumerate(self.head_transcriptions[head_lower_bound][0]):
            if indx >= len(input_ids):
                break
            if tid != input_ids[indx]:
                break

            pattern_indx = self.head_transcriptions[head_lower_bound][1][indx]
            if pattern_indx in [self.delete_indx, self.dst_stress_indx]:
                continue
            head_len = indx
       
        reversed_input_ids = list(reversed(input_ids))
        tail_lower_bound = token_lower_bound(reversed_input_ids, self.tail_transcriptions)
        tail_len = -1
        for indx, tid in enumerate(self.tail_transcriptions[tail_lower_bound][0]):
            if indx >= len(input_ids):
                break
            if tid != reversed_input_ids[indx]:
                break
            pattern_indx = self.tail_transcriptions[tail_lower_bound][1][indx]
            if pattern_indx in [self.delete_indx, self.dst_stress_indx]:
                continue
            tail_len = indx
              
        best_path = {}
        best_path[1] = [ (0.0, 0, BOS_SYMBOL, BOS_SYMBOL) ]
        indx = 2
        while indx < word_len:
            emission_logprobs = np.ones(dst_alphabet_len, dtype=np.float32) * UNK_PATH_EVAL            
            indx1 = word_len - indx - 1            
            is_empty = True
            if indx < head_len:
                vocab_char = self.head_transcriptions[head_lower_bound][1][indx]
                emission_logprobs[vocab_char] = 0.0
                is_empty = False
            if indx1 < tail_len:
                vocab_char = self.tail_transcriptions[tail_lower_bound][1][indx1]
                emission_logprobs[vocab_char] = 0.0
                is_empty = False
            if is_empty and (indx < word_len-1):
                i1 = indx - 1
                i2 = indx + 1
                if self.src_stress_indx == input_ids[i1]:
                    if i1 > 0:
                        i1 -= 1
                if self.src_stress_indx == input_ids[i2]:
                    if i2 < word_len-1:
                        i2 += 1
                src = tuple(input_ids[i1:i2+1])
                if src in self.gram_phrase_table:
                    is_empty = False
                    for key, value in self.gram_phrase_table[src].items():
                        emission_logprobs[key] = value
            if is_empty:
                src = input_ids[indx]
                if src in self.char_phrase_table:
                    for key, value in self.char_phrase_table[src].items():
                        emission_logprobs[key] = value
            
            best_path[indx] = self.viterbi_step(best_path[indx-1], emission_logprobs)
            indx += 1
                   
        indx = word_len-1
        prev = 0
        res = []
        
        while indx > 0 and len(best_path[indx]) > 0:
            path_eval, prev, phoneme1, phoneme2 = best_path[indx][prev]
            res.append(phoneme1)
            indx -= 1

        res.reverse()

        if len(res) > 3: # 4?
            # strip BOS_SYMBOL BOS_SYMBOL ... EOS_SYMBOL
            return replace_auxiliary_symbols(''.join(res[2:-1]))
        return ""


