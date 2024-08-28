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

from .reader import AccentTokenizer, AccentDocument
from .unk_model import UnkModel

WordAcc = collections.namedtuple("WordAcc", ["first", "last", "pos", "state"])
PunctWord = collections.namedtuple("WordPunct", ["punct", "word"])

vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
vowel_plus = "аеёиоуыэюяАЕЁИОУЫЭЮЯ+"

def count_vowels(text):
    return sum(1 for char in text if char in vowels)


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
   

def list_arg_max(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])


def norm_word(tword):
    return tword.casefold().replace('ё', 'е').replace(' ', '!')


def check_ee_comu(tword):
    lcw = tword.casefold()
    if lcw == 'кому':
        return 3
    acc_pos = lcw.find('ё')
    if acc_pos >= 0:
        return acc_pos
    return -1

    
class Accentuator:
    def __init__(self, data_path, device_name=None):
        model_data_path = os.path.join(data_path, 'model')
        self.unk_model = UnkModel(data_path, device_name=device_name)
        
        self.tokenizer = AccentTokenizer(data_path=data_path)
        if device_name is None:           
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)
            
        self.model = BertForTokenClassification.from_pretrained(model_data_path, num_labels=10, cache_dir=None)
        assert self.model
        self.model.eval()
        self.model.to(self.device)
        self.error_counter = 0

    def seed(self, seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)

    def process_model_batch(self, doc, batch, sum_batch):
        sentence_spans, input_ids, attention_mask = batch 
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask)
            logits = logits.detach().cpu().tolist() # cpu().
            input_ids = input_ids.detach().cpu().tolist() # cpu().
                        
            for batch_indx, (t_logits, t_input_ids) in enumerate(zip(logits, input_ids)):
                doc_pos, all_spans = sentence_spans[batch_indx]

                sentence = doc.all_sentences[doc_pos]
                word_list = []

                for (first_word_pos, last_word_pos), (first_token_pos, last_token_pos) in all_spans:
                    tword = sentence[first_word_pos:last_word_pos]

                    acc_pos = check_ee_comu(tword)
                    if acc_pos >= 0:
                        tw = WordAcc(first_word_pos, last_word_pos, acc_pos, 'eee')
                        word_list.append(tw)
                        continue

                    if count_vowels(tword) < 1:
                        tw = WordAcc(first_word_pos, last_word_pos, -1, 'emp')
                        word_list.append(tw)
                        continue
                    
                    tword_key = norm_word(tword)
                    word_acc_id = self.tokenizer.accent_vocab.vocab.get(tword_key, 0)
                    acc_pos = self.tokenizer.accent_vocab.vocab_index[word_acc_id]

                    if len(acc_pos) < 1:                        
                        acc_pos = self.unk_model.get_acc_pos(tword.casefold())
                        tw = WordAcc(first_word_pos, last_word_pos, acc_pos, 'oov')
                        word_list.append(tw)
                        continue
                    
                    if len(acc_pos) < 2:                        
                        tw = WordAcc(first_word_pos, last_word_pos, acc_pos[0], 'sin')
                        word_list.append(tw)
                        continue

                    best_word_index = -1
                    best_word_prob = -1.0
                    best_letter_pos = -1
                    letter_pos = 0

                    for token_indx in range(first_token_pos, last_token_pos):
                        soft_probs = _compute_softmax(t_logits[token_indx])
                        ct = self.tokenizer.token_vowel_pos.get(t_input_ids[token_indx])
                        if ct is None:
                            self.error_counter += 1
                            break
                        num_letters, vowel_pos = ct
                            
                        best_prob = 0
                        best_index = -1
                        
                        for prob_index, tprob in enumerate(soft_probs):
                            if prob_index < 1:
                                continue
                            if prob_index > len(vowel_pos):
                                break
                            tpos = letter_pos + vowel_pos[prob_index-1]

                            if tpos not in acc_pos:
                                continue
                            if tprob > best_prob:
                                best_prob = tprob
                                best_index = prob_index

                        if best_index > 0:
                            if best_prob > best_word_prob:
                                best_word_index = best_index
                                best_word_prob = best_prob
                                best_letter_pos = letter_pos + vowel_pos[best_index-1]
                        letter_pos += num_letters           
                            
                    if best_letter_pos >= 0:
                        tw = WordAcc(first_word_pos, last_word_pos, best_letter_pos, 'var')                        
                        word_list.append(tw)
                        continue
                        
                    acc_pos = self.unk_model.get_acc_pos(tword.casefold())
                    tw = WordAcc(first_word_pos, last_word_pos, acc_pos, 'unk')
                    word_list.append(tw)    

                ct = (doc_pos, word_list)
                sum_batch.append(ct)

    def process_without_bert(self, doc, doc_pos, all_spans):
        sentence = doc.all_sentences[doc_pos]
        word_list = []

        for (first_word_pos, last_word_pos), (first_token_pos, last_token_pos) in all_spans:
            tword = sentence[first_word_pos:last_word_pos]
            
            acc_pos = check_ee_comu(tword)
            if acc_pos >= 0:
                tw = WordAcc(first_word_pos, last_word_pos, acc_pos, 'eee')
                word_list.append(tw)
                continue

            if count_vowels(tword) < 1:
                tw = WordAcc(first_word_pos, last_word_pos, -1, 'emp')
                word_list.append(tw)
                continue

            tword_key = norm_word(tword)
            # state = 'var'
            word_acc_id = self.tokenizer.accent_vocab.vocab.get(tword_key)
            if word_acc_id is not None:
                acc_pos = self.tokenizer.accent_vocab.vocab_index[word_acc_id]
                if len(acc_pos) == 1:
                    assert len(acc_pos)
                    tw = WordAcc(first_word_pos, last_word_pos, acc_pos[0], 'sin')
                    word_list.append(tw)
                    continue
                    
            acc_pos = self.unk_model.get_acc_pos(tword.casefold())
            tw = WordAcc(first_word_pos, last_word_pos, acc_pos, 'unk')
            word_list.append(tw)
        return word_list

    def easy_loop(self, doc, sum_batch):
        for doc_pos, all_spans in doc.easy_sentences:
            word_list = self.process_without_bert(doc, doc_pos, all_spans)
            ct = (doc_pos, word_list)
            sum_batch.append(ct)

    def result_to_text(self, sentence, word_list):
        scale = [0 for _ in range(len(sentence))]
        for first_word_pos, last_word_pos, acc_pos, state in word_list:
            if acc_pos >= 0:
                tp = first_word_pos + acc_pos
                assert tp < len(sentence)
                scale[tp] = 1

        res = []
        for indx, tc in enumerate(sentence):
            if scale[indx]:
                res.append('+')
            res.append(tc)
        return ''.join(res)

    def result_to_word_list(self, sentence, word_list):
        prev_pos = 0
        out_word_list = []
        for first_word_pos, last_word_pos, acc_pos, state in word_list:
            if acc_pos < 0:
                tword = sentence[first_word_pos:last_word_pos]
            else:
                tword_letters = []
                rel_pos = acc_pos + first_word_pos
                for tpos in range(first_word_pos, last_word_pos):
                    if tpos == rel_pos:
                        tword_letters.append('+')
                    tword_letters.append(sentence[tpos])
                tword = ''.join(tword_letters)
                
            punct = sentence[prev_pos:first_word_pos]
            prev_pos = last_word_pos
            ct = PunctWord(punct, tword)
            out_word_list.append(ct)
        punct = sentence[prev_pos:]    
        ct = PunctWord(punct, '')
        out_word_list.append(ct)
        return out_word_list
    
    def accentuate_by_words(self, input_sentence_list):
        if not bool(input_sentence_list):
            raise ValueError('list of strings is required')

        if not isinstance(input_sentence_list, list):
            raise ValueError('list of strings is required')
            
        if not all([isinstance(elem, str) for elem in input_sentence_list]):
            raise ValueError('list of strings is required')
            
        doc = AccentDocument(self.tokenizer, input_sentence_list)
        
        sum_batch = []
        self.easy_loop(doc, sum_batch)
        for t_batch in doc.model_batches:
            self.process_model_batch(doc, t_batch, sum_batch)

        sum_batch_index = []
        for indx in range(len(sum_batch)):
            pos = sum_batch[indx][0]
            sum_batch_index.append((pos, indx))
                       
        output_sentence_word_list = []    

        for pos, indx in sorted(sum_batch_index):
            sentence = doc.all_sentences[pos]
            sentence_words = self.result_to_word_list(sentence, sum_batch[indx][1])
            output_sentence_word_list.append(sentence_words)
        return output_sentence_word_list

    def accentuate_sentence_list(self, input_sentence_list):
        if not bool(input_sentence_list):
            raise ValueError('a list of strings is required')
        if not isinstance(input_sentence_list, list):
            raise ValueError('a list of strings is required')
        if not all([
                isinstance(elem, str) for elem in input_sentence_list]):
            raise ValueError('a list of strings is required')
            
        output_sentence_word_list = self.accentuate_by_words(input_sentence_list)    
        output_sentence_text_list = []
        for t_sentence in output_sentence_word_list:
            word_list = []
            for punct, tword in t_sentence:
                word_list.append(punct)
                word_list.append(tword)
            output_sentence_text_list.append(''.join(word_list))
            
        return output_sentence_text_list

    def accentuate(self, input_text):
        if not bool(input_text):
            raise ValueError('a string or list of strings is required')
        if isinstance(input_text, list):
            input_sentence_list = input_text

            if not all([
                    isinstance(elem, str) for elem in input_sentence_list]):
                raise ValueError('a string or list of strings is required')

        else:
            if isinstance(input_text, str):
                input_sentence_list = [input_text]
            else:
                raise ValueError('a string or list of strings is required')

        output_sentence_list = self.accentuate_sentence_list(input_sentence_list)    
            
        if isinstance(input_text, str):
            return "\n".join(output_sentence_list)
        return output_sentence_list

    # ---------------------------- this is for debugging -----------------------------

    def process_all_easy_sentence(self, doc, doc_pos, all_spans):
        sentence = doc.all_sentences[doc_pos]
        word_list = []

        for (first_word_pos, last_word_pos), (first_token_pos, last_token_pos) in all_spans:
            tword = sentence[first_word_pos:last_word_pos]
            
            acc_pos = check_ee_comu(tword)
            if acc_pos >= 0:
                tw = WordAcc(first_word_pos, last_word_pos, [acc_pos], 'eee')
                word_list.append(tw)
                continue

            tword_key = norm_word(tword)

            word_acc_id = self.tokenizer.accent_vocab.vocab.get(tword_key)
            if word_acc_id is not None:
                acc_pos = self.tokenizer.accent_vocab.vocab_index[word_acc_id]

                tw = WordAcc(first_word_pos, last_word_pos, acc_pos, 'sin')
                word_list.append(tw)
                continue

            tw = WordAcc(first_word_pos, last_word_pos, [], 'unk')
            word_list.append(tw)
        return word_list

    def all_easy_loop(self, doc, sum_batch):
        for doc_pos, all_spans in doc.easy_sentences:
            word_list = self.process_all_easy_sentence(doc, doc_pos, all_spans)
            ct = (doc_pos, word_list)
            sum_batch.append(ct)

    def all_easy_to_word_list(self, sentence, word_list):
        prev_pos = 0
        out_word_list = []
        for first_word_pos, last_word_pos, acc_pos, state in word_list:
            tword = sentence[first_word_pos:last_word_pos]

            if len(acc_pos) < 1:
                tword = sentence[first_word_pos:last_word_pos]
            else:
                tword_letters = []
                stress_sign = '+'
                
                for tpos in range(first_word_pos, last_word_pos):
                    rel_pos = tpos - first_word_pos
                    
                    if not acc_pos:
                        tword_letters.append(stress_sign)
                    elif rel_pos in acc_pos:
                        tword_letters.append(stress_sign)
                    tword_letters.append(sentence[tpos])
                tword = ''.join(tword_letters)
                
            punct = sentence[prev_pos:first_word_pos]
            prev_pos = last_word_pos
            ct = PunctWord(punct, tword)
            out_word_list.append(ct)
        punct = sentence[prev_pos:]    
        ct = PunctWord(punct, '')
        out_word_list.append(ct)
        return out_word_list

    def accentuate_all_easy(self, input_sentence_list):
        if not bool(input_sentence_list):
            raise ValueError('list of strings is required')

        if not isinstance(input_sentence_list, list):
            raise ValueError('list of strings is required')
            
        if not all([isinstance(elem, str) for elem in input_sentence_list]):
            raise ValueError('list of strings is required')
            
        doc = AccentDocument(self.tokenizer, input_sentence_list)
        
        doc.get_all_easy(self.tokenizer)
        
        sum_batch = []
        self.all_easy_loop(doc, sum_batch)

        sum_batch_index = []
        for indx in range(len(sum_batch)):
            pos = sum_batch[indx][0]
            sum_batch_index.append((pos, indx))
            
        output_sentence_word_list = []    

        for pos, indx in sorted(sum_batch_index):
            sentence = doc.all_sentences[pos]
            sentence_words = self.all_easy_to_word_list(sentence, sum_batch[indx][1])
            output_sentence_word_list.append(sentence_words)

        output_sentence_text_list = []
        for t_sentence in output_sentence_word_list:
            word_list = []
            for punct, tword in t_sentence:
                word_list.append(punct)
                word_list.append(tword)
            output_sentence_text_list.append(''.join(word_list))
            
        return output_sentence_text_list
