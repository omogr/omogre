# coding=utf-8

import os
import collections
import torch
from .tokenizer import BertTokenizer
import pickle

InfTokenSpan = collections.namedtuple("TokenSpan", ["word_tokens", "punct", "first", "last"])

alphabet = '-абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
vowels = 'аеёиоуыэюяАЕЁИОУЫЭЮЯ'

class AccentVocab:
    def __init__(self, data_path):
        vocab_file = os.path.join(data_path, 'wv_word_acc.pickle')
        with open(vocab_file, "rb") as finp:
            (self.vocab, self.vocab_index) = pickle.load(finp)


class AccentTokenizer:
    def __init__(self, data_path):
        self.accent_vocab = AccentVocab(data_path=data_path)
        bert_vocab_path = os.path.join(data_path, 'model/vocab.txt')
        self.tokenizer = BertTokenizer(bert_vocab_path, do_lower_case=False)
        
        self.get_vowel_pos()
        self.pad_token_id = 0

    def get_vowel_pos(self):
        self.token_vowel_pos = {}

        for token_text, token_id in self.tokenizer.vocab.items():
             
            vowel_pos = []
            letter_pos = 0
            if token_text.startswith('##'):
                tt = token_text[2:]
            else:
                tt = token_text
            for letter_pos, tc in enumerate(tt):
                if tc in vowels:
                    vowel_pos.append(letter_pos)
                if letter_pos > 0:
                    assert tc != '#', (token_text, token_id)
            ct = (len(tt), vowel_pos)
            self.token_vowel_pos[token_id] = ct

    def encode(self, txt):
        tokens = self.tokenizer.tokenize(txt)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def get_num_vars(self, tword):
        tword_index = self.accent_vocab.vocab.get(tword)
        if tword_index is None:
            return -1
        return len(self.accent_vocab.vocab_index[tword_index])

    def tokenize_word(self, letter_list, tokens, first_pos, last_pos):
        tword = ''.join(letter_list) # casefold() ?
                
        num_vars = self.get_num_vars(tword) 
        if num_vars >= 0:            
            tokens.append(InfTokenSpan(self.encode(tword), [], first_pos, last_pos)) 
            
            if num_vars > 1:
                return True
            return False
                   
        if '-' not in tword:
            tokens.append(InfTokenSpan(self.encode(tword), [], first_pos, last_pos))
            return False

        parts = tword.split('-')
        if len(parts) < 1:
            return False
            
        tpos = first_pos
            
        for tp in parts:           
            next_pos = tpos + len(tp)
            tokens[-1].punct.append('-')
            if len(tp) > 0:
                tokens.append(InfTokenSpan(self.encode(tp), [], tpos, next_pos))
            tpos = next_pos + 1

        for tp in parts:
            if self.get_num_vars(tp) > 1:
                return True
        return False

    def tokenize_punct(self, letters_list):
        return self.encode(''.join(letters_list))
    
    def get_inf_tokens(self, sentence0):
        sentence = sentence0.replace('+', '')
        tokens = []
        tokenizer_bos = 2
        tokenizer_sep = 3
        
        ct = InfTokenSpan([tokenizer_bos], [], 0, 0)
        tokens.append(ct)
        tword = []
        
        first_pos = -1        
        is_easy = True
        for char_pos, cur_char0 in enumerate(sentence):
            if cur_char0 in alphabet:
                cur_char = cur_char0.casefold()
                if first_pos < 0:
                    first_pos = char_pos
                tword.append(cur_char)
                continue

            if tword:
                if self.tokenize_word(tword, tokens, first_pos, char_pos):
                    is_easy = False

                tword = []
            first_pos = -1
            tokens[-1].punct.append(cur_char0)

        if tword:
            if self.tokenize_word(tword, tokens, first_pos, len(sentence)):
                is_easy = False
            tword = []
                                    
        ct = InfTokenSpan([tokenizer_sep], [], 0, 0)
        tokens.append(ct)

        all_ids = []
        all_spans = []
        for ws in tokens:
            first_token = len(all_ids)
            all_ids.extend(ws.word_tokens)
            
            if ws.last > 0:
                last_token = len(all_ids)
                text_span = (ws.first, ws.last)
                token_span = (first_token, last_token)
                ct = (text_span, token_span)
                all_spans.append(ct)
            
            tpunct_str = ''.join(ws.punct).replace(' ', '')
            if len(tpunct_str) > 0:
                punct_tokens = self.encode(tpunct_str)
                all_ids.extend(punct_tokens)
                
        if len(all_ids) > 510: # 512
            is_easy = True

        return all_ids, all_spans, is_easy


class AccentDocument:
    def __init__(self, acc_tokenizer, all_sentences, first_pos=0, max_batch_token_num=2048):
        self.all_sentences = all_sentences        

        self.max_batch_token_num = max_batch_token_num
        self.easy_sentences = []
        self.model_batches = []
        self.too_long_sentence_cnt = 0
        self.pad_token_id = acc_tokenizer.pad_token_id

        self.get_batches(acc_tokenizer, first_pos)
        
    def num_entries(self):
        return len(self.sentence_list)
        
    def add_model_batch(self, bert_sentences, max_length):
        all_input_ids = []
        sentence_spans = []
        all_attention_mask = []
        
        batch_length = max_length
            
        for t_doc_pos, t_input_ids, t_spans in bert_sentences:
            len_input_ids = len(t_input_ids)
            attention_mask = [1] * len_input_ids
            ct = (t_doc_pos, t_spans)
            sentence_spans.append(ct)
            if len_input_ids > batch_length:
                # Truncate t_input_ids and attention_mask to max length
                assert False
                t_input_ids = t_input_ids[:batch_length]
                
                attention_mask = attention_mask[:batch_length]
            elif len(t_input_ids) < batch_length:
                # Pad t_input_ids and attention_mask to max length
                padding_length = batch_length - len_input_ids
                t_input_ids += [self.pad_token_id] * padding_length
                attention_mask += [0] * padding_length

            all_input_ids.append(torch.tensor(t_input_ids, dtype=torch.long))
            all_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
    
        all_input_ids = torch.stack(all_input_ids)
        all_attention_mask = torch.stack(all_attention_mask)
        batch = (sentence_spans, all_input_ids, all_attention_mask)
        
        self.model_batches.append(batch)
    
    def get_batches(self, acc_tokenizer, first_pos):        
        sorted_bert_sentences = []
        doc_pos = first_pos
        
        while doc_pos < len(self.all_sentences):

            cur_sentence = self.all_sentences[doc_pos]
            all_ids, all_spans, is_easy = acc_tokenizer.get_inf_tokens(cur_sentence)
            if is_easy:
                ct = (doc_pos, all_spans)
                self.easy_sentences.append(ct)
                doc_pos += 1
                continue
            
            len_input_ids = len(all_ids)
            
            if (1 + len_input_ids) >= self.max_batch_token_num:
                ct = (doc_pos, cur_sentence, all_spans)
                easy_sentences.append(ct)
                doc_pos += 1
                self.too_long_sentence_cnt += 1
                continue
                
            ct = (len_input_ids, doc_pos, all_ids, all_spans)
            sorted_bert_sentences.append(ct)
            doc_pos += 1

        sorted_bert_sentences.sort()

        max_length = 1
        bert_sentences = []

        for len_input_ids, doc_pos, all_ids, all_spans in sorted_bert_sentences:
            new_max_length = max(max_length, len_input_ids)
            if new_max_length * (1 + len(bert_sentences)) >= self.max_batch_token_num:
                assert len(bert_sentences) > 0
                self.add_model_batch(bert_sentences, max_length)
                max_length = len_input_ids
                bert_sentences = []               
            else:
                max_length = new_max_length

            ct = (doc_pos, all_ids, all_spans)
            bert_sentences.append(ct)

        if len(bert_sentences) > 0:
            self.add_model_batch(bert_sentences, max_length)

    def get_all_easy(self, acc_tokenizer, first_pos=0):
        doc_pos = first_pos
        self.easy_sentences = []
        while doc_pos < len(self.all_sentences):
            all_ids, all_spans, is_easy = acc_tokenizer.get_inf_tokens(self.all_sentences[doc_pos])

            ct = (doc_pos, all_spans)
            self.easy_sentences.append(ct)
            doc_pos += 1


if __name__ == '__main__':
    pass
    