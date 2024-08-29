# coding=utf-8

import os
import sys
from .transcriptor import Transcriptor
from .accentuator import Accentuator


def punct_filter(input_string, punct=None):
    output_string = []
    is_stace = False
    for tc in input_string:
        if punct is None or tc in punct:
            output_string.append(tc)
        else:
            if not is_stace:
                output_string.append(' ')
                is_stace = True
    return ''.join(output_string)


def find_model(file_name='accentuator_transcriptor_tiny', cache_dir=None, download=True):    
    from pathlib import Path
    if cache_dir is None:
        try:
            omogr_cache = Path(os.getenv('OMOGR_CACHE', Path.home() / '.omogr_data'))
        except (AttributeError, ImportError):
            omogr_cache = os.getenv('OMOGR_CACHE', os.path.join(os.path.expanduser("~"), '.omogr_data'))

        cache_dir = omogr_cache

    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if not cache_dir:
        raise EnvironmentError('Cannot find OMOGR_CACHE path')

    if os.path.exists(cache_dir):
        if os.path.isdir(cache_dir):
            etag_file_name = os.path.join(cache_dir, 'etag')
            if os.path.isfile(etag_file_name):
                return cache_dir

    if not download:
        raise EnvironmentError('Cannot find model data')

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    if not os.path.isdir(cache_dir):
        raise EnvironmentError('Cannot create directory %s'%cache_dir)
            
    from .downloader import download_model
    return download_model(cache_dir, file_name=file_name)
        

class AccentuatorTranscriptor:
    def __init__(self, data_path=None, download=True, device_name=None, punct='.,!?'):        
        loaded_data_path = find_model(file_name='accentuator_transcriptor_tiny',
            cache_dir=data_path, download=download)
            
        self.punct = punct
        transcriptor_data_path = os.path.join(loaded_data_path, 'transcriptor/')
        self.transcriptor = Transcriptor(data_path=transcriptor_data_path)
        
        accentuator_data_path = os.path.join(loaded_data_path, 'accentuator/')
        self.accentuator = Accentuator(data_path=accentuator_data_path, device_name=device_name)
    
    def accentuate(self, text):
        return self.accentuator.accentuate(text)
        
    def transcribe(self, sentence_list):
        sentence_word_list = self.accentuator.accentuate_by_words(sentence_list)
        transcribed_sentence_list = []
        for t_sentence in sentence_word_list:
            transcribed_sentence = []
            for t_punct, t_word in t_sentence:
                if t_punct:
                    transcribed_sentence.append(punct_filter(t_punct, punct=self.punct))
                if t_word:
                    transcribed_sentence.append(self.transcriptor.transcribe(t_word))
            transcribed_sentence_list.append(''.join(transcribed_sentence))
        return transcribed_sentence_list

    
if __name__ == "__main__":
    pass
    
