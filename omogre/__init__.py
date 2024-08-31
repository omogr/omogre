# coding=utf-8

import os
import sys
from .transcriptor import Transcriptor as TranscriptorImpl
from .accentuator import Accentuator as AccentuatorImpl

INITIAL_MODEL = 'accentuator_transcriptor_tiny'

def punctuation_filter(input_string: str, punct: str=None):
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


def find_model(file_name: str = INITIAL_MODEL, cache_dir: str = None, download: bool = True, reload=False):    
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
                if not reload:
                    return cache_dir

    if not download:
        raise EnvironmentError('Cannot find model data')

    print('data_path', cache_dir, file=sys.stderr)
        
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    if not os.path.isdir(cache_dir):
        raise EnvironmentError('Cannot create directory %s'%cache_dir)
            
    from .downloader import download_model
    return download_model(cache_dir, file_name=file_name)
        

class Transcriptor:
    def __init__(self, data_path: str = None, download: bool = True, device_name:str = None, punct:str = '.,!?'):        
        loaded_data_path = find_model(file_name=INITIAL_MODEL,
            cache_dir=data_path, download=download)
            
        self.punct = punct
        transcriptor_data_path = os.path.join(loaded_data_path, 'transcriptor/')
        self.transcriptor = TranscriptorImpl(data_path=transcriptor_data_path)
        
        accentuator_data_path = os.path.join(loaded_data_path, 'accentuator/')
        self.accentuator = AccentuatorImpl(data_path=accentuator_data_path, device_name=device_name)
    
    def accentuate(self, text):
        return self.accentuator.accentuate(text)
        
    def transcribe(self, sentence_list: list) -> list:
        sentence_word_list = self.accentuator.accentuate_by_words(sentence_list)
        transcribed_sentence_list = []
        for t_sentence in sentence_word_list:
            transcribed_sentence = []
            for t_punct, t_word in t_sentence:
                if t_punct:
                    transcribed_sentence.append(punctuation_filter(t_punct, punct=self.punct))
                if t_word:
                    transcribed_sentence.append(self.transcriptor.transcribe(t_word))
            transcribed_sentence_list.append(''.join(transcribed_sentence))
        return transcribed_sentence_list

    def __call__(self, sentence_list: list) -> list:
        return self.transcribe(sentence_list)


class Accentuator:
    def __init__(self, data_path: str = None, download: bool = True, device_name:str = None):        
        loaded_data_path = find_model(file_name=INITIAL_MODEL,
            cache_dir=data_path, download=download)
                    
        accentuator_data_path = os.path.join(loaded_data_path, 'accentuator/')
        self.accentuator = AccentuatorImpl(data_path=accentuator_data_path, device_name=device_name)
    
    def accentuate(self, text):
        return self.accentuator.accentuate(text)
        
    def __call__(self, text):
        return self.accentuate(text)