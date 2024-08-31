# coding=utf-8

from omogre import Transcriptor, find_model
import time
from argparse import ArgumentParser

parser = ArgumentParser(description="Accentuate and transcribe natasha ruslan markup")

parser.add_argument("--data_path", type=str, default=None, help="Omogre model direcory")
parser.add_argument("--head", type=int, default=None, help="Process only head of file")
args = parser.parse_args()
          
transcriptor = Transcriptor(data_path=args.data_path)

def save_markup(src_markup_parts: list, sentence_list: list, fout_name: str):
    assert len(src_markup_parts) == len(sentence_list)

    with open(fout_name, 'w', encoding='utf-8') as fout:
        for parts, out_sent in zip(src_markup_parts, sentence_list):
            parts[1] = out_sent
            print('|'.join(parts), file=fout)


def process(dataset_name: str):
    print('dataset', dataset_name)
    finp_name = 'natasha_ruslan_markup/%s.txt'%dataset_name

    sentence_list = []
    src_markup_parts = []
    print('reading', finp_name)

    with open(finp_name, 'r', encoding='utf-8') as finp:
        for entry in finp:
            parts = entry.strip().split('|')
            assert len(parts) >= 2
            
            sentence_list.append(parts[1].replace('+', ''))
            src_markup_parts.append(parts)
            if args.head is not None:
                if len(src_markup_parts) >= args.head:
                    break
                
    start = time.time()
    output_sentences = transcriptor.accentuate(sentence_list)
    dt = time.time() - start
    print('Accentuated', dataset_name, len(sentence_list), 'sentences, dtime %.1f s'%dt)

    fout_name = 'natasha_ruslan_markup/%s.accentuate'%dataset_name
    save_markup(src_markup_parts, output_sentences, fout_name)

    start = time.time()
    output_sentences = transcriptor.transcribe(sentence_list)
    dt = time.time() - start
    print('Transcribed', dataset_name, len(sentence_list), 'sentences, dtime %.1f s'%dt)

    fout_name = 'natasha_ruslan_markup/%s.transcribe'%dataset_name
    save_markup(src_markup_parts, output_sentences, fout_name)


if __name__ == '__main__':
    # Хабр: Open Source синтез речи SOVA
    # https://habr.com/ru/companies/ashmanov_net/articles/528296/
    
    # ruslan
    # http://dataset.sova.ai/SOVA-TTS/ruslan/ruslan_dataset.tar
    # natasha
    # http://dataset.sova.ai/SOVA-TTS/natasha/natasha_dataset.tar

    
    find_model(cache_dir='natasha_ruslan_markup', file_name='natasha_ruslan')

    for dataset_name in ['natasha', 'ruslan']:
        process(dataset_name)

    
    
