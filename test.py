from omogre import Accentuator, Transcriptor

# данные будут скачаны в директорию 'omogre_data'

transcriptor = Transcriptor(data_path='omogre_data')

accentuator = Accentuator(data_path='omogre_data')

sentence_list = ['стены замка']

print('transcriptor()', transcriptor(sentence_list))
print('accentuator()', accentuator(sentence_list))

print('transcriptor.transcribe', transcriptor.transcribe(sentence_list))
print('accentuator.accentuate', accentuator.accentuate(sentence_list))

print('transcriptor.accentuate', transcriptor.accentuate(sentence_list))
