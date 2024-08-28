from omogre import AccentuatorTranscriptor

# данные будут скачаны в директорию 'omogre_data'
accentuator_transcriptor = AccentuatorTranscriptor(data_path='omogre_data')

sentence_list = ['на двери висит замок.']

print('accentuate', accentuator_transcriptor.accentuate(sentence_list))
   
print('transcribe', accentuator_transcriptor.transcribe(sentence_list))
