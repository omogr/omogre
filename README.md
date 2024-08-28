# Russian accentuator and IPA transcriptor.

# Расстановка ударений и IPA транскрипция для русского языка.

Библиотека состоит из двух независимых частей (**Accentuator** и **Transcriptor**), которые могут работать последовательно.

**Accentuator** получает на вход список предложений и возвращает эти предложения с расставленными ударениями.

**Transcriptor** получает на вход список предложений с расставленными ударениями и возвращает их транскрипцию.

## Способы установки

* С помощью GIT
```
pip install git+https://github.com/omogr/omogre.git
```

* При помощи pip
Скачать код с гитхаба. Hаходясь в корневой директории проекта (в той, в которой находится setup.py), выполнить команду
```
pip install -e .
```

* Вручную

Если хочется предварительно оценить работоспособность пакета, то можно скачать код с гитхаба, руками установить необходимые библиотеки (torch, numpy, tqdm, requests), запустить скрипт test.py
## Скачивание данных

По умолчанию, если не указывать путь, то при первом запуске библиотеки данные для моделей скачиваются автоматически в директорию по умолчанию.

При желании можно указывать путь, данные будут скачаны в указанную директорию. Если в этой директории уже есть данные, то повторного скачивания данных не будет.

## Пример запуска (скрипт test.py)

```
from omogre import AccentuatorTranscriptor

# данные будут скачаны в директорию 'omogre_data'
accentuator_transcriptor = AccentuatorTranscriptor(data_path='omogre_data')

sentence_list = ['на двери висит замок.']

print(accentuator_transcriptor.accentuate(sentence_list))        
print(accentuator_transcriptor.transcribe(sentence_list))
```
       
## Параметры инициализации класса AccentuatorTranscriptor

Все параметры не являются обязательными. 

class AccentuatorTranscriptor(load_model=True, data_path=None, device_name=None, punct='.,!?')

	load_model - загружать ли данные, если они ещё не загружены
	data_path - директория, в которую загружать данные.
	device_name - параметр для torch.load_model. Допустимые значения - "cpu", "cuda", "cuda:0" и т.д.
		По умолчанию "cuda", если есть соответствующее GPU, иначе "cpu".
	 
Входы класса **AccentuatorTranscriptor**:

	accentuate(sentence_list)
	transcribe(sentence_list)
	
В случае **accentuate** выполняется только расcтановка ударений, в случае **transcribe** - расcтановка ударений и последующая транскрипция.

Оба входа получают на вход список предложений.
Оба входа возвращают список предложений, длина которого равняется длине входного списка.

## Пример работы

Скрипт ruslan_markup.py расставляет ударения и порождает транскрипцию для markup-файлов акустических корпусов [ruslan](http://dataset.sova.ai/SOVA-TTS/ruslan/ruslan_dataset.tar) и [natasha](http://dataset.sova.ai/SOVA-TTS/natasha/natasha_dataset.tar).

markup-файлы этих корпусов уже содержат расстановку ударений, которая была [сделана](https://habr.com/ru/companies/ashmanov_net/articles/528296/) вручную. 

Скрипт ruslan_markup.py порождает для тех же файлов свою собственную расстановку ударений. Изначальная ручная разметка никак не использовалась ни при тестировании ни при обучении. Таким образом, её можно использовать для оценки точности расстановки ударений.

## Предварительная обработка текста и некоторые особенности транскрипции

Библиотека не занимается предварительной обработкой текста, т.е. разбиением текста на предложения, обработкой сокращений, обработкой цифр и других небуквенных символов. Предполагается, что поступающий на вход текст уже разбит на предложения.

### Accentuator

При расстановке ударений небуквенные символы никак не меняются.

### Transcriptor

При транскрипции посторонние символы фильтруются (за исключением фиксированного списка символов, который можно задавать отдельным параметром). По умолчанию не фильтруются только четыре знака пунктуации.
 
## Учёт контекста

### Accentuator
Ударения расставляются с учётом контекста. Если на вход вместо списка отдельных предложений подавать очень большие куски текста, то контекст учитываться не будет. Ударения будут ставиться только в тех случаях, когда это возможно без учёта контекста.

Для слов из одного слова ударение тоже ставится. В некоторых случаях это может выглядеть странно, но упрощает последующее определение транскрипции. 

### Transcriptor
Транскрипция определяется пословно, без учёта контекста.

## Зависимости

Для запуска библиотеки требуются следующие модули:
* [Python 3](https://www.python.org/)
* [Pytorch](https://pytorch.org/)
* [Numpy](https://numpy.org/)
Кроме того, для автоматического скачивания моделей требуются:
* [tqdm](https://tqdm.github.io/)
* [requests](python -m pip install requests)

## Загрузка данных без запуска AccentuatorTranscriptor.

Если запустить скрипт download_data.py, то данные будут скачаны. В качестве параметров запуска можно указать директорию, в которую скачивать данные.

## Обратная связь
Почта для вопросов, замечаний и предложений - omogrus@ya.ru.

## Лицензия
CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ru)

