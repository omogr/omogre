# Russian accentuator and IPA transcriptor.

# Расстановка ударений и IPA транскрипция для русского языка.

Библиотека может использоваться либо для расстановки ударений в тексте, либо для его транскрипции.

## Способы установки

* С помощью GIT

```
pip install git+https://github.com/omogr/omogre.git
```

* При помощи pip

Скачать код с гитхаба. Hаходясь в корневой директории проекта (в той, в которой находится `setup.py`), выполнить команду

```
pip install -e .
```

* Вручную

Если хочется предварительно оценить работоспособность пакета, то можно скачать код с гитхаба, руками установить необходимые библиотеки (torch, numpy, tqdm, requests), запустить скрипт `test.py`.

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

```
class AccentuatorTranscriptor(data_path=None, device_name=None, punct='.,!?')
```

`data_path` - директория, в которую загружать данные.

`device_name` - параметр, определяющий использование GPU. Соответствует параметру инициализации класса [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device).	Допустимые значения - "cpu", "cuda", "cuda:0" и т.д. По умолчанию если torch видит GPU, то "cuda", иначе "cpu".

`punct` - список небуквенных символов, которые переносятся из исходного текста в транскрипцию.
	 
Входы класса `AccentuatorTranscriptor`:

```
	accentuate(sentence_list)
	transcribe(sentence_list)
```
	
В случае `accentuate` выполняется расcтановка ударений, в случае `transcribe` - транскрипция. Оба входа получают на вход список строк и возращают список строк. Строками могут быть предложения или не очень большие куски текста.

## Пример работы

Скрипт `ruslan_markup.py` расставляет ударения и порождает транскрипцию для markup-файлов акустических корпусов [ruslan](http://dataset.sova.ai/SOVA-TTS/ruslan/ruslan_dataset.tar) и [natasha](http://dataset.sova.ai/SOVA-TTS/natasha/natasha_dataset.tar).

markup-файлы этих корпусов уже содержат расстановку ударений, которая была [сделана](https://habr.com/ru/companies/ashmanov_net/articles/528296/) вручную. 

Скрипт `ruslan_markup.py` порождает для тех же файлов свою собственную расстановку ударений. Изначальная ручная разметка никак не используется при тестировании и не использовалась при обучении. Таким образом, её можно использовать для оценки точности расстановки ударений.

## Учёт контекста, предварительная обработка текста и некоторые другие особенности

### Ударения

При расстановке ударений небуквенные символы между словами никак не меняются.

Ударения расставляются с учётом контекста. Если во входном списке строк встретятся очень длинные строки (для текущей модели это больше 510 токенов), то для таких длинных строк контекст учитываться не будет. Ударения в этих строках будут ставиться только там, где это возможно без учёта контекста. 

В словах из одного слога ударение тоже ставится. В некоторых случаях это может выглядеть странно, но упрощает последующее определение транскрипции. 

### Транскрипция

При транскрипции посторонние символы фильтруются (за исключением фиксированного списка символов, который можно задавать отдельным параметром). По умолчанию не фильтруются только четыре знака пунктуации. Транскрипция определяется пословно, без учёта контекста. Для транскрипции слов используются следующие символы:

```
ʲ`ɪətrsɐnjvmapkɨʊleɫdizofʂɕbɡxːuʐæɵʉɛ
```
 
## Зависимости

Для запуска библиотеки требуются следующее:
* [Python 3](https://www.python.org/)
* [Pytorch](https://pytorch.org/)
* [Numpy](https://numpy.org/)

Кроме того, для автоматического скачивания данных требуются:
* [tqdm](https://tqdm.github.io/)
* [requests](https://pypi.org/project/requests/)

## Загрузка данных без запуска AccentuatorTranscriptor

Если запустить скрипт `download_data.py`, то данные будут скачаны. В качестве параметров запуска можно указать директорию, в которую скачивать данные.

## Обратная связь

Почта для вопросов, замечаний и предложений - `omogrus@ya.ru`.

## Лицензия

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ru)

