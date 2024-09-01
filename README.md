# Omogre

## Russian accentuator and IPA transcriptor.

Библиотека для [Python 3](https://www.python.org/). Автоматическая расстановка ударений и [IPA транскрипция](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D0%B6%D0%B4%D1%83%D0%BD%D0%B0%D1%80%D0%BE%D0%B4%D0%BD%D1%8B%D0%B9_%D1%84%D0%BE%D0%BD%D0%B5%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B9_%D0%B0%D0%BB%D1%84%D0%B0%D0%B2%D0%B8%D1%82) для русского языка.


## Зависимости

Установка библиотеки повлечет за собой установку [Pytorch](https://pytorch.org/) и [Numpy](https://numpy.org/). Кроме того, для скачивания моделей  установятся [tqdm](https://tqdm.github.io/) и [requests](https://pypi.org/project/requests/).


## Установка

### С помощью GIT

```
pip install git+https://github.com/omogr/omogre.git
```

### При помощи pip

Скачать код с [гитхаба](https://github.com/omogr/omogre). В директории, где находится setup.py, выполнить

```
pip install -e .
```

### Вручную

Скачать код с [гитхаба](https://github.com/omogr/omogre). Установить [Pytorch](https://pytorch.org/), [Numpy](https://numpy.org/), [tqdm](https://tqdm.github.io/) и [requests](https://pypi.org/project/requests/). Запустить test.py.


## Скачивание данных

По умолчанию, если не указывать путь, то при первом запуске библиотеки данные для моделей скачиваются автоматически в директорию по умолчанию.

При желании можно указывать путь, данные будут скачаны в указанную директорию. Если в этой директории уже есть данные, то повторного скачивания данных не будет.

Скрипт `download_data.py` также позволяет загружать данные. В качестве параметра запуска этого скрипта тоже можно указать директорию, в которую надо скачать данные.


## Пример запуска

Скрипт test.py.

```
from omogre import Accentuator, Transcriptor

# данные будут скачаны в директорию 'omogre_data'
transcriptor = Transcriptor(data_path='omogre_data')

accentuator = Accentuator(data_path='omogre_data')

sentence_list = ['стены замка']

print('transcriptor', transcriptor(sentence_list))
print('accentuator', accentuator(sentence_list))

# длугие способы вызовов, отличаются только формой записи
print('transcriptor.transcribe', transcriptor.transcribe(sentence_list))
print('accentuator.accentuate', accentuator.accentuate(sentence_list))

print('transcriptor.accentuate', transcriptor.accentuate(sentence_list))
```


## Параметры классов

### Transcriptor

Все параметры инициализации класса не являются обязательными. 

```
class Transcriptor(data_path: str = None, download: bool = True, device_name: str = None, punct: str = '.,!?')
```

`data_path` - директория, в которой должна находиться модель.

`device_name` - параметр, определяющий использование GPU. Соответствует параметру инициализации класса [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device).	Допустимые значения - "cpu", "cuda", "cuda:0" и т.д. По умолчанию если torch видит GPU, то "cuda", иначе "cpu".

`punct` - список небуквенных символов, которые переносятся из исходного текста в транскрипцию. По умолчанию '.,!?'.

`download` - следует ли загружать модель из интернета, если она не найдена в директории `data_path`. По умолчанию True.

	 
Входы класса `Transcriptor`:

```
	accentuate(sentence_list: list) -> list
	transcribe(sentence_list: list) -> list
```
	
В случае `accentuate` выполняется расcтановка ударений, в случае `transcribe` - транскрипция. Оба входа получают на вход список строк и возращают список строк. Строками могут быть предложения или не очень большие куски текста.

### Accentuator

Расстановка ударений классом Accentuator ничем не отличается от расстановки ударений классом Transcriptor. Разница только в том, что Accentuator не загружает данные для транскрипции. Это позволяет уменьшить время инициализации класса и расход оперативной памяти.

Все параметры инициализации класса не являются обязательными. Смысл параметров инициализации такой же, как у класса Transcriptor.

```
class Accentuator(data_path: str = None, download: bool = True, device_name: str = None)
```

`data_path` - директория, в которой должна находиться модель.

`device_name` - параметр, определяющий использование GPU. Соответствует параметру инициализации класса [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device).	Допустимые значения - "cpu", "cuda", "cuda:0" и т.д. По умолчанию если torch видит GPU, то "cuda", иначе "cpu".

`download` - следует ли загружать модель из интернета, если она не найдена в директории `data_path`. По умолчанию True.

Входы класса `Accentuator`:

```
	accentuate(sentence_list: list) -> list
```


## Пример работы

Скрипт `ruslan_markup.py` расставляет ударения и порождает транскрипцию для markup-файлов акустических корпусов [ruslan](http://dataset.sova.ai/SOVA-TTS/ruslan/ruslan_dataset.tar) и [natasha](http://dataset.sova.ai/SOVA-TTS/natasha/natasha_dataset.tar).

markup-файлы этих корпусов уже содержат расстановку ударений, которая [была сделана](https://habr.com/ru/companies/ashmanov_net/articles/528296/) вручную. 

Скрипт `ruslan_markup.py` порождает для тех же файлов свою собственную расстановку ударений. Изначальная ручная разметка никак не используется при тестировании и не использовалась при обучении. Таким образом, её можно использовать для оценки точности расстановки ударений.


## Учёт контекста и некоторые другие особенности

### Ударения

Ударения расставляются с учётом контекста. Если во входном списке строк встретятся очень длинные строки (для текущей модели это больше 510 токенов), то для таких длинных строк контекст учитываться не будет. Ударения в этих строках будут ставиться только там, где это возможно без учёта контекста. 

В словах из одного слога ударение тоже ставится. В некоторых случаях это может выглядеть странно, но упрощает последующее определение транскрипции. 

### Транскрипция

При порождении транскрипции посторонние символы фильтруются. Список небуквенных символов, которые не фильтруются можно задавать отдельным параметром. По умолчанию не фильтруются четыре знака пунктуации  (`.,!?`). Транскрипция определяется пословно, без учёта контекста. Для транскрипции слов используются следующие символы:

```
ʲ`ɪətrsɐnjvmapkɨʊleɫdizofʂɕbɡxːuʐæɵʉɛ
```


## Обратная связь

Почта для вопросов, замечаний и предложений - `omogrus@ya.ru`.


## Лицензия

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ru)

