

# Omogre

## Russian accentuation and IPA transcription library.

Library for [Python 3](https://www.python.org/). Automatic stress placement and [IPA transcription](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) for the Russian language.


## Dependencies

Installing the library will entail installing [Pytorch](https://pytorch.org/) and [Numpy](https://numpy.org/). In addition, [tqdm](https://tqdm.github.io/) and [requests](https://pypi.org/project/requests/) will be installed to download the models.


## Installation

### Using GIT

```bash
pip install git+https://github.com/omogr/omogre.git
```

### Using pip

Download the code from [github](https://github.com/omogr/omogre). In the directory containing the `setup.py` file, run

```bash
pip install -e .
```

### Manually

Download the code from [github](https://github.com/omogr/omogre). Install [Pytorch](https://pytorch.org/), [Numpy](https://numpy.org/), [tqdm](https://tqdm.github.io/) and [requests](https://pypi.org/project/requests/). Run test.py.


## Data Download

By default, if no path is specified, the model data will be downloaded the first time the library is run. The `download_data.py` script also allows you to download this data.

You can optionally specify the path where the model data should be located. If this directory already contains data, it will not be downloaded again.


## Example Run

The test.py script.

```python
from omogre import Accentuator, Transcriptor

# the data will be downloaded to the 'omogre_data' directory
transcriptor = Transcriptor(data_path='omogre_data')
accentuator = Accentuator(data_path='omogre_data')

sentence_list = ['стены замка']

print('transcriptor', transcriptor(sentence_list))
print('accentuator', accentuator(sentence_list))

# other ways to call, differ only in the form of writing
print('transcriptor.transcribe', transcriptor.transcribe(sentence_list))
print('accentuator.accentuate', accentuator.accentuate(sentence_list))

print('transcriptor.accentuate', transcriptor.accentuate(sentence_list))
```


## Class Parameters

### Transcriptor

All class initialization parameters are optional.

```python
class Transcriptor(data_path: str = None,
                   download: bool = True,
                   device_name: str = None,
                   punct: str = '.,!?')
```

`data_path` - the directory where the model should be located.

`device_name` - the parameter that determines the use of GPU. Corresponds to the [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) class initialization parameter. Allowed values are `"cpu"`, `"cuda"`, `"cuda:0"`, etc. By default, if torch sees GPU, then `"cuda"`, otherwise `"cpu"`.

`punct` - a list of non-letter characters that are transferred from the source text to the transcription. By default `'.,!?'`.

`download` - whether to download the model from the Internet if it is not found in the `data_path` directory. By default `True`.

         
Inputs of the `Transcriptor` class:

```python
        accentuate(sentence_list: list) -> list
        transcribe(sentence_list: list) -> list
```
        
In the case of `accentuate`, stress is placed, in the case of `transcribe`, transcription is performed. Both inputs receive a list of strings as input and return a list of strings. The strings can be sentences or not very large chunks of text.

### Accentuator

Stress placement by the Accentuator class is no different from stress placement by the Transcriptor class. The only difference is that Accentuator does not load data for transcription. This allows you to reduce the class initialization time and RAM usage.

All class initialization parameters are optional. The meaning of the initialization parameters is the same as for the Transcriptor class.

```python
class Accentuator(data_path: str = None,
                  download: bool = True,
                  device_name: str = None)
```

`data_path` - the directory where the model should be located.

`device_name` - the parameter that determines the use of GPU. Corresponds to the [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) class initialization parameter. Allowed values are `"cpu"`, `"cuda"`, `"cuda:0"`, etc. By default, if torch sees GPU, then `"cuda"`, otherwise `"cpu"`.

`download` - whether to download the model from the Internet if it is not found in the `data_path` directory. By default `True`.

Inputs of the `Accentuator` class:

```python
        accentuate(sentence_list: list) -> list
```


## Example of Work

The `ruslan_markup.py` script places stress and generates transcription for markup files of the acoustic corpora [ruslan](http://dataset.sova.ai/SOVA-TTS/ruslan/ruslan_dataset.tar) and [natasha](http://dataset.sova.ai/SOVA-TTS/natasha/natasha_dataset.tar).

The markup files of these corpora already contain stress placement, which [was done](https://habr.com/ru/companies/ashmanov_net/articles/528296/) manually.

The `ruslan_markup.py` script generates its own stress placement for the same files. The initial manual markup is not used in testing and was not used in training. Thus, it can be used to assess the accuracy of stress placement.


## Context Awareness and Other Features

### Stress Placement

Stress is placed taking context into account. If very long strings are encountered in the input list of strings (for the current model, this is more than 510 tokens), then context will not be taken into account for such long strings. Stress will be placed in these strings only where possible without taking context into account.

Stress is also placed in single-syllable words. In some cases, this may look strange, but it simplifies the subsequent definition of transcription.

### Transcription

Extraneous symbols are filtered out when generating transcription. The list of non-letter characters that are not filtered can be specified by a separate parameter. By default, four punctuation marks (`.,!?`) are not filtered. Transcription is determined word by word, without taking context into account. The following symbols are used for the transcription of words:

```
ʲ`ɪətrsɐnjvmapkɨʊleɫdizofʂɕbɡxːuʐæɵʉɛ
```

## Feedback
Email for questions, comments and suggestions - `omogrus@ya.ru`.

## License
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

(translated by gemini-1.5-flash-api-0514)
