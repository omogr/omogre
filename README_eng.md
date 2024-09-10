


# Omogre

## Russian Accentuator and IPA Transcriptor

A library for [`Python 3`](https://www.python.org/). Automatic stress placement and [IPA transcription](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) for the Russian language.

## Dependencies

Installing the library will also install [`Pytorch`](https://pytorch.org/) and [`Numpy`](https://numpy.org/). Additionally, for model downloading, [`tqdm`](https://tqdm.github.io/) and [`requests`](https://pypi.org/project/requests/) will be installed.

## Installation

### Using GIT

```bash
pip install git+https://github.com/omogr/omogre.git
```

### Using pip

Download the code from [GitHub](https://github.com/omogr/omogre). In the directory containing [`setup.py`](https://github.com/omogr/omogre/blob/main/setup.py), run:

```bash
pip install -e .
```

### Manually

Download the code from [GitHub](https://github.com/omogr/omogre). Install [`Pytorch`](https://pytorch.org/), [`Numpy`](https://numpy.org/), [`tqdm`](https://tqdm.github.io/), and [`requests`](https://pypi.org/project/requests/). Run [`test.py`](https://github.com/omogr/omogre/blob/main/test.py).

## Model downloading

By default, data for models will be downloaded on the first run of the library. The script [`download_data.py`](https://github.com/omogr/omogre/blob/main/download_data.py) can also be used to download this data.

You can specify a path where the model data should be stored. If data already exists in this directory, it won't be downloaded again.

## Example

Script [`test.py`](https://github.com/omogr/omogre/blob/main/test.py).

```python
from omogre import Accentuator, Transcriptor

# Data will be downloaded to the 'omogre_data' directory
transcriptor = Transcriptor(data_path='omogre_data')
accentuator = Accentuator(data_path='omogre_data')

sentence_list = ['стены замка']

print('transcriptor', transcriptor(sentence_list))
print('accentuator', accentuator(sentence_list))

# Alternative call methods, differing only in notation
print('transcriptor.transcribe', transcriptor.transcribe(sentence_list))
print('accentuator.accentuate', accentuator.accentuate(sentence_list))

print('transcriptor.accentuate', transcriptor.accentuate(sentence_list))
```

## Class Parameters

### Transcriptor

All initialization parameters for the class are optional.

```python
class Transcriptor(data_path: str = None,
                   download: bool = True,
                   device_name: str = None,
                   punct: str = '.,!?')
```

- `data_path`: Directory where the model should be located.
- `device_name`: Parameter defining GPU usage. Corresponds to the initialization parameter of [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device). Valid values include `"cpu"`, `"cuda"`, `"cuda:0"`, etc. Defaults to `"cuda"` if GPU is available, otherwise `"cpu"`.
- `punct`: List of non-letter characters to be carried over from the source text to the transcription. Default is `'.,!?'`.
- `download`: Whether to download the model from the internet if not found in `data_path`. Default is `True`.

Class methods:

```python
accentuate(sentence_list: list) -> list
transcribe(sentence_list: list) -> list
```

`accentuate` places stresses, `transcribe` performs transcription. Both inputs take a list of strings and return a list of strings.

### Accentuator

The `Accentuator` class for stress placement is identical to the `Transcriptor` in terms of stress functionality, except it doesn't load transcription data, reducing initialization time and memory usage.

All initialization parameters are optional, with the same meanings as for `Transcriptor`.

```python
class Accentuator(data_path: str = None,
                  download: bool = True,
                  device_name: str = None)
```

- `data_path`: Directory where the model should be located.
- `device_name`: Parameter for GPU usage. See above for details.
- `download`: Whether to download the model if not found. Default is `True`.

Class method:

```python
accentuate(sentence_list: list) -> list
```

## Usage Examples

### markup files for acoustic corpora

The script [`ruslan_markup.py`](https://github.com/omogr/omogre/blob/main/ruslan_markup.py) places stresses and generates transcriptions for markup files of the acoustic corpora [`RUSLAN`](https://ruslan-corpus.github.io/) ([`RUSLAN with manually accentuated markup`](http://dataset.sova.ai/SOVA-TTS/ruslan/ruslan_dataset.tar)) and [`NATASHA`](http://dataset.sova.ai/SOVA-TTS/natasha/natasha_dataset.tar).

These markup files already contain [manually placed stresses](https://habr.com/ru/companies/ashmanov_net/articles/528296/).

The script [`ruslan_markup.py`](https://github.com/omogr/omogre/blob/main/ruslan_markup.py) generates its own stress placement for these files, allowing for an evaluation of the accuracy of stress placement.

### Speech Synthesis

Accentuation and transcription can be useful for speech synthesis. The [`Colab` notebook](https://github.com/omogr/omogre/blob/main/XTTS_ru_ipa.ipynb) contains an example of running the [`XTTS`](https://github.com/coqui-ai/TTS) model trained on transcription for the Russian language. The model was trained on the [`RUSLAN`](https://ruslan-corpus.github.io/) and [`Common Voice`](https://commonvoice.mozilla.org/ru) datasets.
The model weights can be downloaded from [`Hugging Face`](https://huggingface.co/omogr/XTTS-ru-ipa).

### Extraction of transcription from audio files

Accentuation and transcription can be useful for acustic corpora analysis. The [`Colab` notebook](https://github.com/omogr/omogre/blob/main/Wav2vec2_ru_ipa.ipynb) contains an example of running the wav2vec2-lv-60-espeak-cv-ft model finetuned with transcription of [`RUSLAN`](https://ruslan-corpus.github.io/) and [`Common Voice`](https://commonvoice.mozilla.org/ru).

## Context Awareness and Other Features

### Stresses

Stresses are placed considering context. If very long strings are encountered (for the current model, more than 510 tokens), context won't be considered for these. Stresses in these strings will be placed only where possible without context.

Stresses are also placed in one-syllable words, which might look unusual but simplifies subsequent transcription determination.

### Transcription

During transcription generation, extraneous characters are filtered out. Non-letter characters that are not filtered can be specified by a parameter. By default, four punctuation marks (`.,!?`) are not filtered. Transcription is determined word by word, without context. The following symbols are used for transcription:

```
ʲ`ɪətrsɐnjvmapkɨʊleɫdizofʂɕbɡxːuʐæɵʉɛ
```

## Feedback
Email for questions, comments and suggestions - `omogrus@ya.ru`.

## License
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

(translated by grok-2-2024-08-13)
