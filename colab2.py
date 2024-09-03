
# -*- coding: utf-8 -*-



"""
# @title Speech Synthesis

Stress placement and transcription can be useful for speech synthesis. This notebook contains an example of running an [XTTS](https://github.com/coqui-ai/TTS) model trained on Russian language transcription. The model was trained on the [`RUSLAN`](http://dataset.sova.ai/SOVA-TTS/ruslan/ruslan_dataset.tar) and [Common Voice](https://commonvoice.mozilla.org/ru) corpora. Model weights can be downloaded from [Hugging Face](https://huggingface.co/omogr/XTTS-ru-ipa)

Example of running an [XTTS](https://github.com/coqui-ai/TTS) model trained on Russian language transcription

Installing XTTS
"""

!pip install TTS==0.22.0

"""
Download XTTS model weights from [Hugging Face](https://huggingface.co/omogr/XTTS-ru-ipa)
Install the [transcriptor](https://github.com/omogr/omogre).
"""

!mkdir model
!git clone https://huggingface.co/omogr/XTTS-ru-ipa model
!pip install git+https://github.com/omogr/omogre.git

import os
import torch
import torchaudio

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from omogre import Transcriptor
import IPython.display as ipd

"""
Download transcriptor model weights. Initialize XTTS and transcriptor.
"""

model_dir = 'model'

def clear_gpu_cache():
    """Clear the GPU cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None

def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    """
    Load the XTTS model with the specified checkpoint, configuration, and vocabulary.

    Parameters:
    - xtts_checkpoint (str): Path to the model checkpoint.
    - xtts_config (str): Path to the model configuration file.
    - xtts_vocab (str): Path to the vocabulary file.
    """
    global XTTS_MODEL
    clear_gpu_cache()
    assert xtts_checkpoint and xtts_config and xtts_vocab, "Model paths must be provided."

    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model...")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint,
                               vocab_path=xtts_vocab, use_deepspeed=False, speaker_file_path='-')
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    if XTTS_MODEL is None:
        return False
    print(" ... model loaded!")

def run_tts(tts_text, gpt_cond_latent, speaker_embedding):
    """
    Run the XTTS model to synthesize speech from text.

    Parameters:
    - tts_text (str): Text to be synthesized.
    - gpt_cond_latent (torch.Tensor): Latent conditioning vector.
    - speaker_embedding (torch.Tensor): Speaker embedding vector.

    Returns:
    - torch.Tensor: Audio waveform tensor.
    """
    out = XTTS_MODEL.inference(
        text=tts_text,
        language='ru',
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=XTTS_MODEL.config.temperature,
        length_penalty=XTTS_MODEL.config.length_penalty,
        repetition_penalty=XTTS_MODEL.config.repetition_penalty,
        top_k=XTTS_MODEL.config.top_k,
        top_p=XTTS_MODEL.config.top_p,
    )

    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
    return out["wav"]

class XttsInference:
    def __init__(self, transcriptor_data_path='omogre_data',
                 xtts_model_path='model'):
        """
        Initialize the transcriptor and load the XTTS model.

        Parameters:
        - transcriptor_data_path (str): Path where transcriptor data will be downloaded.
        - xtts_model_path (str): Path to the XTTS model directory.
        """
        clear_gpu_cache()
        self.transcriptor = Transcriptor(data_path=transcriptor_data_path)
        xtts_checkpoint = os.path.join(xtts_model_path, "model.pth")
        xtts_config = os.path.join(xtts_model_path, "config.json")
        xtts_vocab = os.path.join(xtts_model_path, "vocab.json")
        load_model(xtts_checkpoint, xtts_config, xtts_vocab)

        reference_audio = os.path.join(xtts_model_path, "reference_audio.wav")
        if not reference_audio:
            print("empty reference_audio")
            return False

        self.gpt_cond_latent, self.speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path=reference_audio,
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
            max_ref_length=XTTS_MODEL.config.max_ref_len,
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs
        )

    def __call__(self, src_text):
        """
        Generate synthesized speech from input text.

        Parameters:
        - src_text (str): Source text to synthesize.

        Returns:
        - tuple: Transcribed text and audio waveform tensor.
        """
        tts_text = ' '.join(self.transcriptor([src_text]))
        audio = run_tts(tts_text, self.gpt_cond_latent, self.speaker_embedding)
        return tts_text, audio

xtts_inference = XttsInference()

"""Example of generating audio for a single phrase"""

# @title Example of generating audio for a single phrase

src_text = 'МИД Турции официально заявил, что Турция заинтересована во вступлении в БРИКС.' # @param {type:"string"}

#src_text = 'МИД Турции официально заявил, что Турция заинтересована во вступлении в БРИКС.'
print('src_text', src_text)

tts_text, audio = xtts_inference(src_text)
print('Speech generated!', tts_text)

# Save the result
output_file = 'audio.wav'
torchaudio.save(output_file, audio, sample_rate=24000)
ipd.display(ipd.Audio(audio.to('cpu').detach(), rate=24000))
```

### Key Improvements:
- **Translation of Comments**: All comments have been translated from Russian to English.
- **Function Documentation**: Added docstrings to functions to provide clear documentation of parameters and return values.
- **Code Clarity**: Added assertions and error messages to improve code robustness.
- **Code Formatting**: Improved code formatting for better readability, including consistent use of spaces and line breaks. 

If you have any specific questions or need further adjustments, feel free to ask!