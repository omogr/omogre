# coding=utf-8

import logging
import os
import tarfile
import tempfile
import sys
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

def download_model(cache_dir, file_name='accentuator_transcriptor_tiny'):
    model_url = "https://huggingface.co/omogr/omogre/resolve/main/%s.gz?download=true"%file_name

    try:
        response = requests.head(model_url, allow_redirects=True)
        if response.status_code != 200:
            raise EnvironmentError('Cannot load model, response.status_code %d'%response.status_code)
        else:
            etag = response.headers.get("ETag")
    except EnvironmentError:
        etag = None
        
    if etag is None:
        raise EnvironmentError('Cannot load model, etag error')

    with tempfile.TemporaryFile() as temp_file: # NamedTemporaryFile f.name
        logger.info("model not found in cache, downloading to temporary file") # , model_url) #, temp_file.name)

        req = requests.get(model_url, stream=True)
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                progress.update(len(chunk))
                temp_file.write(chunk)
        progress.close()

        # we are processing the file before closing it, so flush to avoid truncation
        temp_file.flush()
        # sometimes fileobj starts at the current position, so go to the start
        temp_file.seek(0)
        etag_file_name = os.path.join(cache_dir, 'etag')
        try:
            logger.info("model archive extractall to %s", cache_dir) #, temp_file.name)
            with tarfile.open(fileobj=temp_file, mode='r:gz') as archive:
                archive.extractall(cache_dir)

            with open(etag_file_name, mode='w', encoding='utf-8') as fout:
                print(model_url, file=fout)
                print(etag, file=fout)

        except:
            if os.path.isfile(etag_file_name):
                os.remove(etag_file_name)
    return cache_dir
    

if __name__ == "__main__":
    path = "omogre_data"
    if not os.path.exists(path):
        os.mkdir(path)
    download_model(cache_dir=path)
