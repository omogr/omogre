
from setuptools import find_packages, setup

setup(
    name="omogre",
    version="0.1.0",
    author="omogr",
    author_email="omogrus@ya.ru",
    description="Russian accentuator and IPA transcriptor",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='Russian accentuator IPA transcriptor',
    license='CC BY-NC-SA 4.0',
    url="https://github.com/omogr/omogre",
    packages=find_packages(),
        
    install_requires=['torch>=0.4.1',
                      'numpy',
                      'requests',
                      'tqdm'],
    
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Free for non-commercial use',
        'Natural Language :: Russian',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
    ],
)
