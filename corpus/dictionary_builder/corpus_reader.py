import codecs

import regex as re
import os
from typing import List, Optional, Generator

from corpus.dictionary_builder.alphabet import Alphabet, alphabet_by_code


class CorpusReader:
    def __init__(self):
        self.corpus_folder = ''
        self.corpus_lang = ''
        self.alphabet: Optional[Alphabet] = None
        self.encoding: str = 'utf-8'
        self.processor = RawTextProcessor

        self.sentences: List[List[str]] = []
        self.sentence_length = 11

    def read(self,
             corpus_folder: str,
             corpus_lang: str,
             encoding: str = 'utf-8',
             processor: 'RawTextProcessor' = None):
        self.corpus_folder = corpus_folder
        self.corpus_lang = corpus_lang
        self.encoding = encoding
        self.alphabet = alphabet_by_code[corpus_lang]
        self.processor = processor or RawTextProcessor

        files = [f for f in os.listdir(corpus_folder)]
        for file_name in files:
            _, file_extension = os.path.splitext(file_name)
            if file_extension != '.txt':
                continue
            file_path = os.path.join(corpus_folder, file_name)
            self.read_file(file_path)

    def read_file(self, file_path: str):
        with codecs.open(file_path, mode='r', encoding=self.encoding) as fr:
            file_text = fr.read()
        sentc: List[str]
        if self.sentences:
            sentc = self.sentences[-1]
        else:
            sentc = []
            self.sentences.append(sentc)
        for word in self.processor.extract_words(file_text, self.alphabet):
            word = self.alphabet.preprocess_word(word)
            if len(sentc) < self.sentence_length:
                sentc.append(word)
            else:
                sentc = [word]
                self.sentences.append(sentc)


class RawTextProcessor:
    reg_joins = re.compile(r"'\w|-\w")
    reg_numbers = re.compile(r"\(\w\)|\w\)|\[\w\]|\w\]")
    reg_repeated = re.compile(r"(.)\1{1,}")
    min_word_weight = 0.8

    @classmethod
    def extract_words(cls, text: str, abet: Alphabet) -> Generator[str, None, None]:
        if not text:
            return
        text = text.lower()
        text = text.replace('  ', ' ')
        text = cls.reg_joins.sub('', text)
        text = cls.reg_numbers.sub('', text)
        for w in abet.reg_word.finditer(text):
            word = w.group(0)
            if cls.reg_repeated.match(word):
                continue
            if abet.is_number(word):
                continue
            yield word

    @classmethod
    def process_text(cls, text: str, abet: Alphabet) -> str:
        if not text:
            return ''
        clear_text = ''
        first_item = True
        for word in cls.extract_words(text, abet):
            if not first_item:
                clear_text += ' '
            first_item = False
            clear_text += word

        word_weight = len(clear_text) / len(text)
        if word_weight < cls.min_word_weight:
            return ''
        return clear_text