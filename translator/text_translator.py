import codecs
from typing import List, Tuple, Dict

from corpus.dictionary_builder.corpus_file_manager import CorpusFileManager
from corpus.dictionary_builder.corpus_reader import CorpusReader, RawTextProcessor
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from translator.synonym_finder import SynonymFinder, SynonymsFound
from translator.translation import TranslatedWord


class TextTranslator:
    def __init__(self,
                 src_lang_code: str,
                 dst_lang_code: str,
                 corpus_alt_path: str = ''):
        self.words: List[Tuple[str, int, int]] = []
        self.src_lang_code = src_lang_code
        corpus_mgr = CorpusFileManager()
        self.dict_src: LangDictionary = corpus_mgr.load(src_lang_code, corpus_alt_path)
        self.dict_dst: LangDictionary = corpus_mgr.load(dst_lang_code, corpus_alt_path)
        self.synonyms: Dict[str, SynonymsFound] = {}
        self.src_card_by_word = {w.word: w for w in self.dict_src.words}
        self.translations: List[TranslatedWord] = []

    def translate(self,
                  src_path: str,
                  src_encoding: str = 'utf-8',
                  text_processor: RawTextProcessor = RawTextProcessor):
        with codecs.open(src_path, mode='r', encoding=src_encoding) as fr:
            file_text = fr.read()

        rdr = CorpusReader()
        rdr.setup_reader(self.src_lang_code, src_encoding, text_processor)
        self.words = rdr.split_text_words(file_text)
        sf = SynonymFinder(self.dict_src.words, self.dict_dst.words)

        for wrd, start, end in self.words:
            sn = self.synonyms.get(wrd)
            if not sn:
                sn = sf.find_synonyms(True, wrd)
                self.synonyms[wrd] = sn
            card = self.src_card_by_word.get(wrd)
            translated = TranslatedWord(
                card, start, end, synonyms=sn.synonyms)
            self.translations.append(translated)


