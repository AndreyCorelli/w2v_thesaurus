import os
from typing import Optional, Dict, Any, List
import msgpack

from corpus.dictionary_builder.alphabet import alphabet_by_code
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from corpus.models import WordCard


class CorpusFileManager:
    def __init__(self):
        self.default_path = os.path.dirname(os.path.realpath(__file__))
        self.default_path = os.path.join(self.default_path, '..', '..', 'data', 'dictionaries')

    def get_lang_codes(self, target_folder: str = '') -> List[str]:
        codes = []
        target_folder = target_folder or self.default_path
        for file in os.listdir(target_folder):
            if os.path.isfile(os.path.join(target_folder, file)):
                lang_code = os.path.splitext(file)[0]
                if lang_code in alphabet_by_code:
                    codes.append(lang_code)
        return codes

    def save(self, ld: LangDictionary, target_folder: Optional[str] = None) -> None:
        target_path = self.get_file_path(target_folder, ld.lang_code)
        cards = {'data': [c.to_dict() for c in ld.words]}
        bt_data: bytes = msgpack.packb(cards, use_bin_type=True, use_single_float=True)
        with open(target_path, mode='wb') as f:
            f.write(bt_data)

    def load(self, lang_code: str, target_folder: Optional[str] = None) -> LangDictionary:
        target_path = self.get_file_path(target_folder, lang_code)
        with open(target_path, 'rb') as pages_f:
            pdfbox_res: Dict[str, Any] = msgpack.unpack(pages_f, raw=False)
            word_dicts = pdfbox_res['data']
            word_cards = [WordCard(**d) for d in word_dicts]
        return LangDictionary(lang_code, word_cards)

    def get_file_path(self, target_folder: str, lang_code: str) -> str:
        target_folder = target_folder or self.default_path
        return os.path.join(target_folder, f'{lang_code}.msgpack')
