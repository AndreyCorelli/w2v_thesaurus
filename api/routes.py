from typing import Dict

from flask import current_app as app
from flask import render_template, request

from corpus.dictionary_builder.alphabet import alphabet_by_code
from corpus.dictionary_builder.corpus_file_manager import CorpusFileManager
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from translator.synonym_finder import SynonymFinder

DICT_BY_LANG: Dict[str, LangDictionary] = {}


def get_dictionary(lang_code: str) -> LangDictionary:
    dct = DICT_BY_LANG.get(lang_code)
    if dct:
        return dct
    mgr = CorpusFileManager()
    dct = mgr.load(lang_code)
    DICT_BY_LANG[lang_code] = dct
    return dct


@app.route('/')
def home():
    """Landing page."""
    return render_template(
        'home.html',
        title="Jinja Demo Site",
        description="Smarter page templates with Flask & Jinja."
    )


@app.route('/dictionaries')
def dictionaries():
    mgr = CorpusFileManager()
    codes = mgr.get_lang_codes()
    code_title = [(c, alphabet_by_code[c].title) for c in codes]
    return render_template(
        'dictionaries.html',
        code_title=code_title
    )


@app.route('/dictionary/<language>')
def dictionary(language: str = 'en'):
    dct = get_dictionary(language)
    words = [w.word for w in dct.words]
    words.sort()
    mgr = CorpusFileManager()
    codes = mgr.get_lang_codes()
    code_title = [(c, alphabet_by_code[c].title) for c in codes]
    # words = ','.join([f"'{w}'" for w in words])
    return render_template(
        'dictionary.html',
        language=language,
        words=words,
        code_title=code_title
    )


@app.route('/translate_word', methods = ['POST'])
def translate_word():
    data = request.json
    word, src_lang, dst_lang = data['word'], data['src_language'], data['target_language']
    dc_src, dc_dst = get_dictionary(src_lang), get_dictionary(dst_lang)
    word_card = next(c for c in dc_src.words if c.word == word)
    if not word_card:
        return {'synonyms': [], 'message': 'Word was not found', 'word_card': None}

    sf = SynonymFinder(dc_src.words, dc_dst.words)
    synonyms = sf.find_synonyms(True, word)
    return {'synonyms': [s for s in synonyms.synonyms],
            'message': synonyms.message,
            'word_card': word_card.to_dict()}
