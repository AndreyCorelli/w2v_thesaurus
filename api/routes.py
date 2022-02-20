import datetime
from typing import Dict
import regex as re

from flask import current_app as app
from flask import render_template, request

from api.optimization.optimization_track import TRACK_SCORE, OptimizationRecord
from corpus.dictionary_builder.alphabet import alphabet_by_code
from corpus.dictionary_builder.corpus_file_manager import CorpusFileManager
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from translator.metaparams import METAPARAMS
from translator.synonym_finder import SynonymFinder

DICT_BY_LANG: Dict[str, LangDictionary] = {}
REG_NUMBER = re.compile(r'[+-]?([0-9]*[.])?[0-9]+')


def get_dictionary(lang_code: str) -> LangDictionary:
    dct = DICT_BY_LANG.get(lang_code)
    if dct:
        return dct
    mgr = CorpusFileManager()
    dct = mgr.load(lang_code)
    DICT_BY_LANG[lang_code] = dct
    return dct


#@app.route('/css/<path:path>')
#def send_js(path):
#    return send_from_directory('css', path)


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


@app.route('/translate_word', methods=['POST'])
def translate_word():
    data = request.json
    word, src_lang, dst_lang = data['word'], data['src_language'], data['target_language']
    dc_src, dc_dst = get_dictionary(src_lang), get_dictionary(dst_lang)
    word_card = next(c for c in dc_src.words if c.word == word)
    if not word_card:
        return {'synonyms': [], 'message': 'Word was not found', 'word_card': None}

    sf = SynonymFinder(dc_src, dc_dst)
    synonyms = sf.find_synonyms(True, word)
    return {'synonyms': [s for s in synonyms.synonyms],
            'message': synonyms.message,
            'word_card': word_card.to_dict()}


@app.route('/optimization_track')
def optimization_track():
    tracks = TRACK_SCORE.get_tracks()
    return render_template(
        'optimization_track.html',
        tracks=tracks,
        weights=METAPARAMS.word_vector_weights
    )


@app.route('/optimization_apply', methods=['POST'])
def optimization_apply():
    # apply & save new meta-params for similar words search
    data = request.json
    word_weight_str = data['word_weights']
    word_weights = [float(m.group()) for m in REG_NUMBER.finditer(word_weight_str)]
    METAPARAMS.word_vector_weights = word_weights
    METAPARAMS.save()
    return {'status': 'ok'}


@app.route('/optimization_evaluate')
def optimization_evaluate():
    sample_set = TRACK_SCORE.prepare_sample_set()
    return render_template(
        'optimization_evaluate.html',
        sample_set=sample_set
    )


@app.route('/optimization_eval_save', methods=['POST'])
def optimization_eval_save():
    # save evaluated results
    data = request.json
    score = data['score']
    track = OptimizationRecord()
    track.score = score
    track.coeffs = METAPARAMS.word_vector_weights
    track.record_date = datetime.datetime.now()
    TRACK_SCORE.add_record(track)
    return {'status': 'ok'}
