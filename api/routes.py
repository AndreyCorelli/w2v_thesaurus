from flask import current_app as app
from flask import render_template

from corpus.dictionary_builder.alphabet import alphabet_by_code
from corpus.dictionary_builder.corpus_file_manager import CorpusFileManager


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
    """Landing page."""
    return render_template(
        'dictionary.html',
        language=language
    )