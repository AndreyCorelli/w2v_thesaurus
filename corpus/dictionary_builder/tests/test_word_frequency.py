from unittest import TestCase

from corpus.dictionary_builder.word_freq_card import WordFrequencyCard


class TestFindPdfFragment(TestCase):
    def test_semi_uniform(self):
        cards = WordFrequencyCard.calc_stat(list('...a..+++...++.....+..aa'))
        self.assertEqual(3, len(cards))
        self.assertEqual(1, cards['+'].rank)
        self.assertEqual(0, cards['.'].rank)
        self.assertLess(cards['+'].nu, cards['a'].nu)

    def test_uniform(self):
        cards = WordFrequencyCard.calc_stat(list('++..++..'*50))
        self.assertLess(cards['.'].nu, 0.5)
        self.assertLess(cards['+'].nu, 0.5)