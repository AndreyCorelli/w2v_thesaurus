from unittest import TestCase

from corpus.dictionary_builder.word_freq_card import WordFrequencyCard


class TestFindPdfFragment(TestCase):
    def test_semi_uniform(self):
        cards = WordFrequencyCard.calc_stat(list('.....+++...++.....+..'))
        self.assertEqual(2, len(cards))
        self.assertEqual(1, cards['+'].rank)
        self.assertEqual(0, cards['.'].rank)
        self.assertLess(cards['.'].nu, cards['+'].nu)

    def test_uniform(self):
        cards = WordFrequencyCard.calc_stat(list('++..++..'*50))
        self.assertLess(cards['.'].nu, 1)
        self.assertLess(cards['+'].nu, 1)