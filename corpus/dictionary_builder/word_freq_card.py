import math
from typing import List, Dict


class WordFrequencyCard:
    def __init__(self, word: str):
        self.word = word
        self.frequency = 0
        self.nu: float = 0
        self.rank = 0

        self.distances: List[int] = []
        self.last_index = -1

    def __str__(self):
        return f'[{self.word}]: rank={self.rank}, f={self.frequency:.3f}, nu={self.nu:.2f}'

    def __repr__(self):
        return str(self)

    @classmethod
    def calc_stat(cls, words: List[str]) -> Dict[str, 'WordFrequencyCard']:
        wrd_len = len(words)
        wrd_cards = {w: WordFrequencyCard(w) for w in set(words)}
        index = 0
        for w in words:
            card = wrd_cards[w]
            card.frequency = card.frequency + 1
            if card.last_index >= 0:
                card.distances.append(index - card.last_index)
            card.last_index = index
            index += 1
        for card in wrd_cards.values():
            card.distances.append(len(words) - card.last_index)

        freq_set = set()
        for w in wrd_cards:
            card = wrd_cards[w]
            freq_set.add(card.frequency)
            if card.frequency > 1:
                avg_dist = len(words) / card.frequency
                card.nu = sum([abs(d - avg_dist) for d in card.distances]) / len(card.distances) / avg_dist
            else:
                card.nu = 100  # this is the only entry

        freqs = list(freq_set)
        freqs.sort(reverse=True)
        freq_rank = {freqs[i]: i for i in range(len(freqs))}

        for w in wrd_cards:
            card = wrd_cards[w]
            card.rank = freq_rank[card.frequency]

        for w in wrd_cards:
            card = wrd_cards[w]
            card.nu = math.sqrt(card.nu) / (len(wrd_cards) - 0.5)
            card.frequency = card.frequency / wrd_len

        return wrd_cards