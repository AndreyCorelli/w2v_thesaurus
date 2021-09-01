from unittest import TestCase

from corpus.dictionary_builder.sorted_items import SortedItems


class TestSortedItems(TestCase):
    def test_range(self):
        items = [(1, 0.1), (2, 10), (3, 0.2), (4, 0.2), (5, 0.4), (6, 11)]
        s = SortedItems(4)
        for id, val in items:
            s.update(val, id)

        self.assertEqual(4, len(s.items))
        self.assertEqual(0.4, max([it[1] for it in s.items]))
