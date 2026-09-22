from collections import defaultdict
from itertools import combinations
from typing import Iterable, Sequence

Rule = tuple[frozenset, frozenset, float, float, float]


class _RuleMiner:
    def __init__(self, min_support: float = 0.3, min_confidence: float = 0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets_: dict[frozenset, float] = {}
        self.rules_: list[Rule] = []

    def _mine(self, transactions: list[frozenset]) -> dict[frozenset, float]:
        raise NotImplementedError

    def fit(self, transactions: Iterable[Iterable]) -> "_RuleMiner":
        tx = [frozenset(t) for t in transactions]
        self.frequent_itemsets_ = self._mine(tx)
        self.rules_ = self._rules()
        return self

    def _rules(self) -> list[Rule]:
        rules = []
        for itemset, support in self.frequent_itemsets_.items():
            if len(itemset) < 2:
                continue
            for r in range(1, len(itemset)):
                for antecedent in map(frozenset, combinations(itemset, r)):
                    consequent = itemset - antecedent
                    confidence = support / self.frequent_itemsets_[antecedent]
                    if confidence >= self.min_confidence:
                        lift = confidence / self.frequent_itemsets_[consequent]
                        rules.append((antecedent, consequent, support, confidence, lift))
        return sorted(rules, key=lambda r: (-r[3], -r[2], sorted(map(str, r[0])), sorted(map(str, r[1]))))


class Apriori(_RuleMiner):
    """Level-wise candidate generation with support pruning."""

    def _mine(self, tx: list[frozenset]) -> dict[frozenset, float]:
        n = len(tx)
        counts: dict[frozenset, int] = defaultdict(int)
        for t in tx:
            for item in t:
                counts[frozenset([item])] += 1
        level = {s: c / n for s, c in counts.items() if c / n >= self.min_support}
        frequent = dict(level)
        k = 2
        while level:
            keys = sorted(level, key=lambda s: sorted(map(str, s)))
            candidates = {a | b for a, b in combinations(keys, 2) if len(a | b) == k}
            candidates = {c for c in candidates if all(frozenset(s) in level for s in combinations(c, k - 1))}
            counts = defaultdict(int)
            for t in tx:
                for c in candidates:
                    if c <= t:
                        counts[c] += 1
            level = {s: c / n for s, c in counts.items() if c / n >= self.min_support}
            frequent.update(level)
            k += 1
        return frequent


class _FPNode:
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item, parent):
        self.item, self.count, self.parent = item, 0, parent
        self.children: dict = {}
        self.link: "_FPNode | None" = None


class FPGrowth(_RuleMiner):
    """FP-tree with conditional pattern bases; produces the same itemsets as Apriori."""

    def _build_tree(self, patterns: Sequence[tuple[Sequence, int]], min_count: float):
        counts: dict = defaultdict(int)
        for items, c in patterns:
            for item in items:
                counts[item] += c
        counts = {i: c for i, c in counts.items() if c >= min_count}
        if not counts:
            return None, {}
        order = {item: (-c, str(item)) for item, c in counts.items()}
        root = _FPNode(None, None)
        header: dict = {}
        for items, c in patterns:
            node = root
            for item in sorted((i for i in items if i in counts), key=order.get):
                if item not in node.children:
                    child = _FPNode(item, node)
                    node.children[item] = child
                    if item in header:
                        child.link, header[item] = header[item], child
                    else:
                        header[item] = child
                node = node.children[item]
                node.count += c
        return root, header

    def _grow(self, patterns, min_count, suffix: frozenset, out: dict):
        _, header = self._build_tree(patterns, min_count)
        for item, node in sorted(header.items(), key=lambda kv: str(kv[0])):
            itemset = suffix | {item}
            total, conditional = 0, []
            while node is not None:
                total += node.count
                path, p = [], node.parent
                while p.item is not None:
                    path.append(p.item)
                    p = p.parent
                conditional.append((path, node.count))
                node = node.link
            out[itemset] = total
            self._grow(conditional, min_count, itemset, out)

    def _mine(self, tx: list[frozenset]) -> dict[frozenset, float]:
        n = len(tx)
        counts: dict[frozenset, int] = {}
        self._grow([(list(t), 1) for t in tx], self.min_support * n - 1e-9, frozenset(), counts)
        return {s: c / n for s, c in counts.items()}
