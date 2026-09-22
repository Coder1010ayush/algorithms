import numpy as np

from ml.models.association import Apriori, FPGrowth
from ml.models.topic import LatentDirichletAllocation


def test_lda_recovers_planted_topics(rng):
    n_topics, words_per_topic = 3, 6
    vocab = n_topics * words_per_topic
    topics = np.zeros((n_topics, vocab))
    for k in range(n_topics):
        topics[k, k * words_per_topic : (k + 1) * words_per_topic] = 1.0 / words_per_topic
    docs = []
    for _ in range(60):
        theta = rng.dirichlet(np.full(n_topics, 0.1))
        z = rng.choice(n_topics, size=40, p=theta)
        counts = np.zeros(vocab)
        for k in z:
            counts[rng.choice(vocab, p=topics[k])] += 1
        docs.append(counts)
    X = np.array(docs)

    lda = LatentDirichletAllocation(n_topics=3, alpha=0.1, beta=0.01, n_iter=40, random_state=0).fit(X)
    assert lda.components_.shape == (3, vocab)
    assert np.allclose(lda.components_.sum(axis=1), 1.0)
    planted = {frozenset(range(k * words_per_topic, (k + 1) * words_per_topic)) for k in range(n_topics)}
    learned = {frozenset(row) for row in lda.top_words(words_per_topic)}
    assert learned == planted
    theta = lda.transform(X[:5])
    assert theta.shape == (5, 3) and np.allclose(theta.sum(axis=1), 1.0)


TRANSACTIONS = [
    {"bread", "milk"},
    {"bread", "diaper", "beer", "eggs"},
    {"milk", "diaper", "beer", "cola"},
    {"bread", "milk", "diaper", "beer"},
    {"bread", "milk", "diaper", "cola"},
]


def test_apriori_supports_and_rules():
    model = Apriori(min_support=0.6, min_confidence=0.6).fit(TRANSACTIONS)
    fs = model.frequent_itemsets_
    assert fs[frozenset({"bread"})] == 0.8
    assert fs[frozenset({"milk"})] == 0.8
    assert fs[frozenset({"diaper"})] == 0.8
    assert fs[frozenset({"beer"})] == 0.6
    assert fs[frozenset({"bread", "milk"})] == 0.6
    assert fs[frozenset({"diaper", "beer"})] == 0.6
    assert frozenset({"cola"}) not in fs
    assert all(len(s) <= 2 for s in fs)
    rules = {(a, c): (conf, lift) for a, c, _, conf, lift in model.rules_}
    conf, lift = rules[(frozenset({"beer"}), frozenset({"diaper"}))]
    assert np.isclose(conf, 1.0) and np.isclose(lift, 1.25)
    conf, _ = rules[(frozenset({"diaper"}), frozenset({"beer"}))]
    assert np.isclose(conf, 0.75)


def test_fpgrowth_matches_apriori():
    for min_support in (0.4, 0.6):
        a = Apriori(min_support=min_support, min_confidence=0.5).fit(TRANSACTIONS)
        f = FPGrowth(min_support=min_support, min_confidence=0.5).fit(TRANSACTIONS)
        assert a.frequent_itemsets_.keys() == f.frequent_itemsets_.keys()
        for k, v in a.frequent_itemsets_.items():
            assert np.isclose(v, f.frequent_itemsets_[k])
        assert [(r[0], r[1]) for r in a.rules_] == [(r[0], r[1]) for r in f.rules_]
