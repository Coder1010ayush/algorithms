import numpy as np

from ml.base import BaseModel


class LatentDirichletAllocation(BaseModel):
    """LDA fitted by collapsed Gibbs sampling on a document-term count matrix."""

    def __init__(
        self,
        n_topics: int = 5,
        alpha: float = 0.1,
        beta: float = 0.01,
        n_iter: int = 100,
        random_state: int | None = None,
    ):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.rng = np.random.default_rng(random_state)
        self.components_: np.ndarray | None = None
        self.doc_topic_: np.ndarray | None = None

    def _tokens(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        docs, words = np.nonzero(X)
        counts = X[docs, words].astype(int)
        return np.repeat(docs, counts), np.repeat(words, counts)

    def _gibbs(self, X: np.ndarray, topic_word: np.ndarray, update_topic_word: bool) -> np.ndarray:
        n_docs, n_words = X.shape
        docs, words = self._tokens(X)
        z = self.rng.integers(self.n_topics, size=len(docs))
        doc_topic = np.zeros((n_docs, self.n_topics))
        np.add.at(doc_topic, (docs, z), 1)
        if update_topic_word:
            np.add.at(topic_word, (z, words), 1)
        topic_total = topic_word.sum(axis=1)
        for _ in range(self.n_iter):
            for i in range(len(docs)):
                d, w, k = docs[i], words[i], z[i]
                doc_topic[d, k] -= 1
                if update_topic_word:
                    topic_word[k, w] -= 1
                    topic_total[k] -= 1
                p = (doc_topic[d] + self.alpha) * (topic_word[:, w] + self.beta) / (topic_total + n_words * self.beta)
                k = self.rng.choice(self.n_topics, p=p / p.sum())
                z[i] = k
                doc_topic[d, k] += 1
                if update_topic_word:
                    topic_word[k, w] += 1
                    topic_total[k] += 1
        return doc_topic

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LatentDirichletAllocation":
        X = np.asarray(X)
        topic_word = np.zeros((self.n_topics, X.shape[1]))
        doc_topic = self._gibbs(X, topic_word, update_topic_word=True)
        self.components_ = (topic_word + self.beta) / (topic_word + self.beta).sum(axis=1, keepdims=True)
        self.doc_topic_ = (doc_topic + self.alpha) / (doc_topic + self.alpha).sum(axis=1, keepdims=True)
        self._topic_word_counts = topic_word
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        doc_topic = self._gibbs(np.asarray(X), self._topic_word_counts.copy(), update_topic_word=False)
        return (doc_topic + self.alpha) / (doc_topic + self.alpha).sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.transform(X), axis=1)

    def top_words(self, n: int = 10) -> np.ndarray:
        return np.argsort(-self.components_, axis=1)[:, :n]
