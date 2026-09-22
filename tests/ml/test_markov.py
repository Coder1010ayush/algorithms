import numpy as np

from ml.models.markov import HiddenMarkovModel, MarkovChain


def test_markov_chain_recovers_transition_matrix():
    truth = np.array([[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]])
    seq = MarkovChain(n_states=3, smoothing=0.0).__class__(n_states=3)
    seq.transition_, seq.initial_, seq.n_states = truth, np.ones(3) / 3, 3
    sample = seq.sample(5000, random_state=0)
    model = MarkovChain(smoothing=0.0).fit(sample)
    assert np.max(np.abs(model.transition_ - truth)) < 0.05
    assert np.allclose(model.transition_.sum(axis=1), 1.0)
    assert model.predict([0, 1]) == int(np.argmax(truth[1]))
    pi = model.stationary_distribution()
    assert np.allclose(pi @ model.transition_, pi, atol=1e-8)
    assert model.score(sample[:50]) < 0.0


def test_markov_chain_multiple_sequences_and_smoothing():
    model = MarkovChain(n_states=3).fit([np.array([0, 0, 1]), np.array([2, 2])])
    assert model.transition_.shape == (3, 3) and np.all(model.transition_ > 0)
    assert np.allclose(model.initial_.sum(), 1.0)


def _two_state_hmm() -> HiddenMarkovModel:
    hmm = HiddenMarkovModel(n_states=2, n_symbols=3, random_state=0)
    return hmm.set_params(
        initial=[0.6, 0.4],
        transition=[[0.9, 0.1], [0.1, 0.9]],
        emission=[[0.8, 0.15, 0.05], [0.05, 0.15, 0.8]],
    )


def test_viterbi_recovers_obvious_path():
    hmm = _two_state_hmm()
    obs = np.array([0, 0, 0, 0, 2, 2, 2, 2, 0, 0])
    assert hmm.viterbi(obs).tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    assert np.isfinite(hmm.score(obs))


def test_score_matches_brute_force_marginalisation():
    hmm = _two_state_hmm()
    obs = np.array([0, 2, 1, 0])
    total = 0.0
    for path in np.ndindex(*(2,) * len(obs)):
        p = hmm.initial_[path[0]] * hmm.emission_[path[0], obs[0]]
        for t in range(1, len(obs)):
            p *= hmm.transition_[path[t - 1], path[t]] * hmm.emission_[path[t], obs[t]]
        total += p
    assert np.isclose(hmm.score(obs), np.log(total))


def test_baum_welch_improves_likelihood_and_recovers_emissions():
    truth = _two_state_hmm()
    seqs = [truth.sample(200, random_state=s)[1] for s in range(5)]
    model = HiddenMarkovModel(n_states=2, n_symbols=3, n_iter=40, tol=1e-6, random_state=1).fit(seqs)
    hist = np.array(model.log_likelihood_history)
    assert np.all(np.diff(hist) >= -1e-6)
    learned = model.emission_
    perm = learned if learned[0, 0] > learned[1, 0] else learned[::-1]
    assert np.max(np.abs(perm - truth.emission_)) < 0.15
    states, obs = model.sample(20, random_state=3)
    assert states.shape == obs.shape == (20,) and obs.max() < 3
