

def test_environment_normalizer():
    from src.utils import random_hypercube_samples
    from src.models import Normalizer
    from src.environments.helpers import EnvironmentNormalizer
    from src.environments.discontinous import Step

    f = Step()
    X = random_hypercube_samples(100, f.bounds)
    Y = f(X)
    X_norm = Normalizer(X)
    Y_norm = Normalizer(Y)
    normalized_f = EnvironmentNormalizer(f, X_norm, Y_norm)
    f.plot()
    normalized_f.plot()


def test_active_subspace():
    from notebook_header import *
    # Test that it is indeed active subspace of dim 1:
    # Alpha = TwoKinkDEmbedding.generate_alpha(D=10)
    Alpha = [
        [0.78695576],
        [0.70777112],
        [0.34515641],
        [0.20288506],
        [0.52388727],
        [0.2025096 ],
        [0.31752746],
        [0.24497726],
        [0.89249818],
        [0.64264009]]
    f = TwoKinkDEmbedding(Alpha=Alpha)
    #f = KinkDCircularEmbedding(D=10)
    X = random_hypercube_samples(1000, f.bounds)
    G = f.derivative(X)
    model = ActiveSubspace()
    model.fit(X, f(X), G)
    model.W.shape[-1]
    assert model.W.shape[1] == 1, "Subspace Dimensionality should be 1 since it is assumed by the model."
