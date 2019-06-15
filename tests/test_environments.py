

def test_environment_normalizer():
    from src.utils import random_hypercube_samples
    from src.models import Normalizer
    from src.environments import Step, EnvironmentNormalizer

    f = Step()
    X = random_hypercube_samples(100, f.bounds)
    Y = f(X)
    X_norm = Normalizer(X)
    Y_norm = Normalizer(Y)
    normalized_f = EnvironmentNormalizer(f, X_norm, Y_norm)
    f.plot()
    normalized_f.plot()
