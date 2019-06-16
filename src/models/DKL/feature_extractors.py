import torch


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, D=None, layers=(50, 25, 10, 2), normalize_output=True, relu_output=False):
        super().__init__()

        assert D is not None, "Dimensionality D has to be specified."
        assert len(layers) >= 1, "You need to specify at least an output layer size."
        layers = (D,) + tuple(layers)

        for i in range(0, len(layers) - 2):
            in_ = layers[i]
            out = layers[i + 1]
            self.add_module('linear{}'.format(i), torch.nn.Linear(in_, out))
            self.add_module('relu{}'.format(i), torch.nn.ReLU())

        self.output_dim = layers[-1]
        self.add_module('linear{}'.format(i+1), torch.nn.Linear(layers[-2], self.output_dim))
        if relu_output:
            self.add_module('relu{}'.format(i+1), torch.nn.ReLU())
        if normalize_output:
            self.add_module('normalization', torch.nn.BatchNorm1d(self.output_dim, affine=False, momentum=1))


class RFFEmbedding(torch.nn.Module):
    def __init__(self, D, M):
        super().__init__()
        self.D = D
        self.M = M
        self.output_dim = M

        assert M % 2 == 0, "M has to be even since there is a feature for both sin and cos."

        # register lengthscale as parameter
        self.lengthscale = torch.nn.Parameter(torch.Tensor(1))
        self.lengthscale.data.fill_(0.61)

        # sample self.unscaled_W shape: (M, D)
        self.unscaled_W = torch.randn(self.M // 2, self.D)

    def forward(self, X): # NxD -> MxD
        W = self.unscaled_W * (1.0 / self.lengthscale)

        # M x D @ D x N
        Z = torch.mm(W, X.t())
        uniform_weight = torch.sqrt(torch.tensor(2.0 / self.M)) # * np.sqrt(self.kernel_.variance)
        Q_cos = uniform_weight * torch.cos(Z)
        Q_sin = uniform_weight * torch.sin(Z)

        return torch.cat((Q_cos, Q_sin), dim=0).t()

    def extra_repr(self):
        return 'D={}, M={}'.format(self.D, self.M)


# Custom alternative to BatchNorm (incomplete):
# def FeatureNorm(nn.Module):
#     __constants__ = ['eps', 'running_mean', 'running_var']

#     def __init__(self, num_features, eps=1e-5):
#         self.num_features = num_features
#         self.eps = eps

#         self.register_buffer('running_mean', torch.zeros(num_features))
#         self.register_buffer('running_var', torch.ones(num_features))

#     def reset_running_stats(self):
#         if self.track_running_stats:
#             self.running_mean.zero_()
#             self.running_var.fill_(1)

#     def reset_parameters(self):
#         self.reset_running_stats()


# training: record mean and std: eps=1e-6
        # mean = x.mean(-1)
        # std = x.std(-1)
# forward pass: add normalization
        # (x - mean) / (std + self.eps)

