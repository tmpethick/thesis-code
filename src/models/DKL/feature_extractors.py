import torch
from collections.abc import Iterable

ACTIVATIONS = dict(
    relu=torch.nn.ReLU,
    tanh=torch.nn.Tanh,
    leaky_relu=torch.nn.LeakyReLU,
)

# def init_weights(m):
#     pass
#     if type(m) == torch.nn.Linear:
#         #torch.nn.init.xavier_uniform(m.weight)
#         m.bias.data.fill_(0.0)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, D=None, layers=(50, 25, 10, 2), normalize_output=True, output_activation=False, activation='relu'):
        super().__init__()

        assert D is not None, "Dimensionality D has to be specified."
        assert len(layers) >= 1, "You need to specify at least an output layer size."
        layers = (D,) + tuple(layers)
        
        if isinstance(activation, str):
            activation = [activation] * len(layers)

        i = -1
        for i in range(0, len(layers) - 2):
            Activation = ACTIVATIONS[activation[i]]
            # import matplotlib.pyplot as plt
            # X_torch = torch.linspace(-1,1, 100)
            # Y_torch = Activation()(X_torch)
            # plt.plot(X_torch.numpy(), Y_torch.numpy())
            # plt.show()

            in_ = layers[i]
            out = layers[i + 1]
            self.add_module('linear{}'.format(i), torch.nn.Linear(in_, out))
            self.add_module('activation{}'.format(i), Activation())
            # self.add_module('normalization{}'.format(i), torch.nn.BatchNorm1d(out, affine=False, momentum=1))

        self.output_dim = layers[-1]
        self.add_module('linear{}'.format(i+1), torch.nn.Linear(layers[-2], self.output_dim))
        if output_activation:
            Activation = ACTIVATIONS[activation[i+1]]
            print(Activation)
            self.add_module('activation{}'.format(i+1), Activation())
        if normalize_output:
            self.add_module('normalization', torch.nn.BatchNorm1d(self.output_dim, affine=False, momentum=1))

        #self.apply(init_weights)


class RFFEmbedding(torch.nn.Module):
    def __init__(self, D, M, ARD=False, optimize_spectral_points=False):
        super().__init__()
        self.D = D
        self.M = M
        self.output_dim = M

        assert M % 2 == 0, "M has to be even since there is a feature for both sin and cos."

        # register lengthscale as parameter
        self.ARD = ARD
        if self.ARD:
            self.lengthscale = torch.nn.Parameter(torch.tensor(0.61).repeat(D).diag())
        else:
            self.lengthscale = torch.nn.Parameter(torch.Tensor(1))
            self.lengthscale.data.fill_(0.61)

        # sample self.unscaled_W shape: (M, D)
        W_init = torch.randn(self.M // 2, self.D)
        if optimize_spectral_points:
            self.unscaled_W = torch.nn.Parameter(W_init)
        else:
            self.register_buffer('unscaled_W', W_init)
  
    def forward(self, X): # NxD -> MxD
        if self.ARD:
            W = self.unscaled_W.mm(self.lengthscale.inverse())
        else:
            W = self.unscaled_W * (1.0 / self.lengthscale)

        # M x D @ D x N
        Z = torch.mm(W, X.t())
        uniform_weight = torch.sqrt(torch.tensor(2.0 / self.M)) # * np.sqrt(self.kernel_.variance)
        Q_cos = uniform_weight * torch.cos(Z)
        Q_sin = uniform_weight * torch.sin(Z)

        return torch.cat((Q_cos, Q_sin), dim=0).t()

    def extra_repr(self):
        return 'D={}, M={}'.format(self.D, self.M)
    
    def initialize(self, lengthscale=None):
        if self.ARD:
            lengthscale = torch.tensor(lengthscale)
            self.lengthscale.data.as_strided([self.D], [self.D + 1]).copy_(lengthscale)
        else:
            self.lengthscale.data.fill_(lengthscale)


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

