import torch



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


