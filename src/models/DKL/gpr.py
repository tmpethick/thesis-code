import torch
import gpytorch

from src.experiment.config_helpers import LazyConstructor


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self,
        train_x,
        train_y,
        likelihood,
        feature_extractor=None,
        n_grid=None,
        inducing_points=None,
        kernel=LazyConstructor(gpytorch.kernels.RBFKernel, lengthscale_prior=None),
        has_scale_kernel=True):

        if feature_extractor is not None:
            gp_input_dim = feature_extractor.output_dim
        else:
            gp_input_dim = train_x.shape[-1]

        super().__init__(train_x, train_y, likelihood)

        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ZeroMean()

        self.uses_inducing_points = n_grid is not None or inducing_points is not None
        self.n_grid = n_grid
        self.inducing_points = inducing_points

        kernel = kernel(ard_num_dims=gp_input_dim)

        if n_grid is not None:
            kernel = gpytorch.kernels.GridInterpolationKernel(
                kernel,
                num_dims=gp_input_dim,
                grid_size=n_grid,
                grid_bounds=None, # TODO: should we set grid bounds?
            )

        if has_scale_kernel:
            kernel = gpytorch.kernels.ScaleKernel(kernel)

        if inducing_points is not None:
            kernel = gpytorch.kernels.InducingPointKernel(
                kernel,
                inducing_points=torch.randn(inducing_points, gp_input_dim, dtype=train_x.dtype),
                likelihood=likelihood
            )

        self.covar_module = kernel

        # Initialize lengthscale and outputscale to mean of priors
        # if lengthscale_prior is not None:
        #     self.rbf_kernel.lengthscale = lengthscale_prior.mean
        # if outputscale_prior is not None:
        #     self.scale_kernel.outputscale = outputscale_prior.mean

    def forward(self, x):
        if self.feature_extractor is not None:
            projected_x = self.feature_extractor(x)
        else:
            projected_x = x

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
