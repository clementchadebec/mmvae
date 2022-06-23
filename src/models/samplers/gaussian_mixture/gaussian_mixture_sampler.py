# Based on the pythae implementation but adapted to fit in my framework

import logging
from tqdm import tqdm
import torch
from sklearn import mixture
from utils import unpack_data

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class GaussianMixtureSampler():
    """Fits a Gaussian Mixture in the Autoencoder's latent space.

    Args:
        model (BaseAE): The vae model to sample from.
        sampler_config (BaseSamplerConfig): An instance of BaseSamplerConfig in which any sampler's
            parameters is made available. If None a default configuration is used. Default: None.

    .. note::

        The method :class:`~pythae.samplers.GaussianMixtureSampler.fit` must be called to fit the sampler
        before sampling.

    """

    def __init__(
        self, encoder, n_components = 10, device='cuda'
    ):
        self.encoder = encoder
        self.n_components = n_components
        self.device = device

    def fit(self, train_loader):
        """Method to fit the sampler from the training data

        Args:
            train_data (torch.Tensor): The train data needed to retreive the training embeddings
                    and fit the mixture in the latent space. Must be of shape n_imgs x im_channels x ...
                    and in range [0-1]
        """
        self.is_fitted = True

        mu = []

        with torch.no_grad():
            for i,dataT in enumerate(tqdm(train_loader)):
                data = unpack_data(dataT, device=self.device)
                mu_data = self.encoder(data)[0]
                mu.append(mu_data)

        mu = torch.cat(mu)

        if self.n_components > mu.shape[0]:
            self.n_components = mu.shape[0]
            logger.warning(
                f"Setting the number of component to {mu.shape[0]} since"
                "n_components > n_samples when fitting the gmm"
            )

        gmm = mixture.GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            max_iter=2000,
            verbose=0,
            tol=1e-3,
        )
        gmm.fit(mu.cpu().detach())

        self.gmm = gmm

    def sample(
        self,
        num_samples: int = 1,
        batch_size: int = 500,
        output_dir: str = None,
        return_gen: bool = True,
        save_sampler_config: bool = False,
    ) -> torch.Tensor:
        """Main sampling function of the sampler.

        Args:
            num_samples (int): The number of samples to generate
            batch_size (int): The batch size to use during sampling
            output_dir (str): The directory where the images will be saved. If does not exist the
                folder is created. If None: the images are not saved. Defaults: None.
            return_gen (bool): Whether the sampler should directly return a tensor of generated
                data. Default: True.
            save_sampler_config (bool): Whether to save the sampler config. It is saved in
                output_dir

        Returns:
            ~torch.Tensor: The generated images
        """

        if not self.is_fitted:
            raise ArithmeticError(
                "The sampler needs to be fitted by calling smapler.fit() method"
                "before sampling."
            )

        full_batch_nbr = int(num_samples / batch_size)
        last_batch_samples_nbr = num_samples % batch_size

        z_gen_list = []

        for i in range(full_batch_nbr):

            z = (
                torch.tensor(self.gmm.sample(batch_size)[0])
                .to(self.device)
                .type(torch.float)
            )
            z_gen_list.append(z)

        if last_batch_samples_nbr > 0:
            z = (
                torch.tensor(self.gmm.sample(last_batch_samples_nbr)[0])
                .to(self.device)
                .type(torch.float)
            )
            z_gen_list.append(z)


        return torch.cat(z_gen_list, dim=0)
