# """
# Module description:

# """

# __version__ = '0.3.1'
# __author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
# __email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

# import torch
# import torch.optim as optim
# import numpy as np
# from tqdm import tqdm
# import os  # For path handling in save/restore

# from elliot.dataset.samplers import custom_sampler as cs
# from elliot.recommender import BaseRecommenderModel
# from elliot.recommender.recommender_utils_mixin import RecMixin
# from elliot.recommender.base_recommender_model import init_charger

# from elliot.recommender.latent_factor_models.BPRMF_batch.BPRMF_batch_model import BPRMF_batch_model


# class BPRMF_batch(RecMixin, BaseRecommenderModel):
#     r"""
#     Batch Bayesian Personalized Ranking with Matrix Factorization (PyTorch Version)

#     For further details, please refer to the `paper <https://arxiv.org/abs/1205.2618.pdf>`_

#     Args:
#         factors: Number of latent factors
#         lr: Learning rate
#         l_w: Regularization coefficient for latent factors (Gu, Gi)
#         l_b: Regularization coefficient for bias (Bi)

#     To include the recommendation model, add it to the config file adopting the following pattern:

#     .. code:: yaml

#       models:
#         BPRMFPytorch: # Name of this PyTorch class
#           meta:
#             save_recs: True
#           epochs: 10
#           batch_size: 512
#           factors: 10
#           lr: 0.001
#           l_w: 0.1
#           l_b: 0.001
#           seed: 42 # Elliot handles seed passing
#     """

#     @init_charger  # Elliot's decorator for auto-parameter handling
#     def __init__(self, data, config, params, *args, **kwargs):
#         super().__init__(data, config, params, *args, **kwargs)

#         self._params_list = [
#             ("_factors", "factors", "factors", 200, int, None),
#             ("_learning_rate", "lr", "lr", 0.001, float, None),
#             ("_l_w", "l_w", "l_w", 0.0, float, None),  # Regularization for Gu, Gi
#             ("_l_b", "l_b", "l_b", 0.0, float, None)  # Regularization for Bi
#         ]
#         self.autoset_params()  # Process parameters from config

#         if self._batch_size < 1:
#             self._batch_size = self._data.transactions

#         self._ratings = self._data.train_dict  # Used by RecMixin for masking during evaluation

#         # Sampler for BPR: samples (user, positive_item, negative_item) triplets
#         self._sampler = cs.Sampler(self._data.i_train_dict)

#         # Setup device: GPU (cuda) if available, otherwise CPU
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # Instantiate the PyTorch model and move it to the selected device
#         self._model = BPRMF_batch_model(
#             self._num_users, self._num_items, self._learning_rate, self._factors,
#             self._l_w, self._l_b,
#             random_seed=self._seed  # Passed from Elliot's config
#         ).to(self.device)

#     @property
#     def name(self):
#         # Define a name for the model, useful for logging and saving
#         return f"BPRMFPytorch_e{self._epochs}_bs{self._batch_size}_f{self._factors}" \
#                f"_lr{self._learning_rate:.5f}_lw{self._l_w:.5f}_lb{self._l_b:.5f}" \
#             # f"_{self.get_base_params_shortcut()}" \ # If using Elliot's param shortcuts
#         # f"_{self.get_params_shortcut()}"

#     def train(self):
#         if self._restore:  # Check if Elliot wants to restore weights
#             restored_path = self._get_restore_path()  # Helper to get the actual path
#             if restored_path:
#                 self.restore_weights(restored_path)  # Call custom restore

#         for it in self.iterate(self._epochs):  # Elliot's epoch iterator
#             self._model.train()  # Set model to training mode
#             total_loss = 0
#             steps = 0

#             # Progress bar for training batches
#             with tqdm(total=int(self._data.transactions // self._batch_size),
#                       desc=f'Epoch {it + 1}/{self._epochs}',
#                       disable=not self._verbose,  # Disable if not verbose
#                       mininterval=0.5,  # Update progress bar at most every 0.5s
#                       ncols=100) as pbar:

#                 # Iterate over batches from the sampler
#                 for batch in self._sampler.step(self._data.transactions, self._batch_size):
#                     total_loss += self._model.train_step(batch)  # Accumulate loss (item() gets scalar value)
#                     steps += 1
#                     pbar.set_postfix_str(f'Loss: {total_loss / steps:.5f}')  # Update progress bar
#                     pbar.update(1)

#             # Evaluate at the end of each epoch using Elliot's evaluation pipeline
#             self.evaluate(it, total_loss / steps)

#     def get_recommendations(self, k: int = 100):
#         self._model.eval()  # Set model to evaluation mode (disables dropout, etc.)
#         predictions_top_k_test = {}
#         predictions_top_k_val = {}

#         with torch.no_grad():  # Disable gradient calculations for inference
#             for start_offset in range(0, self._num_users, self._batch_size):
#                 stop_offset = min(start_offset + self._batch_size, self._num_users)

#                 # Create a tensor of user indices for the current batch, directly on device
#                 user_indices_batch = torch.arange(start_offset, stop_offset,
#                                                   dtype=torch.long, device=self.device)

#                 if user_indices_batch.numel() == 0:  # Skip if batch is empty
#                     continue

#                 # Get predictions for all items for the batch of users
#                 # predictions_batch will be on self.device
#                 predictions_batch = self._model.predict_all_items(user_indices_batch)

#                 # process_protocol (from RecMixin) likely expects numpy arrays on CPU
#                 # It handles masking of training items and top-K selection per user
#                 recs_val, recs_test = self.process_protocol(k,
#                                                             predictions_batch.cpu(),
#                                                             start_offset,
#                                                             stop_offset)

#                 predictions_top_k_val.update(recs_val)
#                 predictions_top_k_test.update(recs_test)

#         return predictions_top_k_val, predictions_top_k_test

#     # Elliot's saving/loading mechanism for PyTorch models
#     def _get_restore_path(self):
#         # Helper to determine which path to use for restoring
#         if self._save_weights:
#             # Prioritize best model if early stopping was used and best path exists
#             if hasattr(self, '_restore_path_best') and self._restore_path_best and os.path.exists(
#                     self._restore_path_best):
#                 # Ensure it's a PyTorch file
#                 if self._restore_path_best.endswith((".pth", ".pt")):
#                     return self._restore_path_best
#             # Fallback to last saved iteration if best is not suitable
#             if hasattr(self, '_restore_path_it') and self._restore_path_it and os.path.exists(self._restore_path_it):
#                 if self._restore_path_it.endswith((".pth", ".pt")):
#                     return self._restore_path_it
#         return None

#     def save_weights(self, path: str):
#         """
#         Saves model weights and optimizer state to a .pth file.
#         Elliot calls this with a path, e.g., 'model_epoch_X.pth' or 'model_best.pth'.
#         """
#         if not path.endswith((".pth", ".pt")):
#             path += ".pth"  # Ensure PyTorch standard extension

#         torch.save({
#             'model_state_dict': self._model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             # Optionally save other things like epoch, loss, etc.
#             # 'epoch': self._current_epoch, # Assuming you track current epoch
#         }, path)

#     def restore_weights(self, path: str):
#         """
#         Restores model weights and optimizer state from a .pth file.
#         Path is provided by Elliot based on its logic.
#         """
#         if path and os.path.exists(path) and path.endswith((".pth", ".pt")):
#             try:
#                 checkpoint = torch.load(path, map_location=self.device)  # Load to current device
#                 self._model.load_state_dict(checkpoint['model_state_dict'])
#                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#                 # Optionally restore other saved states like epoch
#                 # self._current_epoch_restored = checkpoint.get('epoch', 0)
#                 self._logger.info(f"PyTorch Model weights restored from {path}")
#                 return True  # Indicate successful restoration
#             except FileNotFoundError:
#                 self._logger.warning(
#                     f"PyTorch checkpoint file not found at {path}. Training from scratch if this is unexpected.")
#             except Exception as e:
#                 self._logger.error(f"Error restoring PyTorch weights from {path}: {e}. Training from scratch.")
#         else:
#             self._logger.warning(
#                 f"No valid PyTorch checkpoint path provided or file does not exist: {path}. Training from scratch.")
#         return False  # Indicate restoration failed or not attempted

#     # get_config is usually not needed if BaseRecommenderModel handles it,
#     # but if you had specific PyTorch model params not covered by _params_list:
#     # def get_config(self):
#     #     base_config = super().get_config()
#     #     base_config.update({
#     #         "torch_version": torch.__version__,
#     #         # any other PyTorch specific info
#     #     })
#     #     return base_config
"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os  # For path handling in save/restore

from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.base_recommender_model import init_charger

from elliot.recommender.latent_factor_models.BPRMF_batch.BPRMF_batch_model import BPRMF_batch_model


class BPRMF_batch(RecMixin, BaseRecommenderModel):
    r"""
    Batch Bayesian Personalized Ranking with Matrix Factorization (PyTorch Version)

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.2618.pdf>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        l_w: Regularization coefficient for latent factors (Gu, Gi)
        l_b: Regularization coefficient for bias (Bi)

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        BPRMFPytorch: # Name of this PyTorch class
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          lr: 0.001
          l_w: 0.1
          l_b: 0.001
          seed: 42 # Elliot handles seed passing
    """

    @init_charger  # Elliot's decorator for auto-parameter handling
    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)

        self._params_list = [
            ("_factors", "factors", "factors", 200, int, None),
            ("_learning_rate", "lr", "lr", 0.001, float, None),
            ("_l_w", "l_w", "l_w", 0.0, float, None),  # Regularization for Gu, Gi
            ("_l_b", "l_b", "l_b", 0.0, float, None)  # Regularization for Bi
        ]
        self.autoset_params()  # Process parameters from config

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict  # Used by RecMixin for masking during evaluation

        # Sampler for BPR: samples (user, positive_item, negative_item) triplets
        self._sampler = cs.Sampler(self._data.i_train_dict)

        # Setup device: GPU (cuda) if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Instantiate the PyTorch model and move it to the selected device
        self._model = BPRMF_batch_model(
            self._num_users, self._num_items, self._learning_rate, self._factors,
            self._l_w, self._l_b,
            random_seed=self._seed  # Passed from Elliot's config
        ).to(self.device)

    @property
    def name(self):
        # Define a name for the model, useful for logging and saving
        return f"BPRMFPytorch_e{self._epochs}_bs{self._batch_size}_f{self._factors}" \
               f"_lr{self._learning_rate:.5f}_lw{self._l_w:.5f}_lb{self._l_b:.5f}" \
            # f"_{self.get_base_params_shortcut()}" \ # If using Elliot's param shortcuts
        # f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:  # Check if Elliot wants to restore weights
            restored_path = self._get_restore_path()  # Helper to get the actual path
            if restored_path:
                self.restore_weights(restored_path)  # Call custom restore

        for it in self.iterate(self._epochs):  # Elliot's epoch iterator
            self._model.train()  # Set model to training mode
            total_loss = 0
            steps = 0

            # Progress bar for training batches
            with tqdm(total=int(self._data.transactions // self._batch_size),
                      desc=f'Epoch {it + 1}/{self._epochs}',
                      disable=not self._verbose,  # Disable if not verbose
                      mininterval=0.5,  # Update progress bar at most every 0.5s
                      ncols=100) as pbar:

                # Iterate over batches from the sampler
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    total_loss += self._model.train_step(batch)  # Accumulate loss (item() gets scalar value)
                    steps += 1
                    pbar.set_postfix_str(f'Loss: {total_loss / steps:.5f}')  # Update progress bar
                    pbar.update(1)

            # Evaluate at the end of each epoch using Elliot's evaluation pipeline
            self.evaluate(it, total_loss / steps)

    def get_recommendations(self, k: int = 100):
        self._model.eval()  # Set model to evaluation mode (disables dropout, etc.)
        predictions_top_k_test = {}
        predictions_top_k_val = {}

        with torch.no_grad():  # Disable gradient calculations for inference
            for start_offset in range(0, self._num_users, self._batch_size):
                stop_offset = min(start_offset + self._batch_size, self._num_users)

                # Create a tensor of user indices for the current batch, directly on device
                user_indices_batch = torch.arange(start_offset, stop_offset,
                                                  dtype=torch.long, device=self.device)

                if user_indices_batch.numel() == 0:  # Skip if batch is empty
                    continue

                # Get predictions for all items for the batch of users
                # predictions_batch will be on self.device
                predictions_batch = self._model.predict_all_items(user_indices_batch)

                # process_protocol (from RecMixin) likely expects numpy arrays on CPU
                # It handles masking of training items and top-K selection per user
                recs_val, recs_test = self.process_protocol(k,
                                                            predictions_batch.cpu(),
                                                            start_offset,
                                                            stop_offset)

                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    # Elliot's saving/loading mechanism for PyTorch models
    def _get_restore_path(self):
        # Helper to determine which path to use for restoring
        if self._save_weights:
            # Prioritize best model if early stopping was used and best path exists
            if hasattr(self, '_restore_path_best') and self._restore_path_best and os.path.exists(
                    self._restore_path_best):
                # Ensure it's a PyTorch file
                if self._restore_path_best.endswith((".pth", ".pt")):
                    return self._restore_path_best
            # Fallback to last saved iteration if best is not suitable
            if hasattr(self, '_restore_path_it') and self._restore_path_it and os.path.exists(self._restore_path_it):
                if self._restore_path_it.endswith((".pth", ".pt")):
                    return self._restore_path_it
        return None

    def save_weights(self, path: str):
        """
        Saves model weights and optimizer state to a .pth file.
        Elliot calls this with a path, e.g., 'model_epoch_X.pth' or 'model_best.pth'.
        """
        if not path.endswith((".pth", ".pt")):
            path += ".pth"  # Ensure PyTorch standard extension

        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Optionally save other things like epoch, loss, etc.
            # 'epoch': self._current_epoch, # Assuming you track current epoch
        }, path)

    def restore_weights(self, path: str):
        """
        Restores model weights and optimizer state from a .pth file.
        Path is provided by Elliot based on its logic.
        """
        if path and os.path.exists(path) and path.endswith((".pth", ".pt")):
            try:
                checkpoint = torch.load(path, map_location=self.device)  # Load to current device
                self._model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Optionally restore other saved states like epoch
                # self._current_epoch_restored = checkpoint.get('epoch', 0)
                self._logger.info(f"PyTorch Model weights restored from {path}")
                return True  # Indicate successful restoration
            except FileNotFoundError:
                self._logger.warning(
                    f"PyTorch checkpoint file not found at {path}. Training from scratch if this is unexpected.")
            except Exception as e:
                self._logger.error(f"Error restoring PyTorch weights from {path}: {e}. Training from scratch.")
        else:
            self._logger.warning(
                f"No valid PyTorch checkpoint path provided or file does not exist: {path}. Training from scratch.")
        return False  # Indicate restoration failed or not attempted

    # get_config is usually not needed if BaseRecommenderModel handles it,
    # but if you had specific PyTorch model params not covered by _params_list:
    # def get_config(self):
    #     base_config = super().get_config()
    #     base_config.update({
    #         "torch_version": torch.__version__,
    #         # any other PyTorch specific info
    #     })
    #     return base_config

