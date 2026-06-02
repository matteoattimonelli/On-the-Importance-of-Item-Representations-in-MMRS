# """
# Module description:

# """

# __version__ = '0.3.1'
# __author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
# __email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'


# import torch
# import torch.nn as nn
# import numpy as np
# import random

# class BPRMF_batch_model(nn.Module):
#     def __init__(self,
#                  num_users,
#                  num_items,
#                  learning_rate,
#                  factors=200,
#                  l_w=0, l_b=0,
#                  random_seed=42):
#         super().__init__()
#         random.seed(random_seed)
#         np.random.seed(random_seed)
#         torch.manual_seed(random_seed)
#         torch.cuda.manual_seed(random_seed)
#         torch.cuda.manual_seed_all(random_seed)
#         torch.backends.cudnn.deterministic = True

#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         self._num_users = num_users
#         self._num_items = num_items
#         self._factors = factors
#         self._learning_rate = learning_rate
#         self._l_w = l_w
#         self._l_b = l_b

#         # User and Item Embeddings
#         self.Gu = nn.Embedding(self._num_users, self._factors)
#         nn.init.xavier_uniform_(self.Gu.weight) # Glorot/Xavier initialization

#         self.Gi = nn.Embedding(self._num_items, self._factors)
#         nn.init.xavier_uniform_(self.Gi.weight)

#         # Item Biases (as an embedding layer with 1-dimensional embeddings)
#         self.Bi = nn.Embedding(self._num_items, 1)
#         nn.init.zeros_(self.Bi.weight) # Initialize biases to zero

#         # Setup optimizer
#         self.optimizer =  torch.optim.Adam(self.parameters(), lr=self._learning_rate)

#     def forward(self, users_indices, items_indices):
#         """
#         Calculates the BPR score for given user-item pairs.
#         Args:
#             users_indices: Tensor of user indices.
#             items_indices: Tensor of item indices.
#         Returns:
#             predictions: The predicted scores (xui).
#             beta_i: Item biases for the given items.
#             gamma_u: User embeddings for the given users.
#             gamma_i: Item embeddings for the given items.
#         """
#         # Ensure inputs are LongTensors for embedding lookup
#         gamma_u = self.Gu(users_indices.long())  # (batch_size, factors)
#         gamma_i = self.Gi(items_indices.long())  # (batch_size, factors)
#         beta_i = self.Bi(items_indices.long()).squeeze(-1)  # (batch_size)

#         # Element-wise product and sum over factors dimension
#         xui = beta_i + torch.sum(gamma_u * gamma_i, dim=1) # (batch_size)

#         return xui, beta_i, gamma_u, gamma_i

#     def train_step(self, batch):
#         users, pos_items, neg_items = batch

#         # Convert numpy arrays to PyTorch LongTensors and move to device
#         users_tensor = torch.LongTensor(users).to(self.device)
#         pos_items_tensor = torch.LongTensor(pos_items).to(self.device)
#         neg_items_tensor = torch.LongTensor(neg_items).to(self.device)

#         self.optimizer.zero_grad()  # Clear previous gradients

#         # Forward pass for positive items
#         # Model returns: xui, beta_i, gamma_u, gamma_i
#         xu_pos, beta_pos, gamma_u_pos, gamma_pos = self.forward(users_tensor, pos_items_tensor)

#         # Forward pass for negative items
#         # User embeddings (gamma_u_neg) will be the same as gamma_u_pos for the same users
#         xu_neg, beta_neg, _, gamma_neg = self.forward(users_tensor, neg_items_tensor)

#         # BPR loss calculation: -log(sigmoid(xu_pos - xu_neg))
#         # Using softplus for numerical stability: softplus(-x) = log(1 + exp(-x))
#         difference = xu_pos - xu_neg  # No need to clip if using softplus carefully
#         bpr_loss_batch = torch.nn.functional.softplus(-difference).mean()

#         # Regularization (L2 penalty)
#         # tf.nn.l2_loss(t) = sum(t ** 2) / 2. PyTorch: 0.5 * t.pow(2).sum()
#         reg_loss_params = 0.5 * (gamma_u_pos.pow(2).sum() +
#                                  gamma_pos.pow(2).sum() +
#                                  gamma_neg.pow(2).sum()) / users.shape[0]

#         # The original TensorFlow code had a peculiar scaling for beta_neg regularization:
#         # l_b * l2_loss(beta_pos) + l_b * l2_loss(beta_neg) / 10
#         # Replicating this behavior:
#         reg_loss_bias = 0.5 * (beta_pos.pow(2).sum() + (beta_neg.pow(2).sum() / 10.0))

#         # Total loss
#         loss = bpr_loss_batch + self._l_w * reg_loss_params + self._l_b * reg_loss_bias

#         loss.backward()  # Compute gradients
#         self.optimizer.step()  # Update model parameters

#         return loss.detach().cpu().numpy()

#     def predict_all_items(self, users_indices):
#         """
#         Predicts scores for the given users against all items.
#         Args:
#             users_indices: Tensor of user indices (LongTensor).
#         Returns:
#             scores: A tensor of shape (num_users_in_batch, num_total_items)
#         """
#         gamma_u_batch = self.Gu(users_indices.long()) # (batch_size, factors)
#         all_gamma_i = self.Gi.weight          # (num_items, factors)
#         all_beta_i = self.Bi.weight.squeeze(-1) # (num_items)

#         # Scores = Bi + Gu @ Gi.T
#         # Gu @ Gi.T -> (batch_size, factors) @ (factors, num_items) = (batch_size, num_items)
#         scores = all_beta_i.unsqueeze(0) + torch.matmul(gamma_u_batch, all_gamma_i.t())
#         return scores

#     def get_top_k(self, predictions, train_mask, k=100):
#         """
#         Gets top_k items from predictions, masking out items in train_mask.
#         Args:
#             predictions: Tensor of scores (batch_size, num_items).
#             train_mask: Boolean tensor (batch_size, num_items), True for items to mask.
#             k: Number of top items to return.
#         Returns:
#             top_k_scores: Scores of the top-k items.
#             top_k_indices: Indices of the top-k items.
#         """
#         # Apply mask: set already interacted items to a very low score
#         # Ensure -np.inf tensor is on the same device as predictions
#         masked_predictions = torch.where(torch.tensor(train_mask, device=self.device),
#                                          predictions.to(self.device),
#                                          torch.tensor(-np.inf, device=self.device))
#         top_k_scores, top_k_indices = torch.topk(masked_predictions, k=k, sorted=True)
#         return top_k_scores.cpu(), top_k_indices.cpu()

#     def get_config(self):
#         raise NotImplementedError
"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'


import torch
import torch.nn as nn
import numpy as np
import random

class BPRMF_batch_model(nn.Module):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 factors=200,
                 l_w=0, l_b=0,
                 random_seed=42):
        super().__init__()
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._num_users = num_users
        self._num_items = num_items
        self._factors = factors
        self._learning_rate = learning_rate
        self._l_w = l_w
        self._l_b = l_b

        # User and Item Embeddings
        self.Gu = nn.Embedding(self._num_users, self._factors)
        nn.init.xavier_uniform_(self.Gu.weight) # Glorot/Xavier initialization

        self.Gi = nn.Embedding(self._num_items, self._factors)
        nn.init.xavier_uniform_(self.Gi.weight)

        # Item Biases (as an embedding layer with 1-dimensional embeddings)
        self.Bi = nn.Embedding(self._num_items, 1)
        nn.init.zeros_(self.Bi.weight) # Initialize biases to zero

        # Setup optimizer
        self.optimizer =  torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    def forward(self, users_indices, items_indices):
        """
        Calculates the BPR score for given user-item pairs.
        Args:
            users_indices: Tensor of user indices.
            items_indices: Tensor of item indices.
        Returns:
            predictions: The predicted scores (xui).
            beta_i: Item biases for the given items.
            gamma_u: User embeddings for the given users.
            gamma_i: Item embeddings for the given items.
        """
        # Ensure inputs are LongTensors for embedding lookup
        gamma_u = torch.squeeze(self.Gu(users_indices.long()))  # (batch_size, factors)
        gamma_i = torch.squeeze(self.Gi(items_indices.long()))  # (batch_size, factors)
        beta_i = torch.squeeze(self.Bi(items_indices.long()))  # (batch_size)

        # Element-wise product and sum over factors dimension
        xui = beta_i + torch.sum(gamma_u * gamma_i, dim=1) # (batch_size)
        # xui = torch.sum(gamma_u * gamma_i, dim=1)  # (batch_size)

        return xui,beta_i, gamma_u, gamma_i

    def train_step(self, batch):
        users, pos_items, neg_items = batch

        # Convert numpy arrays to PyTorch LongTensors and move to device
        users_tensor = torch.LongTensor(users).to(self.device)
        pos_items_tensor = torch.LongTensor(pos_items).to(self.device)
        neg_items_tensor = torch.LongTensor(neg_items).to(self.device)

        self.optimizer.zero_grad()  # Clear previous gradients

        # Forward pass for positive items
        # Model returns: xui, beta_i, gamma_u, gamma_i
        xu_pos, beta_pos, gamma_u_pos, gamma_pos = self.forward(users_tensor, pos_items_tensor)
        # xu_pos, gamma_u_pos, gamma_pos = self.forward(users_tensor, pos_items_tensor)

        # Forward pass for negative items
        # User embeddings (gamma_u_neg) will be the same as gamma_u_pos for the same users
        xu_neg, beta_neg, _, gamma_neg = self.forward(users_tensor, neg_items_tensor)
        # xu_neg, _, gamma_neg = self.forward(users_tensor, neg_items_tensor)

        # BPR loss calculation: -log(sigmoid(xu_pos - xu_neg))
        # Using softplus for numerical stability: softplus(-x) = log(1 + exp(-x))
        difference = xu_pos - xu_neg  # No need to clip if using softplus carefully
        bpr_loss_batch = -torch.mean(torch.nn.functional.logsigmoid(difference))

        # Regularization (L2 penalty)
        # tf.nn.l2_loss(t) = sum(t ** 2) / 2. PyTorch: 0.5 * t.pow(2).sum()
        reg_loss_params = 0.5 * (gamma_u_pos.norm(2).pow(2) +
                                 gamma_pos.norm(2).pow(2) +
                                 gamma_neg.norm(2).pow(2)) / users.shape[0]

        # The original TensorFlow code had a peculiar scaling for beta_neg regularization:
        # l_b * l2_loss(beta_pos) + l_b * l2_loss(beta_neg) / 10
        # Replicating this behavior:
        reg_loss_bias = (beta_pos.norm(2).pow(2) + (beta_neg.norm(2).pow(2) / 10.0))

        # Total loss
        loss = bpr_loss_batch + self._l_w * reg_loss_params + self._l_b * reg_loss_bias
        # loss = bpr_loss_batch + self._l_w * reg_loss_params


        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update model parameters

        return loss.detach().cpu().numpy()

    def predict_all_items(self, users_indices):
        """
        Predicts scores for the given users against all items.
        Args:
            users_indices: Tensor of user indices (LongTensor).
        Returns:
            scores: A tensor of shape (num_users_in_batch, num_total_items)
        """
        gamma_u_batch = self.Gu(users_indices.long()) # (batch_size, factors)
        all_gamma_i = self.Gi.weight          # (num_items, factors)
        all_beta_i = self.Bi.weight.squeeze() # (num_items)

        # Scores = Bi + Gu @ Gi.T
        # Gu @ Gi.T -> (batch_size, factors) @ (factors, num_items) = (batch_size, num_items)
        scores = all_beta_i + torch.matmul(gamma_u_batch, all_gamma_i.t())
        # scores = torch.matmul(gamma_u_batch, all_gamma_i.t())
        return scores

    def get_top_k(self, predictions, train_mask, k=100):
        """
        Gets top_k items from predictions, masking out items in train_mask.
        Args:
            predictions: Tensor of scores (batch_size, num_items).
            train_mask: Boolean tensor (batch_size, num_items), True for items to mask.
            k: Number of top items to return.
        Returns:
            top_k_scores: Scores of the top-k items.
            top_k_indices: Indices of the top-k items.
        """
        # Apply mask: set already interacted items to a very low score
        # Ensure -np.inf tensor is on the same device as predictions
        masked_predictions = torch.where(torch.tensor(train_mask, device=self.device),
                                         predictions.to(self.device),
                                         torch.tensor(-np.inf, device=self.device))
        top_k_scores, top_k_indices = torch.topk(masked_predictions, k=k, sorted=True)
        return top_k_scores.cpu(), top_k_indices.cpu()

    def get_config(self):
        raise NotImplementedError
