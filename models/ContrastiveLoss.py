"""
    Created by @namhainguyen2803 on 12/09/2023
"""

import torch
import torch.nn as nn


class ContrastivePrototypicalLoss(nn.Module):
    """
        Loss function in CPP
    """

    def __init__(self, temperature=3, reduction="mean"):
        super(ContrastivePrototypicalLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, z_feature, label, previous_prototype=None):

        assert z_feature.ndim > 1, "z_feature must have number of dimension > 1."
        (batch_size, emb_d) = z_feature.shape
        assert z_feature.shape[0] == label.shape[0], "z_feature.shape[0] != label.shape[0]"

        if z_feature.ndim > 2:
            z_feature = z_feature.reshape(z_feature.shape[0], -1)  # flatten

        if previous_prototype is None:
            concat_z_and_prototype = z_feature
            num_prototype = 0
        else:
            assert previous_prototype.ndim > 1, "previous_prototype must have number of dimension > 1."
            if previous_prototype.ndim > 2:
                previous_prototype = previous_prototype.reshape(previous_prototype.shape[0], -1)
            assert z_feature.shape[1] == previous_prototype.shape[
                1], "z_feature.shape[1] != previous_prototype.shape[1]"
            (num_prototype, _) = previous_prototype.shape
            concat_z_and_prototype = torch.cat([z_feature, previous_prototype], dim=0)

        z_dot_z_T = torch.div(torch.matmul(z_feature, concat_z_and_prototype.T), self.temperature)

        # create mask_for_same_classes
        mask_for_same_classes = torch.zeros(batch_size, batch_size + num_prototype).to(self._device)
        labels = label.contiguous().view(-1, 1)
        current_task_mask = torch.eq(labels, labels.T).float()  # 1 if same class, 0 if not same class
        mask_for_same_classes[:batch_size, :batch_size] = current_task_mask
        mask_for_different_classes = 1 - mask_for_same_classes  # 0 if same class, 1 if not same class
        # numerical stability
        max_z_dot_z_T = torch.max(z_dot_z_T, dim=1, keepdim=True).values
        assert max_z_dot_z_T.shape == (batch_size, 1), "max_z_dot_z_T.shape != (batch_size, 1)."
        z_dot_z_T = z_dot_z_T - max_z_dot_z_T.detach()
        loss_for_each_instance = (-1 / torch.sum(mask_for_same_classes, dim=1, keepdim=True)).reshape(-1, 1) * \
                                 (torch.sum(z_dot_z_T * mask_for_same_classes, dim=1, keepdim=True) -
                                  torch.log(torch.sum(torch.exp(z_dot_z_T * mask_for_different_classes), dim=1,
                                                      keepdim=True) -
                                            torch.sum(1 - mask_for_different_classes, dim=1, keepdim=True)))
        assert loss_for_each_instance.shape == (batch_size, 1), "loss_for_each_instance.shape != (batch_size, 1)"

        if self.reduction == "mean":
            return torch.mean(loss_for_each_instance)
        elif self.reduction == "sum":
            return torch.sum(loss_for_each_instance)
        elif self.reduction == "none":
            return loss_for_each_instance.squeeze(-1)
