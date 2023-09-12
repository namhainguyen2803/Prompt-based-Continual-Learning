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
            assert z_feature.shape[1] == previous_prototype.shape[1], "z_feature.shape[1] != previous_prototype.shape[1]"
            (num_prototype, _) = previous_prototype.shape
            concat_z_and_prototype = torch.cat([z_feature, previous_prototype], dim=0)

        z_dot_z_T = torch.div(torch.matmul(z_feature, concat_z_and_prototype.T), self.temperature)
        # z_dot_z_T.shape == (batch_size, batch_size + num_prototype)

        # create mask_for_same_classes, 1 if same class, 0 if not same class or itself
        # in mask, each row is the mask for corresponding instance
        mask_for_same_classes = torch.zeros(batch_size, batch_size + num_prototype)
        labels = label.contiguous().view(-1, 1)
        current_task_mask = torch.eq(labels, labels.T).float() # 1 if same class, 0 if not same class
        mask_for_same_classes[:batch_size, :batch_size] = current_task_mask
        mask_for_different_classes = 1 - mask_for_same_classes # 0 if same class, 1 if not same class
        mask_for_same_classes[range(batch_size), range(batch_size)] = 0 # 1 if same class, 0 if not same class or itself

        # numerical stability
        # dim=1 -> max in row
        # dim=0 -> max in column
        max_z_dot_z_T = torch.max(z_dot_z_T, dim=1, keepdim=True).values
        assert max_z_dot_z_T.shape == (batch_size, 1), "max_z_dot_z_T.shape != (batch_size, 1)."
        z_dot_z_T = z_dot_z_T - max_z_dot_z_T.detach() # shape == (batch_size, batch_size + num_prototype)

        # sum_for_same_class(z.z^T) - log( sum_for_different_class( exp(z.z^T) )
        loss_for_each_instance = torch.sum(z_dot_z_T * mask_for_same_classes, dim=1, keepdim=True) - \
                                 torch.log(torch.sum(torch.exp(z_dot_z_T * mask_for_different_classes), dim=1, keepdim=True) -
                                           torch.sum(1 - mask_for_different_classes, dim=1, keepdim=True))
        assert loss_for_each_instance.shape == (batch_size, 1), "loss_for_each_instance.shape != (batch_size, 1)"

        if self.reduction == "mean":
            return torch.mean(loss_for_each_instance)
        elif self.reduction == "sum":
            return torch.sum(loss_for_each_instance)
        elif self.reduction == "none":
            return loss_for_each_instance.squeeze(-1)

if __name__ == "__main__":
    cpp_loss = ContrastivePrototypicalLoss(temperature=3, reduction="none")
    z_feature = torch.Tensor([[0.6011, 1.8857, 0.5336],
                              [-0.8425, -2.2244, 0.6984],
                              [-0.1734, -0.9198, -1.5337],
                              [-0.2995, 0.7109, 1.5230],
                              [-0.2606, 0.1352, -1.2000]])
    label = torch.Tensor([2, 0, 0, 1, 2])
    previous_prototype = torch.Tensor([[ 0.6475, -1.1527, -0.9547],
                                        [-0.0105, -0.9611, -1.0651]])
    loss_value = cpp_loss(z_feature, label)
    print(loss_value[0]) # it should be around -1.4323, according to my manual calculation