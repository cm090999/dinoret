import torch
import torch.nn as nn
import torch.nn.functional as F

class DistanceWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, weight_factor=1.0, penalty_type='linear', class_weights=None):
        super(DistanceWeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_factor = weight_factor
        self.penalty_type = penalty_type
        self.class_weights = class_weights

    def forward(self, logits, targets):
        # Apply class weights if provided
        if self.class_weights is not None:
            if not isinstance(self.class_weights, torch.Tensor):
                class_weights_tensor = torch.tensor(self.class_weights, dtype=torch.float).to(logits.device)
            else:
                class_weights_tensor = self.class_weights.to(logits.device)
        else:
            class_weights_tensor = None

        # Apply class weights if provided
        ce_loss = F.cross_entropy(logits, targets, weight=class_weights_tensor)

        # Predicted class
        _, predictions = torch.max(logits, 1)

        # Calculate distance between true and predicted classes
        distance = torch.abs(targets - predictions).float()

        # Apply the specified penalty type
        if self.penalty_type == 'linear':
            distance_penalty = distance
        elif self.penalty_type == 'squared':
            distance_penalty = distance ** 2
        else:
            raise ValueError("Invalid penalty type. Choose 'linear' or 'squared'.")

        # Apply weight factor and calculate final loss
        weighted_penalty = self.weight_factor * distance_penalty
        final_loss = ce_loss + weighted_penalty.mean()

        return final_loss