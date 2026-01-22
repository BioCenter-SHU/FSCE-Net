import torch
import torch.nn as nn
import torch.nn.functional as F


## iemocap loss function: same with CE loss
class MaskedCELoss(nn.Module):
    def __init__(self):
        super(MaskedCELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')  # Use Negative Log Likelihood Loss (used after log_softmax)

    def forward(self, pred, target, umask):
        """
        Classification task loss (suitable for IEMOCAP), calculates NLLLoss for samples at valid positions.

        Parameters:
        pred   : [batch*seq_len, n_classes], predicted probability distribution (not yet logged)
        target : [batch*seq_len], ground truth labels (integer type)
        umask  : [batch, seq_len], mask for valid utterances (1=valid, 0=padding)

        Returns:
        Average Cross Entropy Loss (calculated only at valid sample positions)
        """

        umask = umask.view(-1, 1)         # Flatten into a column vector [batch*seq_len, 1]
        target = target.view(-1, 1)       # Same as above
        pred = F.log_softmax(pred, 1)     # Apply log_softmax to prediction results, output log-prob

        # Calculate loss (Note: both pred and target are multiplied by mask to remove invalid positions)
        loss = self.loss(pred * umask, (target * umask).squeeze().long()) / torch.sum(umask)
        return loss


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')  # Use Mean Squared Error Loss, sum by element

    def forward(self, pred, target, umask):
        """
        Regression task loss

        Parameters:
        pred   : [batch*seq_len], continuous sentiment values predicted by the model
        target : [batch*seq_len], ground truth sentiment values (range [-3, 3])
        umask  : [batch*seq_len], mask for valid samples (1=valid, 0=invalid)

        Returns:
        Average MSE (calculated only at valid sample positions)
        """

        pred = pred.view(-1, 1)     # [batch*seq_len, 1]
        target = target.view(-1, 1) # [batch*seq_len, 1]
        umask = umask.view(-1, 1)   # [batch*seq_len, 1]

        # Calculate MSE only for positions valid in the mask, then divide by the total number of valid samples
        loss = self.loss(pred * umask, target * umask) / torch.sum(umask)
        return loss