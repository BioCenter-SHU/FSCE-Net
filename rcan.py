import torch.nn as nn

class CALayer(nn.Module):
    """
    Channel Attention Layer.
    This layer learns the importance of each feature channel and re-weights the features based on this importance.
    This helps the model focus on more informative channel features.
    """
    def __init__(self, num_channels, reduction=16):
        """
        Initialization function.
        :param num_channels: Number of channels in the input feature map.
        :param reduction: Reduction ratio for channel dimensionality reduction, used to reduce parameter count.
        """
        super().__init__()
        # 1. Squeeze operation: Global Average Pooling
        # Use adaptive average pooling to compress the feature map of each channel (length L) into a single value.
        # Input shape: [N, C, L] -> Output shape: [N, C, 1]
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 2. Excitation operation: Two fully connected layers (implemented using 1x1 convolutions)
        # This is a bottleneck structure used to learn non-linear relationships between channels.
        self.conv_du = nn.Sequential(
            # First layer: Dimensionality reduction. Reduce channel count from num_channels to num_channels // reduction
            nn.Conv1d(num_channels, num_channels // reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            # Second layer: Dimensionality expansion. Restore channel count to num_channels
            nn.Conv1d(num_channels // reduction, num_channels, 1, 1, 0),
            # Sigmoid activation function normalizes the output weights to between (0, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward propagation.
        :param x: Input feature map, shape [N, C, L]
        :return: Feature map weighted by channel attention
        """
        # Shape of y is [N, C, 1], representing global information of each channel
        y = self.avg_pool(x)
        # Shape of y remains [N, C, 1], but its values are the learned attention weights for each channel
        y = self.conv_du(y)
        # 3. Rescale operation:
        # Multiply original input x with learned channel weights y to weight each channel.
        # PyTorch's broadcasting mechanism automatically expands y's shape [N, C, 1] to match x's [N, C, L].
        return x * y

class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB).
    This is the core building block of the RCAN model. It embeds the channel attention mechanism into a standard residual block.
    """
    def __init__(self, num_channels, reduction, res_scale=1.0):
        """
        Initialization function.
        :param num_channels: Input/Output channel count.
        :param reduction: Reduction ratio in the channel attention layer.
        :param res_scale: Residual scaling factor, used to stabilize training of deep networks.
        """
        super().__init__()
        
        # Define the body part of the residual block
        body = [
            nn.Conv1d(num_channels, num_channels, 3, 1, 1), # 3x1 convolution, padding=1 maintains length
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels, num_channels, 3, 1, 1),
        ]
        # Append a channel attention layer after the convolution layers
        body.append(CALayer(num_channels, reduction))
        
        self.body = nn.Sequential(*body)
        self.res_scale = res_scale

    def forward(self, x):
        """
        Forward propagation.
        :param x: Input feature map
        :return: Output of the RCAB block
        """
        # Calculate residual part
        res = self.body(x).mul(self.res_scale)
        # Add residual to original input (Short Skip Connection)
        res += x
        return res

class Group(nn.Module):
    """
    Residual Group.
    A residual group consists of multiple RCAB blocks and a convolution layer, stabilized by a long skip connection.
    This structure allows building very deep networks.
    """
    def __init__(self, num_channels, num_blocks, reduction, res_scale=1.0):
        """
        Initialization function.
        :param num_channels: Channel count.
        :param num_blocks: Number of RCAB blocks in the group.
        :param reduction: Reduction ratio in channel attention layer.
        :param res_scale: Residual scaling factor.
        """
        super().__init__()
        
        body = []
        # Add num_blocks RCAB blocks to the group
        for _ in range(num_blocks):
            body.append(RCAB(num_channels, reduction, res_scale))
        # Add another convolution layer after all RCAB blocks
        body.append(nn.Conv1d(num_channels, num_channels, 3, 1, 1))
        
        self.body = nn.Sequential(*body)

    def forward(self, x):
        """
        Forward propagation.
        :param x: Input feature map
        :return: Output of the Residual Group
        """
        # Calculate residual passing through the whole group
        res = self.body(x)
        # Add residual to the group's input (Long Skip Connection)
        res += x
        return res