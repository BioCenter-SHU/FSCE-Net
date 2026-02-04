import torch
import torch.nn as nn


class AdaptiveConfidenceFusion(nn.Module):
    """
    A module for adaptive confidence weighting and multimodal feature fusion.
    1. Learn a confidence score for each modality (view).
    2. Weight the features of each modality using this score.
    3. Use multi-head self-attention mechanism to fuse the weighted features.
    """
    def __init__(self, feature_dim, n_classes, dropout=0.5):
        """
        Args:
            feature_dim (int): Feature dimension of each input modality.
            n_classes (int): Total number of classification classes.
            num_heads (int): Number of heads in the multi-head attention mechanism.
            dropout (float): Dropout ratio.
        """
        super(AdaptiveConfidenceFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_modalities = 3  # Fixed number of modalities: Audio, Text, Video

        # Create a confidence predictor for each modality (a simple linear layer)
        # Input is features, output is a 1D confidence score
        self.confidence_predictors = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())
            for _ in range(self.num_modalities)
        ])

        # ===== Build a deeper, more powerful classifier =====
        # Designed mimicking the Feed-Forward Network (FFN) of Transformers
        # Usually, the intermediate layer dimension of FFN expands first and then compresses
        # ffn_hidden_dim1 = feature_dim * 2  # First hidden layer, performing dimension expansion
        # ffn_hidden_dim2 = feature_dim      # Second hidden layer, can keep or change dimension

        # self.modality_classifiers = nn.ModuleList([
        #     nn.Sequential(
        #         # First layer
        #         nn.Linear(feature_dim, ffn_hidden_dim1),
        #         nn.LayerNorm(ffn_hidden_dim1), # Layer Normalization
        #         nn.GELU(),                     # GELU activation
        #         nn.Dropout(dropout),
                
        #         # Second layer
        #         nn.Linear(ffn_hidden_dim1, ffn_hidden_dim2),
        #         nn.LayerNorm(ffn_hidden_dim2),
        #         nn.GELU(),
        #         nn.Dropout(dropout),

        #         # Output layer
        #         nn.Linear(ffn_hidden_dim2, n_classes)
        #     ) for _ in range(self.num_modalities)
        # ])
        # Modified to a simple linear mapping layer
        self.modality_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, n_classes) for _ in range(self.num_modalities)
        ])
        
        # Multi-head attention fusion layer
        # batch_first=True because we will reshape [Time, Batch, Dim] to [Time*Batch, Modal, Dim]
        # self.attention_fusion = nn.MultiheadAttention(
        #     embed_dim=feature_dim,
        #     num_heads=num_heads,
        #     dropout=dropout,
        #     batch_first=True
        # )
        # self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, audio_feats, text_feats, video_feats):
        """
        Args:
            audio_feats (Tensor): Audio features [Time, Batch, Dim]
            text_feats (Tensor): Text features [Time, Batch, Dim]
            video_feats (Tensor): Video features [Time, Batch, Dim]

        Returns:
            Tensor: Fused feature tensor [Time, Batch, Dim * 3]
        """
        time_len, batch_size, _ = audio_feats.shape
        features_list = [audio_feats, text_feats, video_feats]

        flat_features = [f.transpose(0, 1).contiguous().view(-1, f.size(2)) for f in features_list]

        weighted_features = []
        modality_logits = []
        modality_confidences = []

        for i in range(self.num_modalities):
            logits = self.modality_classifiers[i](flat_features[i])
            modality_logits.append(logits)

            confidence_score = self.confidence_predictors[i](flat_features[i])
            modality_confidences.append(confidence_score)
            
            confidence_weight = torch.sigmoid(confidence_score)
            weighted = flat_features[i] * confidence_weight
            weighted_features.append(weighted)

        flat_fused = torch.cat(weighted_features, dim=1)

        final_output = flat_fused.view(batch_size, time_len, -1).transpose(0, 1).contiguous()

        return final_output, modality_logits, modality_confidences
