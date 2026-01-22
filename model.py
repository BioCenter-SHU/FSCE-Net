import torch
import torch.nn as nn

from torch.autograd import Variable
from graph import batch_graphify

from glow import Glow, ZeroConv2d, gaussian_log_p
from rcan import Group
from torch_geometric.nn import RGATConv

from AdaptiveConfidenceFusion import AdaptiveConfidenceFusion


# Concatenate and align outputs of shape [num_utterance, dim] to [seqlen, batch, dim]
def utterance_to_conversation(outputs, seq_lengths, umask, no_cuda):
    input_conversation_length = torch.tensor(seq_lengths)  # Number of utterances per conversation
    start_zero = input_conversation_length.data.new(1).zero_()  # Build [0]

    if not no_cuda:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)  # Max conversation length in this batch
    # Calculate start utterance index for each conversation [0, 6, 30, 43]
    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)

    # Slice + pad each conversation to get a uniform length batch tensor and transpose to [seqlen, batch, dim]
    outputs = torch.stack([pad(outputs.narrow(0, s, l), max_len, no_cuda) 
                           for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())], 0).transpose(0, 1)
    return outputs



# Pad input tensor to meet target length (extend only on seq_len dimension)
def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):  # If input is Variable type (deprecated), handle similarly
        var = tensor
        if length > var.size(0):  # If length is insufficient, pad with zeros
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var  # Target length met, return directly
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor

    
class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_relations, hidden_size=64, dropout=0.5, no_cuda=False):
        super(GraphNetwork, self).__init__()
        self.no_cuda = no_cuda                # Whether to use CUDA
        self.hidden_size = hidden_size        # Hidden dimension of Graph Neural Network

        self.rgat = RGATConv( # Parameter count: 70,900 / 40,900 - 30,500 = 10,400
            in_channels=num_features,
            out_channels=hidden_size,
            num_relations=1, # Original code was 2 here; if your edge_type categories exceed 2, adjust accordingly
            heads=1,
            concat=False,
            dropout=0.1,
            attention_mechanism='within-relation',
            edge_dim=1  # <--- Core addition: specify edge feature dimension as 1
        )
        # self.gat = GATConv( # Parameter count: 30500
        #     in_channels=num_features,
        #     out_channels=hidden_size,
        #     heads=1,
        #     concat=False,
        #     dropout=0.1,
        #     edge_dim=1  # <--- Keep this parameter to receive edge features
        # )

    def forward(self, features, edge_index, edge_type, seq_lengths, umask, edge_attr=None):

        out = self.rgat(features, edge_index, edge_type, edge_attr)
        # out = self.gat(features, edge_index, edge_attr)

        outputs = torch.cat([features, out], dim=-1)  # Concatenate original input with graph representation => [num_nodes, num_features + hidden_size]

        ## Convert to conversation sequence format => organize each utterance by conversation structure
        outputs = outputs.reshape(-1, outputs.size(1))  # [num_utterance, dim]
        outputs = utterance_to_conversation(outputs, seq_lengths, umask, self.no_cuda)  # [seqlen, batch, dim]
        outputs = outputs.reshape(outputs.size(0), outputs.size(1), 1, -1)  # Add a dimension [seqlen, batch, 1, dim]

        ## Flatten dimension before feeding into GRU
        seqlen = outputs.size(0)
        batch = outputs.size(1)
        outputs = torch.reshape(outputs, (seqlen, batch, -1))  # Flatten into [seqlen, batch, dim]

        hidden = outputs

        return hidden,  # Return final feature representation for each time step



class GraphModel(nn.Module):
    def __init__(self, adim, tdim, vdim, lstm_hidden_size, graph_hidden_size, flow_hidden_size, # 150, 100, 128
                 n_speakers, window_past, window_future, dataset, 
                 n_classes, dropout=0.5, no_cuda=False, use_bn=False,
                 n_flow=8, num_blocks_rec=10, reduction_rec=16, n_block_glow=1, **kwargs):
        self.dataset = dataset
        super(GraphModel, self).__init__()
        
        # ============================ #
        #     Initialize partial parameters
        # ============================ #
        
        self.no_cuda = no_cuda
        self.n_speakers = n_speakers               # Number of speakers in conversation
        self.window_past = window_past             # Past window length considered in graph construction
        self.window_future = window_future         # Future window length considered in graph construction
        self.flow_hidden_size = flow_hidden_size

        self.edge_weight_sigma = 5    # Gaussian kernel sigma for temporal weights
        self.edge_weight_beta = 0.5   # Discount factor for speaker consistency weights

        # ============================ #
        #     Unimodal LSTM
        # ============================ #

        self.lstm_a = nn.LSTM(input_size=adim, hidden_size=lstm_hidden_size, num_layers=1, bidirectional=True, dropout=dropout)
        self.lstm_t = nn.LSTM(input_size=tdim, hidden_size=lstm_hidden_size, num_layers=1, bidirectional=True, dropout=dropout)
        self.lstm_v = nn.LSTM(input_size=vdim, hidden_size=lstm_hidden_size, num_layers=1, bidirectional=True, dropout=dropout)

        # ============================ #
        #     Unimodal Graph Neural Network
        # ============================ #

        self.graph_net_temporal_a = GraphNetwork(lstm_hidden_size * 2, 2, graph_hidden_size, dropout, self.no_cuda)
        self.graph_net_temporal_t = GraphNetwork(lstm_hidden_size * 2, 2, graph_hidden_size, dropout, self.no_cuda)
        self.graph_net_temporal_v = GraphNetwork(lstm_hidden_size * 2, 2, graph_hidden_size, dropout, self.no_cuda)

        graph_out_size = lstm_hidden_size * 2 + graph_hidden_size

        # ============================ #
        #     Normalization
        # ============================ #

        self.ln_a = nn.LayerNorm(graph_out_size)
        self.ln_t = nn.LayerNorm(graph_out_size)
        self.ln_v = nn.LayerNorm(graph_out_size)

        # ============================ #
        #     Projection to Normalizing Flow Dimension
        # ============================ #

        self.proj_to_common_a = nn.Linear(graph_out_size, flow_hidden_size)
        self.proj_to_common_t = nn.Linear(graph_out_size, flow_hidden_size)
        self.proj_to_common_v = nn.Linear(graph_out_size, flow_hidden_size)

        # ============================ #
        #     Normalizing Flow Model
        # ============================ #

        self.flow_a = Glow(in_channel=flow_hidden_size, n_flow=n_flow, n_block=n_block_glow, affine=True, conv_lu=True)
        self.flow_t = Glow(in_channel=flow_hidden_size, n_flow=n_flow, n_block=n_block_glow, affine=True, conv_lu=True)
        self.flow_v = Glow(in_channel=flow_hidden_size, n_flow=n_flow, n_block=n_block_glow, affine=True, conv_lu=True)

        # ============================ #
        #     Class Distribution
        # ============================ #

        n_classes_priors = kwargs.get('n_classes_priors', 2)
        self.priors = nn.ModuleList([
            ZeroConv2d(flow_hidden_size, flow_hidden_size * 2) for _ in range(n_classes_priors)
        ])

        # ============================ #
        #     Reconstruct Missing Features (Dim 400)
        # ============================ #

        self.fusion_mapping_a = nn.Linear(3 * flow_hidden_size, graph_out_size)
        self.fusion_mapping_t = nn.Linear(3 * flow_hidden_size, graph_out_size)
        self.fusion_mapping_v = nn.Linear(3 * flow_hidden_size, graph_out_size)

        # ============================ #
        #     Reconstruct Missing Features (Dim a/t/v, Original Dim)
        # ============================ #

        self.rec_a = nn.Sequential(
            nn.Conv1d(graph_out_size, flow_hidden_size, kernel_size=1),
            Group(num_channels=flow_hidden_size, num_blocks=num_blocks_rec, reduction=reduction_rec),
            nn.Conv1d(flow_hidden_size, adim, kernel_size=1)
        )
        self.rec_t = nn.Sequential(
            nn.Conv1d(graph_out_size, flow_hidden_size, kernel_size=1),
            Group(num_channels=flow_hidden_size, num_blocks=num_blocks_rec, reduction=reduction_rec),
            nn.Conv1d(flow_hidden_size, tdim, kernel_size=1)
        )
        self.rec_v = nn.Sequential(
            nn.Conv1d(graph_out_size, flow_hidden_size, kernel_size=1),
            Group(num_channels=flow_hidden_size, num_blocks=num_blocks_rec, reduction=reduction_rec),
            nn.Conv1d(flow_hidden_size, vdim, kernel_size=1)
        )

        # ============================ #
        #     Confidence Weighted Fusion
        # ============================ #

        self.confidence_fusion_module = AdaptiveConfidenceFusion(
            feature_dim=graph_out_size,
            n_classes=n_classes,
            dropout=dropout
        )

        # ============================ #
        #     Fused Feature LSTM
        # ============================ #

        self.lstm_atv = nn.LSTM(input_size=graph_out_size * 3, hidden_size=lstm_hidden_size, num_layers=1, bidirectional=True, dropout=dropout)

        # ============================ #
        #     Fused Feature Graph Learning
        # ============================ #

        self.graph_net_temporal = GraphNetwork(lstm_hidden_size * 2, 2, graph_hidden_size, dropout, self.no_cuda)

        # ============================ #
        #     Classifier
        # ============================ #

        self.smax_fc = nn.Linear(graph_out_size, n_classes) # Classification Head



    def forward(self, inputfeats, qmask, umask, seq_lengths, input_features_mask, adim, tdim, vdim, label=None):

        # ============================ #
        #     Unimodal Feature Extraction
        # ============================ #

        audio_feats = inputfeats[0][:, :, :adim]
        text_feats = inputfeats[0][:, :, adim:adim + tdim]
        video_feats = inputfeats[0][:, :, adim + tdim:]

        # ============================ #
        #     Unimodal, Bi-LSTM Global Information Extraction
        # ============================ #

        audio_out, _ = self.lstm_a(audio_feats)
        text_out, _ = self.lstm_t(text_feats)
        video_out, _ = self.lstm_v(video_feats)

        # ============================ #
        #     Unimodal, Graph Network Local Information Extraction
        # ============================ #

        outputs_mask= input_features_mask[0]  # Input feature mask [Time, Batch, 1]
        outputs_mask = outputs_mask.unsqueeze(2)  # Add dummy dimension for subsequent concatenation [Time, Batch, 1, Dim]

        outputs_a = audio_out.unsqueeze(2)  # -> [Time, Batch, 1, 300]
        outputs_t = text_out.unsqueeze(2)  # -> [Time, Batch, 1, 300]
        outputs_v = video_out.unsqueeze(2)  # -> [Time, Batch, 1, 300]

        # ============================ #
        #     Build a graph for each modality
        # ============================ #

        features_a, edge_index, edge_type, edge_type_mapping, umask_flat, features_speaker_ids, features_node_positions = batch_graphify(
            outputs_a, qmask, seq_lengths,
            self.n_speakers, self.window_past, self.window_future,
            'temporal', self.no_cuda, outputs_mask)
        features_t, edge_index, edge_type, edge_type_mapping, umask_flat, features_speaker_ids, features_node_positions = batch_graphify(
            outputs_t, qmask, seq_lengths,
            self.n_speakers, self.window_past, self.window_future,
            'temporal', self.no_cuda, outputs_mask)
        features_v, edge_index, edge_type, edge_type_mapping, umask_flat, features_speaker_ids, features_node_positions = batch_graphify(
            outputs_v, qmask, seq_lengths,
            self.n_speakers, self.window_past, self.window_future,
            'temporal', self.no_cuda, outputs_mask)
        
        edge_type = torch.zeros_like(edge_type)

        # ============================ #
        #     Calculate Edge Weights
        # ============================ #

        with torch.no_grad(): # Usually edge weight calculation does not involve gradient backprop
            src, dest = edge_index[0], edge_index[1]
            # Calculate temporal weight f_T
            p_src = features_node_positions[src].float()
            p_dest = features_node_positions[dest].float()
            # `features_node_positions` stores temporal position of each node.
            # Efficiently obtain temporal position of all source/target nodes via indices `src` and `dest`.
            temporal_dist_sq = (p_src - p_dest)**2
            # Element-wise calculation of squared difference, getting (p_u - p_v)^2.
            temporal_weight = torch.exp(-temporal_dist_sq / (2 * self.edge_weight_sigma**2))
            # Implement Gaussian kernel formula to get temporal weight for each edge.
            # Calculate speaker consistency weight f_S
            src_speakers = features_speaker_ids[src]
            dest_speakers = features_speaker_ids[dest]
            # Similar to position info, get speaker IDs for all source/target nodes.
            # Create a new tensor with same shape as input_tensor, but initialized to 1.0.
            speaker_consistency_weight = torch.ones_like(src_speakers, dtype=torch.float)
            # First create a tensor of all ones, which is the default weight (corresponds to speaker(u) == speaker(v)).
            speaker_consistency_weight[src_speakers != dest_speakers] = self.edge_weight_beta
            edge_weight = temporal_weight * speaker_consistency_weight
            # New method:
            edge_attr = torch.stack([
                edge_weight
            ], dim=1) # shape: [num_edges, 3]

        # ============================ #
        #     GAT Graph Neural Network Learning
        # ============================ #

        hidden1_a = self.graph_net_temporal_a(
            features_a,
            edge_index,
            edge_type,
            seq_lengths,
            umask,
            edge_attr=edge_attr
        )
        hidden1_t = self.graph_net_temporal_t(
            features_t,
            edge_index,
            edge_type,
            seq_lengths,
            umask,
            edge_attr=edge_attr
        )
        hidden1_v = self.graph_net_temporal_v(
            features_v,
            edge_index,
            edge_type,
            seq_lengths,
            umask,
            edge_attr=edge_attr
        )
        hidden0_a = hidden1_a[0]
        hidden0_t = hidden1_t[0]
        hidden0_v = hidden1_v[0]

        # ============================ #
        #     Normalization
        # ============================ #

        x_prime_a = self.ln_a(hidden0_a)
        x_prime_t = self.ln_t(hidden0_t)
        x_prime_v = self.ln_v(hidden0_v)
        
        # ============================ #
        #     Projection to Shared Dimension
        # ============================ #

        x_prime_a_common = self.proj_to_common_a(x_prime_a)
        x_prime_t_common = self.proj_to_common_t(x_prime_t)
        x_prime_v_common = self.proj_to_common_v(x_prime_v)

        # ============================ #
        #     Dimension Transformation
        # ============================ #

        common_latent_dim = self.flow_hidden_size
        T, B, _ = x_prime_a_common.shape # Get T and B from x_prime_a_common
        # Flatten: [T, B, C] -> [B*T, C]
        # Use permute and contiguous().view() for efficiency
        x_prime_a_flat = x_prime_a_common.permute(1, 0, 2).contiguous().view(-1, common_latent_dim)
        x_prime_t_flat = x_prime_t_common.permute(1, 0, 2).contiguous().view(-1, common_latent_dim)
        x_prime_v_flat = x_prime_v_common.permute(1, 0, 2).contiguous().view(-1, common_latent_dim)
        # Add H=1, W=1 dimensions: [B*T, C] -> [B*T, C, 1, 1]
        x_prime_a_4d = x_prime_a_flat.unsqueeze(-1).unsqueeze(-1)
        x_prime_t_4d = x_prime_t_flat.unsqueeze(-1).unsqueeze(-1)
        x_prime_v_4d = x_prime_v_flat.unsqueeze(-1).unsqueeze(-1)

        # ============================ #
        #     Distribution Transfer
        # ============================ #

        flow_outputs = {}
        # Prepare modality masks
        mask_a = input_features_mask[0][:, :, 0].bool().permute(1, 0)
        mask_t = input_features_mask[0][:, :, 1].bool().permute(1, 0)
        mask_v = input_features_mask[0][:, :, 2].bool().permute(1, 0)

        # Step 1: Pass all data (including generated features for missing modalities) through Glow network
        logdet_a, z_a_out = self.flow_a(x_prime_a_4d)
        logdet_t, z_t_out = self.flow_t(x_prime_t_4d)
        logdet_v, z_v_out = self.flow_v(x_prime_v_4d)

        flat_labels = label.view(-1)
        
        # Classification Task (IEMOCAP, MELD)
        # Labels are indices themselves
        label_indices = flat_labels.long()

        # ============================ #
        #     Pre-calculate means and stds for all priors
        # ============================ #
        
        all_means = []
        all_log_sds = []
        zero_input = torch.zeros(1, self.flow_hidden_size, 1, 1, device=z_a_out.device)
        # This small loop is fixed (2 or 7 times), very fast
        for prior_model in self.priors:
            # mean, log_sd = self.priors[0](zero_input).chunk(2, 1)
            mean, log_sd = prior_model(zero_input).chunk(2, 1)
            all_means.append(mean.squeeze())
            all_log_sds.append(log_sd.squeeze())
        all_means = torch.stack(all_means, dim=0)   # Shape: [num_priors, C]
        all_log_sds = torch.stack(all_log_sds, dim=0) # Shape: [num_priors, C]

        # New method: Squeeze H and W dimensions
        z_a_flat = z_a_out.squeeze(-1).squeeze(-1) # [B*T, 256, 1, 1] -> [B*T, 256]
        z_t_flat = z_t_out.squeeze(-1).squeeze(-1)
        z_v_flat = z_v_out.squeeze(-1).squeeze(-1)

        # --- Create final valid masks ---
        flat_umask = umask.view(-1).bool()
        flat_mask_a = mask_a.T.contiguous().view(-1)
        flat_mask_t = mask_t.T.contiguous().view(-1)
        flat_mask_v = mask_v.T.contiguous().view(-1)
        final_valid_mask_a = flat_umask & flat_mask_a
        final_valid_mask_t = flat_umask & flat_mask_t
        final_valid_mask_v = flat_umask & flat_mask_v
        
        log_p_sum_a, log_p_sum_t, log_p_sum_v = 0.0, 0.0, 0.0
        
        # ============================ #
        #     Calculate Gaussian PDF
        # ============================ #
        # Audio
        if final_valid_mask_a.any():
            valid_z_a = z_a_flat[final_valid_mask_a]
            valid_indices_a = label_indices[final_valid_mask_a]
            batch_means_a = all_means[valid_indices_a]
            batch_log_sds_a = all_log_sds[valid_indices_a]
            # !!! Assign calculation result to log_p_sum_a !!!
            log_p_sum_a = gaussian_log_p(valid_z_a, batch_means_a, batch_log_sds_a).sum()

        # Text
        if final_valid_mask_t.any():
            valid_z_t = z_t_flat[final_valid_mask_t]
            valid_indices_t = label_indices[final_valid_mask_t]
            batch_means_t = all_means[valid_indices_t]
            batch_log_sds_t = all_log_sds[valid_indices_t]
            # !!! Assign calculation result to log_p_sum_t !!!
            log_p_sum_t = gaussian_log_p(valid_z_t, batch_means_t, batch_log_sds_t).sum()
        
        # Video
        if final_valid_mask_v.any():
            valid_z_v = z_v_flat[final_valid_mask_v]
            valid_indices_v = label_indices[final_valid_mask_v]
            batch_means_v = all_means[valid_indices_v]
            batch_log_sds_v = all_log_sds[valid_indices_v]
            # !!! Assign calculation result to log_p_sum_v !!!
            log_p_sum_v = gaussian_log_p(valid_z_v, batch_means_v, batch_log_sds_v).sum()
            
        # --- Combine into final log-likelihood ---
        # Directly use final_valid_mask on [B*T] logdet tensor
        log_p_a_final = 0.0
        if final_valid_mask_a.any(): # Check if any valid audio samples exist
            # Note: logdet_a is [B*T], final_valid_mask_a is also [B*T]
            log_p_a_final = 0.01 * logdet_a[final_valid_mask_a].sum() + log_p_sum_a

        log_p_t_final = 0.0
        if final_valid_mask_t.any():
            log_p_t_final = 0.01 * logdet_t[final_valid_mask_t].sum() + log_p_sum_t

        log_p_v_final = 0.0
        if final_valid_mask_v.any():
            log_p_v_final = 0.01 * logdet_v[final_valid_mask_v].sum() + log_p_sum_v

        # --- Store results in flow_outputs ---
        # Keep only non-zero losses
        if final_valid_mask_a.any():
            flow_outputs['a'] = {'log_p': log_p_a_final}
        if final_valid_mask_t.any():
            flow_outputs['t'] = {'log_p': log_p_t_final}
        if final_valid_mask_v.any():
            flow_outputs['v'] = {'log_p': log_p_v_final}


        # ============================ #
        #     Missing Feature Completion
        # ============================ #

        # Note: final_* variables store pre-projection, modality-specific features
        # final_a, final_t, final_v = x_prime_a, x_prime_t, x_prime_v
        final_a, final_t, final_v = x_prime_a.clone(), x_prime_t.clone(), x_prime_v.clone()
        with torch.no_grad():
            # --- Temporarily reshape z back to [B, C, T, 1] for fusion logic ---
            # Need T and B from original sequence shape
            T_seq, B_seq = final_a.shape[:2]
            C_common = self.flow_hidden_size
            # If needed, recompute z_out, or use stored z_out if gradients are detached
            _, z_a_out_flow = self.flow_a(x_prime_a_4d) # Recalculate z with detached gradients: [B*T, C, 1, 1]
            _, z_t_out_flow = self.flow_t(x_prime_t_4d)
            _, z_v_out_flow = self.flow_v(x_prime_v_4d)
            # Reshape for fusion: [B*T, C, 1, 1] -> [B, T, C] -> [B, C, T, 1]
            z_a_out_seq = z_a_out_flow.squeeze(-1).squeeze(-1).view(B_seq, T_seq, C_common).permute(0, 2, 1).unsqueeze(-1)
            z_t_out_seq = z_t_out_flow.squeeze(-1).squeeze(-1).view(B_seq, T_seq, C_common).permute(0, 2, 1).unsqueeze(-1)
            z_v_out_seq = z_v_out_flow.squeeze(-1).squeeze(-1).view(B_seq, T_seq, C_common).permute(0, 2, 1).unsqueeze(-1)
            # --- Fusion Reshape End ---

        # --- Helper Function 1: [B,C,T,1] -> [B*T, C, 1, 1] (for use with flow_model.reverse)
        def z_seq_to_flat_flow(z_seq, C_common):
            return z_seq.squeeze(-1).permute(0, 2, 1).contiguous().view(-1, C_common).unsqueeze(-1).unsqueeze(-1)

        # --- New Core Reconstruction Function (v3: with mapping layer) ---
        def reconstruct_with_mapping(
            mapping_layer,  # !!! Added: e.g., self.fusion_mapping_a
            flow_model, rec_model, 
            x_intra_seq, z_cross_list_seq, 
            C_common, target_tensor, update_mask):
                        
            # [B*T,C,1,1] -> [B*T,C]
            x_intra_seq_flat = x_intra_seq.squeeze(-1).squeeze(-1)
            concat_list = [x_intra_seq_flat]
            
            # 2. Get "two values obtained by inputting zt, zv into glow-1a respectively"
            for z_cross_seq in z_cross_list_seq:
                z_cross_flat_for_flow = z_seq_to_flat_flow(z_cross_seq, C_common)
                # Step (A): Input to glow-1a
                x_tilde_4d = flow_model.reverse(z_cross_flat_for_flow)
                # Step (B): Get "value" (squeeze H, W dimensions)
                x_tilde_flat = x_tilde_4d.squeeze(-1).squeeze(-1)
                concat_list.append(x_tilde_flat)
                
            # 3. "Get concatenation result of three values"
            # [B*T, C] + [B*T, C] + [B*T, C] -> [B*T, 3*C]
            x_hat_input_flat = torch.cat(concat_list, dim=1)
            
            # 5. --- !!! New Mapping Layer !!! ---
            # This is the step you requested:
            # [B*T, 3*C] -> [B*T, 400]
            x_hat_flat = mapping_layer(x_hat_input_flat)
            
            # 8. Reshape back to sequence [T, B, 512]
            T_seq, B_seq = target_tensor.shape[:2]
            x_hat_seq = x_hat_flat.view(B_seq, T_seq, -1).permute(1, 0, 2)
            
            # 9. Assign using mask
            target_tensor[update_mask.T] = x_hat_seq[update_mask.T]
            return target_tensor

        
        umask_a, umask_t, umask_v = mask_a, mask_t, mask_v

        # ============================ #
        # --- Case 1: Missing Audio ---
        # ============================ #
        mask = (~umask_a) & umask_t & umask_v # [B, T]
        if mask.any():
            final_a = reconstruct_with_mapping(self.fusion_mapping_a, # Pass mapping layer for A
                                               self.flow_a, self.rec_a,
                                               x_prime_a_4d, [z_t_out_seq, z_v_out_seq], # z_a is intra, [z_t, z_v] is cross
                                               C_common, final_a, mask)
            
        # ============================ #
        # --- Case 2: Missing Text ---
        # ============================ #
        mask = umask_a & (~umask_t) & umask_v
        if mask.any():
            final_t = reconstruct_with_mapping(self.fusion_mapping_t, # Pass mapping layer for T
                                               self.flow_t, self.rec_t,
                                               x_prime_t_4d, [z_a_out_seq, z_v_out_seq], # z_t is intra, [z_a, z_v] is cross
                                               C_common, final_t, mask)
        # ============================ #
        # --- Case 3: Missing Video ---
        # ============================ #
        mask = umask_a & umask_t & (~umask_v)
        if mask.any():
            final_v = reconstruct_with_mapping(self.fusion_mapping_v, # Pass mapping layer for V
                                               self.flow_v, self.rec_v,
                                               x_prime_v_4d, [z_a_out_seq, z_t_out_seq], # z_v is intra, [z_a, z_t] is cross
                                               C_common, final_v, mask)
            
        # ============================ #
        # --- Case 4: Missing Audio and Text (Only V exists) ---
        # ============================ #        
        mask = (~umask_a) & (~umask_t) & umask_v
        if mask.any():
            # 1. Recover Audio (A)
            final_a = reconstruct_with_mapping(self.fusion_mapping_a, 
                                               self.flow_a, self.rec_a,
                                               x_prime_a_4d, [z_v_out_seq, z_t_out_seq], # z_a intra, [z_t, z_v] cross
                                               C_common, final_a, mask)
            
            # 2. Recover Text (T)
            final_t = reconstruct_with_mapping(self.fusion_mapping_t, 
                                               self.flow_t, self.rec_t,
                                               x_prime_t_4d, [z_v_out_seq, z_a_out_seq], # z_t intra, [z_a, z_v] cross
                                               C_common, final_t, mask)

        # ============================ #
        # --- Case 5: Missing Audio and Video (Only T exists) ---
        # ============================ #
        mask = (~umask_a) & umask_t & (~umask_v)
        if mask.any():
            # 1. Recover Audio (A)
            final_a = reconstruct_with_mapping(self.fusion_mapping_a, 
                                               self.flow_a, self.rec_a,
                                               x_prime_a_4d, [z_t_out_seq, z_v_out_seq],
                                               C_common, final_a, mask)
            
            # 2. Recover Video (V)
            final_v = reconstruct_with_mapping(self.fusion_mapping_v, 
                                               self.flow_v, self.rec_v,
                                               x_prime_v_4d, [z_t_out_seq, z_a_out_seq],
                                               C_common, final_v, mask)

        # ============================ #
        # --- Case 6: Missing Text and Video (Only A exists) ---
        # ============================ #
        mask = umask_a & (~umask_t) & (~umask_v)
        if mask.any():
            # 1. Recover Text (T)
            final_t = reconstruct_with_mapping(self.fusion_mapping_t, 
                                               self.flow_t, self.rec_t,
                                               x_prime_t_4d, [z_a_out_seq, z_v_out_seq],
                                               C_common, final_t, mask)
            
            # 2. Recover Video (V)
            final_v = reconstruct_with_mapping(self.fusion_mapping_v, 
                                               self.flow_v, self.rec_v,
                                               x_prime_v_4d, [z_a_out_seq, z_t_out_seq],
                                               C_common, final_v, mask)

        # ============================ #
        #     Adaptive Confidence Weighted Fusion
        # ============================ #

        # Feed outputs of three modalities into new module, and receive all return values
        concat_features, modality_logits, modality_confidences = self.confidence_fusion_module(
            final_a, final_t, final_v
        )

        # ============================ #
        #     Joint Feature Bi-LSTM Learning
        # ============================ #

        projected_features, _ = self.lstm_atv(concat_features)
        # Prepare input format for batch_graphify
        outputs = projected_features.unsqueeze(2)  # -> [Time, Batch, 1, 300]

        outputs_mask= input_features_mask[0]  # Input feature mask [Time, Batch, 1]
        outputs_mask = outputs_mask.unsqueeze(2)  # Add dummy dimension for subsequent concatenation [Time, Batch, 1, Dim]

        # ============================ #
        #     Graph Construction
        # ============================ #

        features, edge_index, edge_type, edge_type_mapping, umask_flat, features_speaker_ids, features_node_positions = batch_graphify(
            outputs, qmask, seq_lengths,
            self.n_speakers, self.window_past, self.window_future,
            'temporal', self.no_cuda, outputs_mask)
        edge_type = torch.zeros_like(edge_type)

        # ============================ #
        #     Graph Learning
        # ============================ #

        # # <--- Modification Start: Pass edge_weight to graph network --->
        hidden1 = self.graph_net_temporal(
            features,
            edge_index,
            edge_type,
            seq_lengths,
            umask,
            edge_attr=edge_attr
        )

        hidden0 = hidden1[0]

        # ============================ #
        #     Classification Probability Calculation
        # ============================ #
        
        log_prob = self.smax_fc(hidden0)  # Classification prediction results [seqlen, batch, n_classes]

        x_prime_dict = {'a': x_prime_a, 't': x_prime_t, 'v': x_prime_v}
        recovered_dict = {'a': final_a, 't': final_t, 'v': final_v}

        # Pack final features before fusion
        final_features_dict = {'a': final_a, 't': final_t, 'v': final_v}        

        # Added umask_flat, x_prime_dict, recovered_dict, flow_outputs to return values
        return log_prob, modality_logits, modality_confidences, \
                x_prime_dict, recovered_dict, flow_outputs, hidden0, final_features_dict