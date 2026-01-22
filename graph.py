import numpy as np
import torch


def edge_perms(l, window_past, window_future):
    """
    Function:
        Generates all legal edge pairs for a sequence based on specified time windows (forward & backward) 
        (used for building graph structures).

    Args:
        l             : Sequence length (e.g., number of utterances)
        window_past   : Size of the connection window to the past; -1 means fully connected history
        window_future : Size of the connection window to the future; -1 means fully connected future

    Returns:
        all_perms : A list of all legal edge pairs, where each element is a tuple (source_idx, target_idx)
    """

    all_perms = set()               # Store all legal edge pairs (deduplicated)
    array = np.arange(l)            # Create index array [0, 1, ..., l-1]

    # Construct connectable neighbor nodes for each node j in the sequence
    for j in range(l):  # j: current source node (start point)

        perms = set()   # All legal outgoing edges for current node j

        # Calculate the valid neighbor index range for j (depends on past and future windows)

        if window_past == -1 and window_future == -1:
            # Fully connected (all nodes are connectable)
            eff_array = array

        elif window_past == -1:
            # No limit on past, only limit future, take [0, j+future]
            eff_array = array[:min(l, j + window_future + 1)]

        elif window_future == -1:
            # No limit on future, only limit past, take [j-past, l]
            eff_array = array[max(0, j - window_past):]

        else:
            # Limit both past and future, take [j-past, j+future]
            eff_array = array[max(0, j - window_past):min(l, j + window_future + 1)]

        # Form an edge (j, item) between current node j and its valid neighbor
        for item in eff_array:
            perms.add((j, item))

        # Merge into the total edge set
        all_perms = all_perms.union(perms)

    return list(all_perms)  # Return the list of edge pairs

## Simplified graph building process [single relation: temporal graph]
def batch_graphify(features, qmask, lengths, n_speakers, window_past, window_future, graph_type, no_cuda, input_features_mask):
    """
    (Function documentation remains unchanged)
    """

    # --- ✅ Modification 1: Greatly simplified edge type definitions ---
    # We no longer care about speaker pairing, only three temporal relations
    edge_type_mapping = {
        'past': 0,
        'now': 1,
        'future': 2
    }
    # Removed all old code related to speaker_types
    # Removed assert n_speakers <= 2
    # --- End of Modification ---

    # qmask is now only used for potential other purposes, but is no longer used to determine edge types in this function
    qmask_ = qmask.cpu().data.numpy().astype(int)

    # <--- Modification Start: Added speaker_ids and node_positions lists --->
    node_features, node_features_mask, edge_index, edge_type, features_speaker_ids, features_node_positions = [], [], [], [], [], []
    # <--- Modification End --->
    length_sum = 0
    batch_size = features.size(1)

    for j in range(batch_size):
        # (This part of the code remains unchanged)
        node_feature = features[:lengths[j], j, :, :]
        node_feature = torch.reshape(node_feature, (-1, node_feature.size(-1)))
        node_features.append(node_feature)

        node_feature_mask = input_features_mask[:lengths[j], j, :]
        node_feature_mask = torch.reshape(node_feature_mask, (-1, node_feature_mask.size(-1)))
        node_features_mask.append(node_feature_mask)

        # <--- Modification Start: Record speaker ID and position for each node in the current dialogue --->
        current_qmask = qmask_[j][:lengths[j]]
        features_speaker_ids.append(torch.LongTensor(current_qmask))
        features_node_positions.append(torch.arange(0, lengths[j], dtype=torch.long))
        # <--- Modification End --->
        
        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(src + length_sum, tgt + length_sum) for (src, tgt) in perms1]
        length_sum += lengths[j]
        
        for (local_src, local_tgt), (global_src, global_tgt) in zip(perms1, perms2):
            edge_index.append([global_src, global_tgt])

            # --- ✅ Modification 2: Enforce using temporal relations to define edge types ---
            # Determine temporal order
            if local_tgt > local_src:
                order_type = 'past'
            elif local_tgt == local_src:
                order_type = 'now'
            else:
                order_type = 'future'

            # Regardless of what graph_type is, we use order_type as the edge type
            edge_type_name = order_type
            
            # Removed all old code related to speaker_type
            # --- End of Modification ---
            
            edge_type.append(edge_type_mapping[edge_type_name])
            
    # (Code for concatenation and moving to GPU at the end of the function remains unchanged)
    node_features = torch.cat(node_features, dim=0)
    node_features_mask = torch.cat(node_features_mask, dim=0)
    edge_index = torch.tensor(edge_index).transpose(0, 1)
    edge_type = torch.tensor(edge_type)
    # <--- Modification Start: Concatenate and convert newly returned tensors --->
    features_speaker_ids = torch.cat(features_speaker_ids, dim=0)
    features_node_positions = torch.cat(features_node_positions, dim=0)
    # <--- Modification End --->
    
    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_type = edge_type.cuda()
        # <--- Modification Start: Move new tensors to GPU as well --->
        node_features_mask = node_features_mask.cuda()
        features_speaker_ids = features_speaker_ids.cuda()
        features_node_positions = features_node_positions.cuda()
        # <--- Modification End --->

    # <--- Modification Start: Update return values --->
    return node_features, edge_index, edge_type, edge_type_mapping, node_features_mask, features_speaker_ids, features_node_positions
    # <--- Modification End --->



def batch_graphify_dynamic(outputs, qmask, seq_lengths, n_speakers, 
                           past_threshold, future_threshold, no_cuda=False, outputs_mask=None):
    """
    Dynamically build the graph based on cumulative edge weight thresholds.
    (V4: Corrected based on the dimension order [Batch, Time] of qmask)
    """
    device = outputs.device
    
    # The dimension order of outputs is [Time, Batch, ...]
    time_size, batch_size, _, feature_dim = outputs.size()

    # Force convert seq_lengths to LongTensor on the correct device
    if isinstance(seq_lengths, list):
        seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long, device=device)
    else:
        seq_lengths_tensor = seq_lengths.to(device=device, dtype=torch.long)

    node_features_list = []
    valid_speaker_ids_list = []

    # Iterate through every dialogue in the batch (j is the batch index)
    for j in range(batch_size):
        true_len = seq_lengths_tensor[j].item()
        if true_len == 0:
            continue

        # 1. Extract features: outputs is [Time, Batch, ...], so j is in the second dimension
        node_feature = outputs[:true_len, j, 0, :]
        node_features_list.append(node_feature)

        # ==================== Core Code Correction ====================
        # 2. Extract Speaker IDs: qmask is [Batch, Time], so j is in the first dimension
        # Old incorrect slicing method: valid_ids = qmask[:true_len, j]
        valid_ids = qmask[j, :true_len]
        # ====================================================
        valid_speaker_ids_list.append(valid_ids)

    # If the entire batch consists of empty sequences, return an empty graph early
    if not node_features_list:
        features = torch.empty((0, feature_dim), device=device)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weight = torch.empty((0, 1), dtype=torch.float32, device=device)
        edge_type = torch.empty((0,), dtype=torch.long, device=device)
        return (features, edge_index, edge_type, {}, None, None, None, edge_weight)

    # Subsequent concatenation logic remains unchanged
    features = torch.cat(node_features_list, dim=0)
    speaker_ids_all = torch.cat(valid_speaker_ids_list, dim=0)
    
    # Dynamic edge construction logic remains unchanged
    all_edges = []
    all_weights = []
    sigma = 5.0
    length_sum = 0

    for j in range(batch_size):
        conv_len = seq_lengths_tensor[j].item()
        if conv_len == 0:
            continue
        
        speaker_ids_local = speaker_ids_all[length_sum : length_sum + conv_len]

        for u_local in range(conv_len):
            u_global = u_local + length_sum
            speaker_u = speaker_ids_local[u_local]
            
            # --- Search past context ---
            past_weight_sum = 0.0
            for v_local in range(u_local, -1, -1):
                v_global = v_local + length_sum
                speaker_v = speaker_ids_local[v_local]
                
                speaker_weight = 1.0 if speaker_u.item() == speaker_v.item() else 0.5
                dist_sq = (u_local - v_local)**2
                temporal_weight = torch.exp(-torch.tensor(dist_sq, dtype=torch.float32, device=device) / (2 * sigma**2))
                final_weight = speaker_weight * temporal_weight
                
                if (past_weight_sum + final_weight > past_threshold) and (u_local != v_local):
                    break
                
                past_weight_sum += final_weight
                all_edges.append([v_global, u_global])
                all_weights.append(final_weight)

            # --- Search future context ---
            self_loop_weight = all_weights[-1]
            future_weight_sum = self_loop_weight
            
            for v_local in range(u_local + 1, conv_len):
                v_global = v_local + length_sum
                speaker_v = speaker_ids_local[v_local]
                
                speaker_weight = 1.0 if speaker_u.item() == speaker_v.item() else 0.5
                dist_sq = (v_local - u_local)**2
                temporal_weight = torch.exp(-torch.tensor(dist_sq, dtype=torch.float32, device=device) / (2 * sigma**2))
                final_weight = speaker_weight * temporal_weight
                
                if future_weight_sum + final_weight > future_threshold:
                    break

                future_weight_sum += final_weight
                all_edges.append([v_global, u_global])
                all_weights.append(final_weight)
        
        length_sum += conv_len

    # --- Ending processing logic remains unchanged ---
    if not all_edges:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weight = torch.empty((0, 1), dtype=torch.float32, device=device)
    else:
        edge_index = torch.tensor(all_edges, dtype=torch.long, device=device).t().contiguous()
        all_weights_tensor = torch.stack(all_weights) if isinstance(all_weights[0], torch.Tensor) else torch.tensor(all_weights, dtype=torch.float32, device=device)
        edge_weight = all_weights_tensor.view(-1, 1)

    edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
    
    return (features, edge_index, edge_type, {}, None, None, None, edge_weight)