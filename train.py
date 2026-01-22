# Built-in libraries
import os                  # Operating system related functions, such as file paths, environment variables, etc.
import time                # Time module, used for timing, getting current time, etc.
import math                # Math library
import argparse            # Command line argument parsing
from utils import Logger, update_best_model, get_loaders, build_model, generate_inputs, random_mask, save_results

import datetime  # Import datetime module, used for handling date and time

# Third-party numerical calculation library
import numpy as np                                 # Numerical calculation library

# PyTorch related libraries
import torch                                        # PyTorch main library
import torch.optim as optim                         # Optimizer module
import torch.nn.functional as F

# Machine learning related functions
from sklearn.metrics import f1_score, accuracy_score        # Metrics for evaluating model performance: F1 score and accuracy

# Local module imports (custom modules)
import path                             # Custom path management module (usually used for path constant definitions)
from loss import MaskedCELoss, MaskedMSELoss # Import custom loss functions

import sys
import copy


def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader, 
                        mask_rate=None, optimizer=None, train=False):
    # ============================ #
    #       Initialize variables   #
    # ============================ #

    preds, masks, labels, vidnames = [], [], [], []
    preds_mod0, preds_mod1, preds_mod2 = [], [], []
    savepreds, savelabels, savespeakers, savefmask = [], [], [], []
    losses, losses1, losses_cdt, losses_rec, truth_losses = [], [], [], [], []
    all_confidences_mod0, all_confidences_mod1, all_confidences_mod2 = [], [], []
    all_hidden_features = []
    all_final_feats = {'a': [], 't': [], 'v': []}

    # Get current configuration parameters
    dataset = args.dataset                  # Dataset name
    lower_bound = args.lower_bound          # Whether to enable the lower bound model (for comparative experiments)
    cuda = torch.cuda.is_available() and not args.no_cuda  # Determine whether to enable GPU

    # If in training mode, an optimizer must be passed
    assert not train or optimizer != None

    # Model mode switch
    if train:
        model.train()  # Enable training mode (enable dropout, BN, etc.)
    else:
        model.eval()   # Enable evaluation mode (disable dropout, BN fixed means)

    # ============================ #
    #   Training/Testing process start #
    # ============================ #

    # Iterate through each batch of data
    for data in dataloader:
        if train:
            optimizer.zero_grad()  # Clear gradient cache

        # ============================ #
        #       Read data              #
        # ============================ #

        # data[0~5]: Modality features (host/guest) audio/text/visual, shape: [seqlen, batch, dim]
        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
        # data[6]: qmask -> speaker mask [batch, seqlen], MELD dataset is special, it's 3D
        # data[7]: umask -> valid utterance mask [batch, seqlen]
        # data[8]: label [batch, seqlen]
        speaker_info, umask, label = data[6], data[7], data[8]
        # data[-1]: list of video names
        # vidnames += data[-1]
        batch_dialogue_names = data[-1]
        # ============================ #
        # Determine qmask and qmask_9, i.e., determine who said each sentence
        # For dataset IEMOCAP, containing two speakers, host stores speaker 1's modality features, guest stores speaker 2's modality features
        # For dataset MELD, there are 9 speakers. For convenience, host stores features of all 9 people, qmask is all 0, create an extra qmask9 to store speaker info
        # ============================ #

        if args.dataset == 'MELD':
            # If it is the MELD dataset, speaker_info is 3D features, we need to create a 2D qmask
            # Our strategy is to always choose Host, so we create a qmask of all zeros
            batch_size = speaker_info.shape[0]
            seq_len = speaker_info.shape[1]
            qmask = torch.zeros(batch_size, seq_len, dtype=torch.long)
            # 2. New: As per your request, generate an extra qmask_9
            #    Used to store the real speaker indices from 0-8
            # speaker_info shape is [B, T, 9] (one-hot)
            # We use argmax along the last dimension (dim=2) to get the index
            # qmask_9 shape will be [B, T], values 0 to 8
            qmask_9 = torch.argmax(speaker_info, dim=2).long()
        else:
            # If it is IEMOCAP or other datasets, speaker_info is the 2D qmask we need
            # Just use it directly
            qmask = speaker_info

        # ============================ #
        #  Get dimensions of each modality #
        # ============================ #

        adim = audio_host.size(2)
        tdim = text_host.size(2)
        vdim = visual_host.size(2)

        # ============================ #
        #   Generate missing masks     #
        # ============================ #
        """
        Randomly generate view missing masks: ensure at least one modality exists for each sample.
        Generate three-modality mask matrices for host and guest respectively (0=missing, 1=retained).
        """
        seqlen = audio_host.size(0)
        batch = audio_host.size(1)
        view_num = 3

        # -------- Host mask -------- #
        matrix = random_mask(view_num, seqlen * batch, mask_rate)  # Generate [seqlen*batch, view_num] mask matrix
        audio_host_mask = np.reshape(matrix[:, 0], (seqlen, batch, 1)) 
        text_host_mask = np.reshape(matrix[:, 1], (seqlen, batch, 1))
        visual_host_mask = np.reshape(matrix[:, 2], (seqlen, batch, 1))
        audio_host_mask = torch.LongTensor(audio_host_mask)
        text_host_mask = torch.LongTensor(text_host_mask)
        visual_host_mask = torch.LongTensor(visual_host_mask)

        # -------- Guest mask -------- #
        matrix = random_mask(view_num, seqlen * batch, mask_rate)
        audio_guest_mask = np.reshape(matrix[:, 0], (seqlen, batch, 1))
        text_guest_mask = np.reshape(matrix[:, 1], (seqlen, batch, 1))
        visual_guest_mask = np.reshape(matrix[:, 2], (seqlen, batch, 1))
        audio_guest_mask = torch.LongTensor(audio_guest_mask)
        text_guest_mask = torch.LongTensor(text_guest_mask)
        visual_guest_mask = torch.LongTensor(visual_guest_mask)

        # Safety check: ensure each sample in random mask retains at least one modality (corresponding to CPM-Net settings)
        if view_num == 2:
            assert mask_rate <= 0.500001, f'Warning: at least one view exists'
        if view_num == 3:
            assert mask_rate <= 0.700001, f'Warning: at least one view exists'

        # ============================ #
        # Generate missing modality features using masks #
        # ============================ #

        if not lower_bound:
            # Standard way: mask based on independent mask of each modality
            masked_audio_host = audio_host * audio_host_mask
            masked_audio_guest = audio_guest * audio_guest_mask
            masked_text_host = text_host * text_host_mask
            masked_text_guest = text_guest * text_guest_mask
            masked_visual_host = visual_host * visual_host_mask
            masked_visual_guest = visual_guest * visual_guest_mask
        else:
            # Lower Bound mode: only retain samples where all three modalities exist simultaneously (stricter)
            host_mask = torch.logical_and(
                            torch.logical_and(audio_host_mask, text_host_mask),
                            visual_host_mask).int()  # [seqlen, batch, 1]
            masked_audio_host = audio_host * host_mask
            masked_text_host = text_host * host_mask
            masked_visual_host = visual_host * host_mask
            audio_host_mask = text_host_mask = visual_host_mask = host_mask

            guest_mask = torch.logical_and(
                            torch.logical_and(audio_guest_mask, text_guest_mask),
                            visual_guest_mask).int()
            masked_audio_guest = audio_guest * guest_mask
            masked_text_guest = text_guest * guest_mask
            masked_visual_guest = visual_guest * guest_mask
            audio_guest_mask = text_guest_mask = visual_guest_mask = guest_mask

        # ============================ #
        #      Move data to GPU        #
        # ============================ #

        if cuda:
            # Original input features to GPU
            audio_host = audio_host.cuda()
            text_host = text_host.cuda()
            visual_host = visual_host.cuda()
            audio_guest = audio_guest.cuda()
            text_guest = text_guest.cuda()
            visual_guest = visual_guest.cuda()

            # Masked features and masks to GPU
            masked_audio_host, audio_host_mask = masked_audio_host.cuda(), audio_host_mask.cuda()
            masked_text_host, text_host_mask = masked_text_host.cuda(), text_host_mask.cuda()
            masked_visual_host, visual_host_mask = masked_visual_host.cuda(), visual_host_mask.cuda()
            masked_audio_guest, audio_guest_mask = masked_audio_guest.cuda(), audio_guest_mask.cuda()
            masked_text_guest, text_guest_mask = masked_text_guest.cuda(), text_guest_mask.cuda()
            masked_visual_guest, visual_guest_mask = masked_visual_guest.cuda(), visual_guest_mask.cuda()

            # Speaker mask, valid utterance mask, label to GPU
            qmask = qmask.cuda()
            umask = umask.cuda()
            label = label.cuda()

        # ============================ #
        # Calculate the length of each dialogue (total number of utterances)
        # ============================ #

        lengths = []
        for j in range(len(umask)):  # Iterate through each dialogue in the batch
            length = (umask[j] == 1).nonzero().tolist()[-1][0] + 1
            lengths.append(length)

        # ============================ #
        #   Multimodal data aggregation
        # ============================ #

        # Original data without missing values
        input_features = generate_inputs(
            audio_host, text_host, visual_host,
            audio_guest, text_guest, visual_guest, qmask)
        
        audio_feats = input_features[0][:, :, :adim]     # [T, B, adim]
        text_feats = input_features[0][:, :, adim:adim+tdim] # [T, B, tdim]
        video_feats = input_features[0][:, :, adim+tdim:] # [T, B, vdim]

        # Mask matrix
        input_features_mask = generate_inputs(
            audio_host_mask, text_host_mask, visual_host_mask,
            audio_guest_mask, text_guest_mask, visual_guest_mask, qmask)

        # Data with missing values after masking
        masked_input_features = generate_inputs(
            masked_audio_host, masked_text_host, masked_visual_host,
            masked_audio_guest, masked_text_guest, masked_visual_guest, qmask)

        # ============================ #
        #    Model forward propagation
        # ============================ #
        if args.dataset == 'MELD':
            log_prob, modality_logits, modality_confidences, \
            x_prime_dict, recovered_dict, flow_outputs, hidden0, final_features_dict = model(
                masked_input_features, qmask_9, umask, lengths, input_features_mask, adim, tdim, vdim, label
            )
        else:
            log_prob, modality_logits, modality_confidences, \
            x_prime_dict, recovered_dict, flow_outputs, hidden0, final_features_dict = model(
                masked_input_features, qmask, umask, lengths, input_features_mask, adim, tdim, vdim, label
            )

        # ============================ #
        #  Save intermediate prediction results
        # ============================ #

        tempseqlen = np.sum(umask.cpu().data.numpy(), 1)          # [batch] Actual number of utterances per dialogue
        temppred = log_prob.transpose(0, 1).cpu().data.numpy()    # [batch, seqlen, num_classes]
        templabel = label.cpu().data.numpy()                      # [batch, seqlen]
        tempqmask = qmask.cpu().data.numpy()                      # [batch, seqlen]
        tempfmask = input_features_mask[0].transpose(0, 1).cpu().data.numpy()  # [batch, seqlen, 3]

        for ii in range(len(tempseqlen)):
            # Get the real length of the current dialogue
            current_len = int(tempseqlen[ii])
            
            # Core fix: Expand names here!
            # Get the ID of the dialogue (e.g. 'Ses01F_impro01' or MELD ID)
            dia_id = batch_dialogue_names[ii]
            # Generate unique name for each utterance in the dialogue (e.g. 'Ses01F_impro01_0', 'Ses01F_impro01_1')
            # Thus the length of vidnames will equal the sum of all utterances
            expanded_names = [f"{dia_id}_{k}" for k in range(current_len)]
            vidnames.extend(expanded_names)

            # Keep the following as is
            itempred = temppred[ii][:current_len, :]
            itemfmask = tempfmask[ii][:current_len, :]
            itemlabel = templabel[ii][:current_len]
            itemspks = tempqmask[ii][:current_len]
            
            savepreds.append(itempred)
            savefmask.append(itemfmask)
            savelabels.append(itemlabel)
            savespeakers.append(itemspks)

        # ============================ #
        #       Loss calculation
        # ============================ #

        # ============================ #
        #      Loss 1: Classification loss
        # ============================ #

        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size(2))  # [batch*seq_len, num_classes]
        labels_ = label.view(-1)                                                # [batch*seq_len]

        # Classification loss (multi-class or regression) determined by dataset
        if dataset in ['IEMOCAPFour', 'IEMOCAPSix', 'MELD']: # <-- Add 'MELD' to the list
            loss1 = cls_loss(lp_, labels_, umask)  # Classification cross entropy
        loss = loss1

        # ============================ #
        # Loss 2: Confidence loss + Unimodal classification loss
        # ============================ #

        truth_loss = 0
        num_modalities = len(modality_logits)

        for i in range(num_modalities):
            logits = modality_logits[i]      # Shape [Time*Batch, n_classes]
            confidences = modality_confidences[i].view(-1)

            if i == 0:
                all_confidences_mod0.append(confidences.cpu().detach().numpy())
            elif i == 1:
                all_confidences_mod1.append(confidences.cpu().detach().numpy())
            elif i == 2:
                all_confidences_mod2.append(confidences.cpu().detach().numpy())

            # # Print statistics of confidence scores for each modality
            # print(f"  - Confidence score (modality {i}) -> "
            #       f"Mean: {torch.mean(confidences).item():.4f}, "
            #       f"Std: {torch.std(confidences).item():.4f}, "
            #       f"Min: {torch.min(confidences).item():.4f}, "
            #       f"Max: {torch.max(confidences).item():.4f}")

            # Scene 1: Classification task (IEMOCAP, MELD)
            if dataset in ['IEMOCAPFour', 'IEMOCAPSix', 'MELD']:
                # 1. Calculate classification loss for this modality
                loss_class = cls_loss(logits, labels_, umask)
                # 2. Calculate confidence target p_target (use softmax + gather)
                with torch.no_grad():
                    pred_probs = F.softmax(logits, dim=1)
                    p_target = torch.gather(pred_probs, dim=1, index=labels_.long().unsqueeze(dim=1)).view(-1)
            
            # 3. Calculate confidence loss (uniformly use regression loss)
            loss_conf = reg_loss(confidences, p_target, umask)

            # print(f"  - Auxiliary loss (modality {i}) -> Confidence loss: {loss_conf.item():.4f}, Classification/Regression loss: {loss_class.item():.4f}")
            
            # 4. Accumulate total auxiliary loss
            truth_loss += (loss_conf + 0.1 * loss_class)

        # ============================ #
        #      Loss 3: cdt: Normalizing flow class normalization loss
        #      Loss 4: Missing feature reconstruction loss
        # ============================ #

        # Initialize new losses
        loss_cdt = torch.tensor(0.0, device=loss.device)
        loss_rec = torch.tensor(0.0, device=loss.device)
        # These two losses are calculated and backpropagated only in training mode
        # Calculate L_cdt (Distribution consistency loss)
        for modal_key, outputs in flow_outputs.items():
            # outputs['log_p'] is now the complete log p(x) = log p(z) + logdet
            log_likelihood = outputs['log_p']
            
            n_pixels = x_prime_dict[modal_key].numel() # Get total pixels/features count for normalization
            loss_cdt += -log_likelihood / (math.log(2) * n_pixels)

        loss_rec = torch.tensor(0.0, device=loss.device if cuda else 'cpu') # Initialize loss

        # if train: # Calculate loss_rec only during training
        # Original input feature dictionary (audio_feats etc. defined earlier)
        original_input_dict = {'a': audio_feats, 't': text_feats, 'v': video_feats}
        # Projection layer dictionary (accessed via model)
        proj_rec_layers = {'a': model.rec_a, 't': model.rec_t, 'v': model.rec_v}

        T_seq, B_seq = audio_feats.shape[:2] # Get T, B from original input
        umask_bool_TB = umask.bool().T      # Valid utterance mask [T, B]

        # input_features_mask[0] shape is [T, B, 3] (Tensor with 0s/1s)
        input_mask_tensor_TB3 = input_features_mask[0].bool() # Convert to boolean [T, B, 3]

        # Extract existence masks for A, T, V respectively, shape [T, B]
        mask_a_TB = input_mask_tensor_TB3[:, :, 0]
        mask_t_TB = input_mask_tensor_TB3[:, :, 1]
        mask_v_TB = input_mask_tensor_TB3[:, :, 2]
        mod_masks_TB = [mask_a_TB, mask_t_TB, mask_v_TB] # Store as list

        modal_keys = ['a', 't', 'v']

        for i, key in enumerate(modal_keys):
            mod_mask_TB = mod_masks_TB[i] # Current modality's *existence* mask [T, B]
            # Find positions where original modality is *missing* (~mod_mask) and utterance is *valid* (umask)
            missing_mask_TB = (~mod_mask_TB) & umask_bool_TB # Shape [T, B]

            if missing_mask_TB.any(): # If current modality is missing in valid utterances
                recovered_features_TB = recovered_dict[key]       # [T, B, 512]
                original_features_TB = original_input_dict[key] # [T, B, adim/tdim/vdim]
                proj_layer = proj_rec_layers[key]

                # --- Project recovered features back to original dimensions ---
                recovered_to_project = recovered_features_TB[missing_mask_TB] # [N_missing, 512]
                recovered_to_project = recovered_to_project.unsqueeze(-1)
                projected_recovered = proj_layer(recovered_to_project)       # [N_missing, adim/tdim/vdim,1]
                projected_recovered = projected_recovered.squeeze(-1)
                # --- End of projection ---

                # --- Get corresponding original features as targets ---
                original_target = original_features_TB[missing_mask_TB] # [N_missing, adim/tdim/vdim]
                # --- End of acquisition ---

                # Calculate MSE loss
                loss_rec += F.mse_loss(
                    projected_recovered,
                    original_target.detach()
                )

        # ============================ #
        #       Total loss summary
        # ============================ #

        loss = loss + truth_loss + args.lambda_cdt * loss_cdt + args.lambda_rec * loss_rec

        # ============================ #
        #       Save batch results
        # ============================ #

        preds.append(lp_.data.cpu().numpy())        # All prediction results [batch*seqlen, num_classes]
        labels.append(labels_.data.cpu().numpy())   # All true labels [batch*seqlen]
        masks.append(umask.view(-1).cpu().numpy())  # Valid utterance mask

        # Multiply loss by number of valid samples to facilitate subsequent weighted averaging
        losses.append(loss.item() * masks[-1].sum())
        losses1.append(loss1.item() * masks[-1].sum())
        losses_cdt.append(loss_cdt.item() * masks[-1].sum())
        losses_rec.append(loss_rec.item() * masks[-1].sum())
        truth_losses.append(truth_loss.item() * masks[-1].sum())

        # Save prediction results for each modality
        for i in range(num_modalities):
            # Process logits for each modality just like log_prob
            mod_lp_ = modality_logits[i]
            if i == 0:
                preds_mod0.append(mod_lp_.data.cpu().numpy())
            elif i == 1:
                preds_mod1.append(mod_lp_.data.cpu().numpy())
            elif i == 2:
                preds_mod2.append(mod_lp_.data.cpu().numpy())

        # ============================ #
        #     Classification feature visualization
        # ============================ #
        # Flatten hidden0 and filter based on umask
        hidden0_flat = hidden0.transpose(0, 1).contiguous().view(-1, hidden0.size(2))
        valid_hidden_features = hidden0_flat.data.cpu().numpy()[masks[-1].astype(bool)]
        all_hidden_features.append(valid_hidden_features)

        # ============================ #
        #     Collect features of different modalities
        # ============================ #
        # final_features_dict['a'] shape is [Time, Batch, Dim]
        # masks[-1] is current batch's umask (flattened numpy)
        mask_bool = masks[-1].astype(bool)
        
        for modal in ['a', 't', 'v']:
            # 1. Get features [Time, Batch, Dim]
            feat_tensor = final_features_dict[modal]
            # 2. Convert dimensions [Time, Batch, Dim] -> [Batch*Time, Dim] (align with umask)
            feat_flat = feat_tensor.transpose(0, 1).contiguous().view(-1, feat_tensor.size(2))
            # 3. Convert to numpy
            feat_np = feat_flat.data.cpu().numpy()
            # 4. Keep only valid utterances (where umask=1)
            valid_feat = feat_np[mask_bool]
            # 5. Add to list
            all_final_feats[modal].append(valid_feat)

        # ============================ #
        #       Training mode: Backpropagation
        # ============================ #

        if train:
            loss.backward()      # Backward propagation to calculate gradients 
            optimizer.step()     # Update model parameters


    # ============================ #
    #   Ensure prediction results were collected
    # ============================ #

    assert preds != [], f'Error: no dataset in dataloader'

    # ============================ #
    #       Concatenate results from each batch
    # ============================ #
    preds  = np.concatenate(preds)   # [batch*seqlen, num_classes] or [batch*seqlen]
    labels = np.concatenate(labels)  # [batch*seqlen]
    masks  = np.concatenate(masks)   # [batch*seqlen], Indicates whether this position is valid (is 1)
    
    preds_mod0 = np.concatenate(preds_mod0)
    preds_mod1 = np.concatenate(preds_mod1)
    preds_mod2 = np.concatenate(preds_mod2)
    all_modality_preds = [preds_mod0, preds_mod1, preds_mod2]

    avg_conf_mod0 = np.mean(np.concatenate(all_confidences_mod0)) if all_confidences_mod0 else 0
    avg_conf_mod1 = np.mean(np.concatenate(all_confidences_mod1)) if all_confidences_mod1 else 0
    avg_conf_mod2 = np.mean(np.concatenate(all_confidences_mod2)) if all_confidences_mod2 else 0

    if dataset in ['IEMOCAPFour', 'IEMOCAPSix', 'MELD']:
        preds = np.argmax(preds, 1)  # Convert predicted probabilities to class labels (take class with max probability)

        # Calculate weighted average loss, using valid sample count as weight
        avg_loss = round(np.sum(losses) / np.sum(masks), 4) # Total loss divided by total valid count
        avg_loss1 = round(np.sum(losses1) / np.sum(masks), 4) # Classification loss
        avg_loss_cdt = round(np.sum(losses_cdt) / np.sum(masks), 4)
        avg_loss_rec = round(np.sum(losses_rec) / np.sum(masks), 4)
        avg_truth_losses = round(np.sum(truth_losses) / np.sum(masks), 4)

        # Calculate classification accuracy (supports weighting with mask)
        avg_accuracy = accuracy_score(labels, preds, sample_weight=masks)
        # Calculate weighted F1 score (considering sample imbalance)
        avg_fscore = f1_score(labels, preds, sample_weight=masks, average='weighted')

        # Then loop to calculate metrics for each independent modality
        modality_accuracies = []
        modality_fscores = []
        for p_mod in all_modality_preds:
            p_mod_labels = np.argmax(p_mod, 1)
            acc = accuracy_score(labels, p_mod_labels, sample_weight=masks)
            f1 = f1_score(labels, p_mod_labels, sample_weight=masks, average='weighted')
            modality_accuracies.append(acc)
            modality_fscores.append(f1)

    # ============================ #
    #       Output evaluation information
    # ============================ #
    print(f'sample number: {np.sum(masks)}')  # Print total number of valid samples

    # Add new metrics to return values
    per_modality_metrics = {
        'acc': modality_accuracies,
        'f1': modality_fscores
    }
    concatenated_masks = masks

    # ============================ #
    #   Concatenate features of three different modalities
    # ============================ #
    # Concatenate lists into large arrays before return
    final_feats_concatenated = {
        'a': np.concatenate(all_final_feats['a'], axis=0),
        't': np.concatenate(all_final_feats['t'], axis=0),
        'v': np.concatenate(all_final_feats['v'], axis=0)
    }

    return avg_accuracy, avg_fscore, vidnames, \
           [avg_loss, avg_loss1, avg_truth_losses, avg_loss_cdt, avg_loss_rec], \
           [savepreds, savelabels, savespeakers, savefmask, concatenated_masks], \
           [avg_conf_mod0, avg_conf_mod1, avg_conf_mod2], \
           per_modality_metrics, all_hidden_features, final_feats_concatenated


if __name__ == '__main__':

    # ============================ #
    #   Create Argument Parser     #
    # ============================ #

    parser = argparse.ArgumentParser()

    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT', help='Audio feature name (optional)')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT', help='Text feature name (optional)')
    parser.add_argument('--video-feature', type=str, default='manet_UTT', help='Video feature name (optional)')
    parser.add_argument('--dataset', type=str, default='MELD', help='Dataset name to use, e.g., IEMOCAPFour, IEMOCAPSix, MELD')

    parser.add_argument('--windowp', type=int, default=3,help='Forward time window size for graph construction (-1 for fully connected)')
    parser.add_argument('--windowf', type=int, default=3,help='Backward time window size for graph construction (-1 for fully connected)')
    parser.add_argument('--lstm_hidden_size', type=int, default=150)
    parser.add_argument('--graph_hidden_size', type=int, default=100)
    parser.add_argument('--flow_hidden_size', type=int, default=128)
    parser.add_argument('--n_classes', type=int, default=2,help='Number of classes for the classification task (determined by dataset)')
    parser.add_argument('--n_classes_priors', type=int, default=2, 
                        help='Number of prior distributions for loss_cdt, usually used to discretize regression problems')
    parser.add_argument('--n_speakers', type=int, default=2,help='Number of speakers in the conversation (usually 2)')

    parser.add_argument('--no-cuda', action='store_true', default=False,help='Disable GPU, force CPU training')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='Learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',help='L2 regularization coefficient')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout',help='Dropout ratio')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',help='Number of training samples per batch')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',help='Total number of training epochs')
    parser.add_argument('--num-folder', type=int, default=5,help='Number of folds for cross-validation (e.g., IEMOCAP uses 5 folds)')
    parser.add_argument('--mask-type', type=str, default='constant-0.1',help='Input modality missing strategy: constant-float; linear; convex; concave')

    parser.add_argument('--lower-bound', action='store_true', default=False,help='Whether to use Lower Bound mode (remove missing modalities during training)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')  # Specify the GPU index to use

    parser.add_argument('--lambda_cdt', type=float, default=1.0, help='Weight for the Distribution Consistency Loss (L_cdt)')
    parser.add_argument('--lambda_rec', type=float, default=0.3, help='Weight for the Contextual Reconstruction Loss (L_rec)')

    parser.add_argument('--early-stop-patience', type=int, default=20, 
                        help='Patience for early stopping. Stops if val F1 score does not improve for this many epochs.')
    parser.add_argument('--early-stop-delta', type=float, default=0.001, 
                        help='Minimum change in val F1 score to be considered an improvement.')

    parser.add_argument('--n_flow', type=int, default=1, 
                        help='Number of Flow steps in each Glow Block')
    parser.add_argument('--num_blocks_rec', type=int, default=5, 
                        help='Number of Group blocks in each RCAN reconstructor (self.rec_*)')
    parser.add_argument('--reduction_rec', type=int, default=16,
                        help='Reduction ratio for Channel Attention (CALayer) in RCAN Group')
    parser.add_argument('--n_block_glow', type=int, default=1,
                        help='Number of Blocks in Glow model (should be 1 for single-scale models)')
    parser.add_argument('--seed', type=int, default=50,help='Random seed to ensure consistency of splits')

    # Parse arguments
    args = parser.parse_args()

    print(f"Setting random seed to: {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cuda = torch.cuda.is_available() and not args.no_cuda  # Determine whether to enable GPU
    if cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) # if use multi-GPU

    # ============================ #
    #   GPU Device Selection       #
    # ============================ #

    # Whether to enable CUDA training (device auto-detection + user parameter control)
    cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')  # Set device, use GPU if CUDA is available, otherwise CPU
    args.device = device  # Store device information in args object

    # ============================ #
    #   Set Log File Save Path     #
    # ============================ #

    save_folder_name = f'{args.dataset}'  # Create save folder name based on dataset type
    save_log = os.path.join(path.LOG_DIR, 'main_result', f'{save_folder_name}')  # Set folder path to save logs
    if not os.path.exists(save_log): os.makedirs(save_log)  # Create folder if it doesn't exist
    time_dataset = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}_{args.dataset}"  # Get current time and create timestamp to distinguish different experiments
    sys.stdout = Logger(filename=f"{save_log}/{time_dataset}_batchsize-{args.batch_size}_lr-{args.lr}_masktype-{args.mask_type}.txt", stream=sys.stdout)  # Set log file path to save output information during model training

    # ============================ #
    #   Dataset Basic Settings     #
    # ============================ #

    if args.dataset == 'IEMOCAPFour':
        args.num_folder = 5         # Use 5-fold cross-validation
        args.n_classes = 4          # 4-class classification task
        args.n_speakers = 2         # Two-person conversation
        args.n_classes_priors = 4 # loss_cdt uses 2 priors: positive/negative sentiment
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder = 5         # Use 5-fold cross-validation
        args.n_classes = 6          # 6-class classification task
        args.n_speakers = 2         # Two-person conversation
        args.n_classes_priors = 6 # loss_cdt uses 2 priors: positive/negative sentiment
    elif args.dataset == 'MELD':  # <-- New branch
        args.num_folder = 1          # MELD has fixed splits, no cross-validation needed
        args.n_classes = 7           # MELD has 7 emotion classes
        args.n_speakers = 9          # MELD has 9 main characters
        args.n_classes_priors = 7 # loss_cdt uses 2 priors: positive/negative sentiment

    # Print command line argument configuration
    print(args)
    print("Dataset:", args.dataset)


    # ============================ #
    #   Load Data                  #
    # ============================ #

    print(f'\n========== Reading Data ==========\n')

    if args.dataset == 'MELD':
        train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_workers=0,
            meld_path=path.PATH_MELD
        )
    else:
        # Read modality feature names
        audio_feature = args.audio_feature
        text_feature = args.text_feature
        video_feature = args.video_feature

        # Print feature path info
        print(f"PATH_TO_FEATURES_Win[args.dataset]: {path.PATH_TO_FEATURES_Win[args.dataset]}")
        print(f"audio_feature: {audio_feature}")

        # Construct feature paths for audio, text, video (concatenate by dataset name)
        audio_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], audio_feature)
        text_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], text_feature)
        video_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], video_feature)

        # Ensure feature file paths exist
        assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(video_root), f'features not exist!'

        # Load training, validation, testing data loaders, and dimension info for each modality
        train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
            audio_root=audio_root,
            text_root=text_root,
            video_root=video_root,
            num_folder=args.num_folder,
            batch_size=args.batch_size,
            dataset=args.dataset,
            num_workers=0
        )

    # Ensure data split is correct
    assert len(train_loaders) == args.num_folder, f'Error: folder number'

    # ============================ #
    #   Model Training             #
    # ============================ #

    # Print start training prompt
    print(f'\n========== Training and Evaluation ==========\n')

    # Initialize containers to save results for each fold
    folder_acc = []        # Best accuracy per fold
    folder_f1 = []         # Best F1 score per fold
    folder_recon = []      # Best reconstruction loss per fold
    folder_save = []       # Best saved predictions/labels etc. per fold
    folder_losswhole = []  # Loss records for all epochs per fold

    # Start cross-validation training
    for ii in range(args.num_folder):
        print(f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        val_loader = val_loaders[ii]
        test_loader = test_loaders[ii]
        start_time = time.time()

        # ============================ #
        #   1. Build Model             #
        # ============================ #
        print('Step1: build model (each folder has its own model)')
        model = build_model(args, adim, tdim, vdim)
        reg_loss = MaskedMSELoss()      # Regression loss
        cls_loss = MaskedCELoss()       # Classification loss
        if cuda:
            model.cuda()
            cls_loss.cuda()

        # ============================ #
        #   Optimizer Settings         #
        # ============================ #
        core_params = []
        flow_rec_params = []
        for name, param in model.named_parameters():
            # Determine which group the parameter belongs to based on name
            # Classify parameters starting with flow_, rec_, priors, proj_to_common_ as flow/rec (non-core)
            if 'flow_' in name or \
               'rec_' in name or \
               'priors' in name or \
               name.startswith('proj_to_common_'): # <-- Modified condition here
                flow_rec_params.append(param)
                # print(f"  -> Flow/Rec Param: {name}") # (Optional) Check grouping
            else:
                core_params.append(param)
                # print(f"  -> Core Param: {name}") # (Optional) Check grouping
        print(f"Total Core parameters: {sum(p.numel() for p in core_params):,}")
        print(f"Total Flow/Rec parameters: {sum(p.numel() for p in flow_rec_params):,}\n")
        # Use parameter groups:
        optimizer = optim.Adam([
            {'params': core_params, 'lr': args.lr, 'name': 'core'}, # Core parameter group
            {'params': flow_rec_params, 'lr': args.lr * 0.1, 'name': 'flow_rec'} # flow/rec parameter group, may need smaller learning rate
        ], weight_decay=args.l2)

        # ============================ #
        #   1. Start Training          #
        # ============================ #

        # Step 2: Train for multiple epochs
        print('Step2: training (multiple epoches)')
        all_losses = []       # Save training/validation/testing loss for each epoch
        all_labels = []       # Save prediction results etc. for each epoch
        val_fscores = []      # Validation F1 score for each epoch
        val_accs = []  # New: Store total validation accuracy for each epoch
        test_fscores, test_accs, test_recon = [], [], []
        best_f1 = 0
        best_val_fscore = 0.0
        epochs_no_improve = 0
        best_model_state = None
        # Initialize list to record validation set confidence history for each epoch
        val_confidence_history = {
            'audio': [],
            'text': [],
            'visual': []
        }
        val_metrics_history = {
            'acc_audio': [], 'acc_text': [], 'acc_visual': [],
            'f1_audio': [], 'f1_text': [], 'f1_visual': []
        }
        phase1_early_stopped = False # New: Mark if phase 1 has early stopped


        epoch = 0 # Move epoch initialization outside the loop
        while epoch < args.epochs: # Use while loop so we can manually skip epochs
            # Only support constant-xx format mask_type
            assert args.mask_type.startswith('constant'), f'mask_type should be constant-x.x'
            mask_rate = float(args.mask_type.split('-')[-1])

            # ============================ #
            #   Training Phase             #
            # ============================ #
            train_acc, train_fscore, train_names, train_loss, trainsave, _ , train_mod_metrics, _, _ = train_or_eval_model(
                args, model, reg_loss, cls_loss, train_loader,
                mask_rate=mask_rate, optimizer=optimizer, train=True)
            
            # ============================ #
            #   Validation Phase           #
            # ============================ #
            val_acc, val_fscore, val_names, val_loss, valsave, val_confidences, val_mod_metrics, _, _ = train_or_eval_model(
                args, model, reg_loss, cls_loss, val_loader,
                mask_rate=mask_rate, optimizer=None, train=False)
            
            # ============================ #
            #   Testing Phase              #
            # ============================ #
            test_acc, test_fscore, test_names, test_loss, testsave, _ , test_mod_metrics, test_hidden_features, test_final_feats = train_or_eval_model(
                args, model, reg_loss, cls_loss, test_loader,
                mask_rate=mask_rate, optimizer=None, train=False)

            # ============================ #
            #   Record Validation Results  #
            # ============================ #
            val_fscores.append(val_fscore)
            val_accs.append(val_acc)
            val_confidence_history['audio'].append(val_confidences[0])
            val_confidence_history['text'].append(val_confidences[1])
            val_confidence_history['visual'].append(val_confidences[2])
            val_metrics_history['acc_audio'].append(val_mod_metrics['acc'][0])
            val_metrics_history['acc_text'].append(val_mod_metrics['acc'][1])
            val_metrics_history['acc_visual'].append(val_mod_metrics['acc'][2])
            val_metrics_history['f1_audio'].append(val_mod_metrics['f1'][0])
            val_metrics_history['f1_text'].append(val_mod_metrics['f1'][1])
            val_metrics_history['f1_visual'].append(val_mod_metrics['f1'][2])

            best_f1, best_model_state = update_best_model(val_fscore, best_f1, model, best_model_state)

            # ============================ #
            #   Record Testing Results     #
            # ============================ #
            test_accs.append(test_acc)
            test_fscores.append(test_fscore)
            test_recon.append(test_loss[4])  # Reconstruction loss
            all_labels.append({'test_labels': testsave[1], 'test_preds': testsave[0], 'test_names': test_names, 'test_fmask': testsave[3], 'test_hidden': test_hidden_features,'test_final_feats': test_final_feats})
            
            # ============================ #
            #   Record Loss Values         #
            # ============================ #

            all_losses.append({'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss})


            # ============================ #
            #   Print Output               #
            # ============================ #

            # Modify print statement to include new metrics
            # Format output string for single modality metrics
            train_mod_acc_str = ', '.join([f'{acc:.2%}' for acc in train_mod_metrics['acc']])
            train_mod_f1_str = ', '.join([f'{f1:.2%}' for f1 in train_mod_metrics['f1']])
            val_mod_acc_str = ', '.join([f'{acc:.2%}' for acc in val_mod_metrics['acc']])
            val_mod_f1_str = ', '.join([f'{f1:.2%}' for f1 in val_mod_metrics['f1']])
            test_mod_acc_str = ', '.join([f'{acc:.2%}' for acc in test_mod_metrics['acc']])
            test_mod_f1_str = ', '.join([f'{f1:.2%}' for f1 in test_mod_metrics['f1']])

            print(f'epoch:{epoch+1}; train_fscore:{train_fscore:2.2%}; train_acc:{train_acc:2.2%}; train_loss:{train_loss[0]}; train_loss1:{train_loss[1]}; train_loss2:{train_loss[2]}; train_loss_cdt:{train_loss[3]}; train_loss_rec:{train_loss[4]}')
            print(f'           L-- [Train] Modality Accs (A,T,V): [{train_mod_acc_str}] | F1s: [{train_mod_f1_str}]')

            print(f'epoch:{epoch+1}; val_fscore:{val_fscore:2.2%}; val_acc:{val_acc:2.2%}; val_loss:{val_loss[0]}; val_loss1:{val_loss[1]}; val_loss2:{val_loss[2]}; val_loss_cdt:{val_loss[3]}; val_loss_rec:{val_loss[4]}')
            print(f'           L-- [Val]   Modality Accs (A,T,V): [{val_mod_acc_str}] | F1s: [{val_mod_f1_str}]')
            
            print(f'epoch:{epoch+1}; test_fscore:{test_fscore:2.2%}; test_acc:{test_acc:2.2%}; test_loss:{test_loss[0]}; test_loss1:{test_loss[1]}; test_loss2:{test_loss[2]}; test_loss_cdt:{test_loss[3]}; test_loss_rec:{test_loss[4]}')
            print(f'           L-- [Test]  Modality Accs (A,T,V): [{test_mod_acc_str}] | F1s: [{test_mod_f1_str}]')

            
            # ============================ #
            #   Save Best Model            #
            # ============================ #
            # Check if current val F1 is better than *global* history best F1
            if val_fscore > best_val_fscore + args.early_stop_delta:
                best_val_fscore = val_fscore
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"✅ Epoch {epoch+1}: New best validation F1: {best_val_fscore:.4f}. Resetting patience.")
            else:
                epochs_no_improve += 1
                print(f"⚠️ Epoch {epoch+1}: No improvement in validation F1 for {epochs_no_improve} epoch(s). Patience: {args.early_stop_patience}.")
            # Check if early stopping is triggered
            if epochs_no_improve >= args.early_stop_patience:
                print(f"🛑 early stopping triggered after {epoch + 1} epochs.")
                break # Exit while loop

            print("-" * 80)
            epoch += 1 # Manually increment epoch counter

        # ============================ #
        #   Record Index of Best Result on Validation Set #
        # ============================ #

        # Step 3: Choose the best epoch on validation set, save its corresponding test result
        print(f'Step3: saving and testing on the {ii+1} folder')
        best_index = np.argmax(np.array(val_fscores))  # Epoch with best val F1
        # best_index = np.argmax(np.array(val_accs))  # Epoch with best val acc

        # ============================ #
        #   Save Corresponding Test Set Performance Metrics Based on Validation Set Index #
        # ============================ #

        bestf1 = test_fscores[best_index]
        bestacc = test_accs[best_index]
        bestrecon = test_recon[best_index]
        bestsave = all_labels[best_index]

        folder_f1.append(bestf1)
        folder_acc.append(bestacc)
        folder_recon.append(bestrecon)
        folder_save.append(bestsave)
        folder_losswhole.append(all_losses)

        # ============================ #
        #   Output Runtime             #
        # ============================ #

        end_time = time.time()
        print(f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time:.2f} >>>>>')

    # ============================ #
    #   Save All Results           #
    # ============================ #

    print('\n====== Aggregating and Saving Final Results =======')
    save_root = path.RESULT_DIR
    os.makedirs(save_root, exist_ok=True)
    
    # Construct unified filename base part
    mask_rate = args.mask_type.split('-')[-1]
    suffix_name = f'{args.dataset.lower()}_Graph_mask:{mask_rate}'
    mean_f1, mean_acc = np.mean(folder_f1), np.mean(folder_acc)
    res_name = f'f1_{mean_f1:.4f}_acc_{mean_acc:.4f}'
    time_now = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
    save_path_base = os.path.join(save_root, f'{time_now}_{suffix_name}_{res_name}')

    all_best_labels = np.concatenate([np.concatenate(fold['test_labels']) for fold in folder_save])
    all_best_preds = np.concatenate([np.concatenate(fold['test_preds']) for fold in folder_save])

    # ============================ #
    #   Save All Results           #
    # ============================ #

    print('\n====== Aggregating and Saving Final Results =======')
    save_root = path.RESULT_DIR
    os.makedirs(save_root, exist_ok=True)
    
    # Construct unified filename base part
    mask_rate = args.mask_type.split('-')[-1]
    suffix_name = f'{args.dataset.lower()}_Graph_mask:{mask_rate}'
    mean_f1, mean_acc = np.mean(folder_f1), np.mean(folder_acc)
    res_name = f'f1_{mean_f1:.4f}_acc_{mean_acc:.4f}'
    time_now = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
    save_path_base = os.path.join(save_root, f'{time_now}_{suffix_name}_{res_name}')

    all_best_labels = np.concatenate([np.concatenate(fold['test_labels']) for fold in folder_save])
    all_best_preds = np.concatenate([np.concatenate(fold['test_preds']) for fold in folder_save])
    
    # ============================ #
    #   [New] Prepare all_names variable
    # ============================ #
    # Extract best test set sample names for each fold from folder_save
    all_best_names = []
    for fold_data in folder_save:
        all_best_names.extend(fold_data['test_names'])
        
    # Simple integrity check
    if len(all_best_names) != len(all_best_preds):
        print(f"Warning: Name count ({len(all_best_names)}) != Pred count ({len(all_best_preds)}).")

    # ============================ #
    #   [Restore] Process prediction results (Argmax or Binary Classification)
    # ============================ #

    # If it is a classification task, need to apply argmax to preds (This is your original logic, must be kept!)
    if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix', 'MELD']:
        all_best_preds = np.argmax(all_best_preds, axis=1)

    # ============================ #
    #   [Modify] Call save_results
    # ============================ #
    
    save_results(
        args, 
        folder_f1, 
        folder_acc, 
        all_best_labels, 
        all_best_preds, 
        save_path_base, 
        all_names=all_best_names  # Pass in sample names
    )