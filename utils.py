import torch
import numpy as np

# Built-in libraries
import os                  # Operating system related functions, such as file paths, environment variables, etc.
import datetime  # Import datetime module, used for handling date and time

# Third-party numerical calculation library
import numpy as np                                 # Numerical calculation library
from numpy.random import randint                   # Import numpy's random integer function

import pandas as pd

# PyTorch related libraries
import torch                                        # PyTorch main library
from torch.utils.data import DataLoader             # Data loading tool
from torch.utils.data.sampler import SubsetRandomSampler  # Subset random sampler
import torch.nn.functional as F

# Machine learning related functions
from sklearn.preprocessing import OneHotEncoder             # One-hot encoder, used to convert classification labels to one-hot format

# Local module imports (custom modules)
import path                              # Custom path management module (usually used for path constant definitions)
from model import GraphModel      # Import Graph Neural Network model class GraphModel
from dataloader_iemocap import IEMOCAPDataset   # Import encapsulation class for IEMOCAP dataset
from dataloader_meld import MyMELDDataset # <-- ✅ Added this line

import sys
import copy

# Import libraries required for generating classification reports and confusion matrices
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Logger class for logging
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):  # Initialize logger
        self.terminal = stream  # Output log to terminal
        self.log = open(filename, 'w')  # Open log file, create new file if it doesn't exist

    def write(self, message):  # Override write method to output log message to both terminal and file
        self.terminal.write(message)  # Output to terminal
        self.log.write(message)  # Output to log file

    def flush(self):  # Define flush method to clear buffer during log output
        pass  # No content needs to be implemented here

def save_model_state(args, model_state, save_root, folder_index, folder_f1, folder_acc, folder_recon):
    """
    Save model state to local file.
    Automatically create the save directory if it doesn't exist.
    """
    # Automatically create directory (if it doesn't exist)
    # Extract mask ratio from mask_type, e.g., 'random-0.2' -> '0.2'
    mask_rate = args.mask_type.split('-')[-1]
    # Construct the middle part of the save path, including dataset name, graph model name, and mask ratio info
    suffix_name = f'{args.dataset.lower()}_Graph_mask:{mask_rate}'
    # Calculate average metrics across multiple folders: F1 score, accuracy, and reconstruction loss
    mean_f1, mean_acc, mean_recon = np.mean(folder_f1), np.mean(folder_acc), np.mean(folder_recon)
    # Construct result info string, including average F1, accuracy, and reconstruction loss (formatted to 2 percentage points or 4 decimal places)
    res_name = f'f1:{mean_f1:2.2%}_acc:{mean_acc:2.2%}_reconloss:{mean_recon:.4f}'
    # Get current time, formatted as Year-Month-Day_Hour_Minute_Second
    time_now = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"


    os.makedirs(save_root, exist_ok=True)
    
    model_path = os.path.join(save_root, f'{time_now}_{suffix_name}_classifier:{res_name}_{folder_index}.pt')
    torch.save(model_state, model_path)
    print(f'>>>>> Best model for folder {folder_index} saved to {model_path} >>>>>')
    return model_path

def update_best_model(val_fscore, best_f1, model, best_model_state):
    """
    Update the best model parameters based on the current val_fscore.
    """
    if val_fscore > best_f1:
        best_f1 = val_fscore
        best_model_state = copy.deepcopy(model.state_dict())
    return best_f1, best_model_state


def get_loaders(dataset, batch_size, num_workers, meld_path=None, audio_root=None, text_root=None, video_root=None, num_folder=5):

    if dataset == 'MELD':
        print("INFO: Loading MELD dataset...")
        if meld_path is None:
            raise ValueError("MELD dataset requires the 'meld_path' argument (path to .pkl file).")

        # MELD has fixed train/valid/test splits, the pkl file already contains this info
        # We create three Dataset instances here
        # Note: MELD pkl usually contains train/dev/test. We use the dev set as the validation set
        train_dataset = MyMELDDataset(path=meld_path, split='train')
        val_dataset = MyMELDDataset(path=meld_path, split='valid')
        test_dataset = MyMELDDataset(path=meld_path, split='test')

        # Create DataLoader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            collate_fn=val_dataset.collate_fn,
            shuffle=False,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            collate_fn=test_dataset.collate_fn,
            shuffle=False,
            num_workers=num_workers
        )

        # To keep consistent with your original code's return format, we wrap loaders in a list
        train_loaders = [train_loader]
        val_loaders = [val_loader]
        test_loaders = [test_loader]
        
        # Get feature dimensions from dataset instance
        adim, tdim, vdim = train_dataset.get_featDim()

        return train_loaders, val_loaders, test_loaders, adim, tdim, vdim

    # If dataset is IEMOCAPFour or IEMOCAPSix, use 5-fold cross-validation
    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:  # Each fold contains (train set, test set)

        # Initialize IEMOCAP dataset object, passing label path and root paths for three modalities
        dataset = IEMOCAPDataset(
            label_path=path.PATH_TO_LABEL_Win[dataset],
            audio_root=audio_root,
            text_root=text_root,
            video_root=video_root
        )

        # Get sample indices corresponding to each session (total 5), used for subsequent splitting
        session_to_idx = {}
        for idx, vid in enumerate(dataset.vids):     # Iterate through all video IDs
            session = int(vid[4]) - 1                # 5th character is session number, e.g., "Ses01F_impro01"
            if session not in session_to_idx:
                session_to_idx[session] = []         # Initialize list for this session
            session_to_idx[session].append(idx)      # Add current index to corresponding session

        # Ensure session count equals preset number of folds (usually 5)
        assert len(session_to_idx) == num_folder, f'Must split into five folder'

        # Construct 5-fold train/test index pairs
        train_test_idxs = []
        for ii in range(num_folder):  # ii: current fold number, 0 to 4
            test_idxs = session_to_idx[ii]  # Current fold as test set
            train_idxs = []
            for jj in range(num_folder):    # Remaining folds as training set
                if jj != ii:
                    train_idxs.extend(session_to_idx[jj])
            train_test_idxs.append([train_idxs, test_idxs])  # Store train/test indices for this fold

        # Create 5 sets of train/test data loaders
        train_loaders = []
        test_loaders = []
        for ii in range(len(train_test_idxs)):
            train_idxs = train_test_idxs[ii][0]
            test_idxs = train_test_idxs[ii][1]

            # Construct training data loader, use SubsetRandomSampler to shuffle indices
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(train_idxs),
                collate_fn=dataset.collate_fn,
                num_workers=num_workers,
                pin_memory=False
            )

            # Construct testing data loader
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(test_idxs),
                collate_fn=dataset.collate_fn,
                num_workers=num_workers,
                pin_memory=False
            )

            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

        # Get feature dimensions for audio, text, and video modalities
        adim, tdim, vdim = dataset.get_featDim()

        # Return train loaders, test loaders (test_loaders twice to maintain function signature consistency)
        return train_loaders, test_loaders, test_loaders, adim, tdim, vdim



def build_model(args, adim, tdim, vdim):
    lstm_hidden_size = args.lstm_hidden_size
    graph_hidden_size = args.graph_hidden_size
    flow_hidden_size = args.flow_hidden_size

    # Instantiate Graph Neural Network model
    model = GraphModel(
        adim, tdim, vdim,
        lstm_hidden_size, graph_hidden_size, flow_hidden_size, 
        n_speakers=args.n_speakers,
        window_past=args.windowp,
        window_future=args.windowf,
        n_classes=args.n_classes,
        dropout=args.dropout,
        no_cuda=args.no_cuda,
        n_classes_priors=args.n_classes_priors,
        dataset=args.dataset,
        n_flow=args.n_flow,
        num_blocks_rec=args.num_blocks_rec,
        reduction_rec=args.reduction_rec,
        n_block_glow=args.n_block_glow
    )

    # Output model structure
    print("\n🔍 Full Model Architecture:")
    print(model)

    # Output total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total Parameters: {total_params:,}")

    # Output parameter count for each submodule
    print("\n📦 Parameter Breakdown by Module:")
    from collections import defaultdict
    module_params = defaultdict(int)

    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        module_params[module_name] += param.numel()

    for module, count in module_params.items():
        print(f" - {module:<20}: {count:,} params")

    return model




# Construct model input features for multimodal fusion (host: main speaker, guest: other)
# Input dimensions: audio/text/visual_* -> [seqlen, batch, dim]
def generate_inputs(audio_host, text_host, visual_host, audio_guest, text_guest, visual_guest, qmask):
    input_features = []
    # Concatenate main speaker's three modality features along the last dimension (modality dim)
    feat1 = torch.cat([audio_host, text_host, visual_host], dim=2)  # [seqlen, batch, featdim]
    # Concatenate guest speaker's three modality features
    feat2 = torch.cat([audio_guest, text_guest, visual_guest], dim=2)
    featdim = feat1.size(-1)  # Get total dimension after concatenation (adim + tdim + vdim)
    # Transpose speaker mask qmask: from [batch, seqlen] -> [seqlen, batch]
    tmask = qmask.transpose(0, 1)
    # Add dimension to align with features, and repeat along modality dim, getting [seqlen, batch, featdim]
    tmask = tmask.unsqueeze(2).repeat(1, 1, featdim)
    # Use qmask to select speaker: 0 represents main speaker, 1 represents guest speaker
    select_feat = torch.where(tmask == 0, feat1, feat2)  # Conditionally select corresponding modality features
    input_features.append(select_feat)  # Store in list (list length is 1, element is [seqlen, batch, featdim])
    return input_features               # Return input features in nested list form



## Generate missing masks (mask) following CPM-Net method
def random_mask(view_num, input_len, missing_rate):
    """
    Randomly generate incomplete modality mask matrix to simulate multimodal missing scenarios.

    Args:
        view_num: Number of modalities (e.g., audio, text, video)
        input_len: Number of samples (usually equal to batch size)
        missing_rate: Missing rate (e.g., 0.2 means about 20% of modalities are missing per sample)

    Returns:
        matrix: 0/1 mask matrix of shape=[input_len, view_num], 1 means modality retained, 0 means missing
    """

    assert missing_rate is not None
    one_rate = 1 - missing_rate       # one_rate represents retention rate

    # Case 1: If one_rate <= 1 / view_num, ensure at least one modality is retained per sample (fully one-hot)
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        # Randomly retain one modality per sample, others are 0
        view_preserve = enc.fit_transform(randint(0, view_num, size=(input_len, 1))).toarray()
        return view_preserve  # shape=[num_samples, num_modalities], only one 1 per row, others 0

    # Case 2: If no missing (i.e., one_rate = 1), return all-ones matrix
    if one_rate == 1:
        matrix = randint(1, 2, size=(input_len, view_num))  # Equivalent to np.ones([...])
        return matrix

    # Case 3: General case, partial modality missing, need to control ratio + ensure at least one modality per row
    # To enhance generalization, we set generation sample size to at least 32
    if input_len < 32:
        alldata_len = 32
    else:
        alldata_len = input_len

    error = 1  # Initial error set to 1, enter loop
    while error >= 0.005:

        # Step 1: One-hot initialization, ensure at least one modality retained per row
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()

        # Step 2: Calculate target number of additional modalities based on expected one_rate
        one_num = view_num * alldata_len * one_rate - alldata_len  # Subtract ones generated by one-hot
        ratio = one_num / (view_num * alldata_len)  # Estimate current ratio

        # Initial missing mask matrix generation (0 or 1), sampling density is ratio
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)

        # Step 3: Calculate overlap with one-hot initialization (avoid duplicate setting of 1s)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))  # Number of overlapping areas

        # Correct the number of additional 1s to sample, considering overlap removal to reach target one_num
        one_num_iter = one_num / (1 - a / one_num)

        # Recalculate ratio based on corrected one_num_iter
        ratio = one_num_iter / (view_num * alldata_len)

        # Regenerate random mask matrix
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)

        # Merge with one-hot initial matrix, retain if at least 1 at each position
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)

        # Calculate current actual retention ratio, compare error with target one_rate
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)  # Update error

    # Final output matrix should have original input_len rows
    matrix = matrix[:input_len, :]
    return matrix

def get_class_names(dataset_name):
    """Return list of class names based on dataset name"""
    if dataset_name == 'IEMOCAPFour':
        return ['hap', 'sad', 'ang', 'neu']
    elif dataset_name == 'IEMOCAPSix':
        return ['hap', 'sad', 'ang', 'neu', 'exc', 'fru']
    elif dataset_name == 'MELD':
        return ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
    else:
        return None


# Note: Be sure to modify function definition line to add all_names=None
def save_results(args, folder_f1, folder_acc, all_labels, all_preds, save_path_base, all_names=None):
    """
    Save final classification metrics and confusion matrix to file.
    :param all_names: New parameter, list of names for all test samples
    """
    # ----------------- 1. Save classification metrics to TXT file -----------------
    report_path = save_path_base + '_metrics.txt'
    print(f'Saving classification report to {report_path}')
    
    # Indentation level here is first level of function
    with open(report_path, 'w') as f:
        f.write("=======================================================\n")
        f.write(f"           Final Evaluation Results\n")
        f.write("=======================================================\n\n")
        
        mean_f1 = np.mean(folder_f1)
        std_f1 = np.std(folder_f1)
        mean_acc = np.mean(folder_acc)
        std_acc = np.std(folder_acc)
        
        f.write(f"Overall Performance ({args.num_folder}-fold cross-validation):\n")
        f.write(f"  - Average Weighted F1-Score: {mean_f1:.4f} (±{std_f1:.4f})\n")
        f.write(f"  - Average Accuracy:        {mean_acc:.4f} (±{std_acc:.4f})\n\n")
        
        f.write("-------------------------------------------------------\n")
        f.write("      Detailed Classification Report (Aggregated)\n")
        f.write("      (Based on test set predictions from the best\n")
        f.write("       epoch of each fold, determined by validation F1)\n")
        f.write("-------------------------------------------------------\n\n")

        class_names = get_class_names(args.dataset)
        
        # Generate detailed report and confusion matrix only if it is a classification task
        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix', 'MELD']:
            # Generate and write sklearn classification report
            report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
            f.write(report)
            
            # ----------------- 2. Save confusion matrix as image -----------------
            # Note: plotting logic is suggested to be outside with open block, or ensure no interference with file writing
            # But for logical coherence, place it here first, as long as variable names don't conflict
            cm_path = save_path_base + '_confusion_matrix.png'
            print(f'Saving confusion matrix to {cm_path}')
            
            cm = confusion_matrix(all_labels, all_preds)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names,
                        annot_kws={"size": 16})  # Added this line, number 16 can be adjusted as needed
            
            plt.title(f'Confusion Matrix - {args.dataset}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close() # Close figure
        else:
            f.write("Regression task - Classification report and confusion matrix are not applicable.\n")

    # =======================================================
    #  New feature: Export prediction results to Excel/CSV
    #  Note: This code is suggested to be placed outside with open() block, as an independent part of the function
    # =======================================================
    if all_names is not None:
        print(f'Saving detailed predictions to {save_path_base}_predictions.xlsx')
        
        # Get class name mapping
        class_names = get_class_names(args.dataset)
        
        # Construct DataFrame data
        data_dict = {
            'Sample_Name': all_names,
            'True_Label_Index': all_labels.astype(int),
            'Predicted_Label_Index': all_preds.astype(int)
        }

        # If class names exist, try adding text label columns
        if class_names:
            try:
                # Handle classification tasks
                if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix', 'MELD']:
                    data_dict['True_Label'] = [class_names[i] for i in all_labels.astype(int)]
                    data_dict['Predicted_Label'] = [class_names[i] for i in all_preds.astype(int)]
            except IndexError:
                print("Warning: Label index out of range for class names, skipping text label columns.")

        # Create DataFrame
        df_pred = pd.DataFrame(data_dict)
        
        # Save as Excel (requires openpyxl: pip install openpyxl)
        try:
            df_pred.to_excel(save_path_base + '_predictions.xlsx', index=False)
        except Exception as e:
            print(f"Error saving Excel: {e}, falling back to CSV.")
            df_pred.to_csv(save_path_base + '_predictions.csv', index=False)
            
    print("Results summary and predictions saved successfully.")