import os
import glob
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


## Get name2features mapping based on label file and feature path
def read_data(label_path, feature_root):

    # ============================ #
    #     Read information from label file
    # ============================ #
    names = []      # List of all utterance names
    speakers = []   # Corresponding speakers (F/M)

    # Load multiple variables from pkl file
    videoIDs, videoLabels, videoSpeakers, videoSentence, trainVid, testVid = pickle.load(
        open(label_path, "rb"), encoding='latin1')

    # All video IDs (containing both training and testing)
    vids = sorted(list(trainVid | testVid))

    # Iterate through each video to extract the name and speaker of each utterance
    for ii, vid in enumerate(vids):
        uids_video = videoIDs[vid]          # Unique ID for each utterance in the current video (e.g., Ses01F_impro01_F000)
        spks_video = videoSpeakers[vid]     # Speaker for each utterance (F/M)
        names.extend(uids_video)
        speakers.extend(spks_video)

    # ============================ #
    #      Load corresponding feature data
    # ============================ #
    features = []       # Features for all utterances
    feature_dim = -1    # Feature dimension (for subsequent initialization)

    for ii, name in enumerate(names):
        speaker = speakers[ii]

        # Find the corresponding feature file or folder (supports prefix matching)
        feature_dir = glob.glob(os.path.join(feature_root, name + '*'))
        assert len(feature_dir) == 1, f"Found multiple or no feature files: {name}"
        feature_path = feature_dir[0]

        # Initialize the feature structure for this utterance, distinguishing between male and female (F/M)
        feature = {'F': [], 'M': []}

        if feature_path.endswith('.npy'):
            # Audio/Text features => Load npy file directly (one per utterance)
            single_feature = np.load(feature_path)
            single_feature = single_feature.squeeze()  # [Dim,] or [Time, Dim]
            feature[speaker].append(single_feature)
            feature_dim = max(feature_dim, single_feature.shape[-1])  # Update max dimension
        else:
            # Video (e.g., FACET) features are in a folder containing multiple .npy files divided by face (e.g., F_001.npy)
            facenames = os.listdir(feature_path)
            for facename in sorted(facenames):
                assert 'F' in facename or 'M' in facename, f"face filename should contain F or M"
                facefeat = np.load(os.path.join(feature_path, facename))
                feature_dim = max(feature_dim, facefeat.shape[-1])  # Update max dimension
                if 'F' in facename:
                    feature['F'].append(facefeat)
                else:
                    feature['M'].append(facefeat)

        # Average multiple frames for each speaker modality to become a fixed-dimension vector
        for speaker in feature:
            single_feature = np.array(feature[speaker]).squeeze()
            if len(single_feature) == 0:
                single_feature = np.zeros((feature_dim, ))  # Pad with 0 if empty
            elif len(single_feature.shape) == 2:
                single_feature = np.mean(single_feature, axis=0)  # Average over multiple frames
            feature[speaker] = single_feature  # Final shape: [Dim]

        features.append(feature)  # Add to all features

    # ============================ #
    #     Build name -> feature mapping
    # ============================ #
    print(f'Input feature {feature_root} ===> dim is {feature_dim}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'

    name2feats = {}
    for ii in range(len(names)):
        name2feats[names[ii]] = features[ii]  # Each name corresponds to its {'F':..., 'M':...} features

    return name2feats, feature_dim  # Return mapping and feature dimension




class IEMOCAPDataset(Dataset):

    def __init__(self, label_path, audio_root, text_root, video_root):
        """
        Initialize dataset: Read features and labels, and build a dual-speaker feature alignment data structure.
        """

        # ========================= #
        # Read feature data for three modalities
        # Returns mapping of name -> {'F': [...], 'M': [...]} and feature dimension
        # ========================= #
        name2audio, adim = read_data(label_path, audio_root)
        name2text, tdim = read_data(label_path, text_root)
        name2video, vdim = read_data(label_path, video_root)

        self.adim = adim
        self.tdim = tdim
        self.vdim = vdim

        # ========================= #
        # Initialize dictionaries for multimodal, labels, speakers, etc.
        # ========================= #
        self.max_len = -1

        self.videoAudioHost = {}     # Host speaker audio features
        self.videoTextHost = {}      # Host speaker text features
        self.videoVisualHost = {}    # Host speaker visual features

        self.videoAudioGuest = {}    # Guest speaker audio features
        self.videoTextGuest = {}     # Guest speaker text features
        self.videoVisualGuest = {}   # Guest speaker visual features

        self.videoLabelsNew = {}     # Labels for each utterance
        self.videoSpeakersNew = {}   # Speaker encoding for each utterance

        speakermap = {'F': 0, 'M': 1}  # Map F/M to integers 0/1

        # ========================= #
        # Load multiple structured information from label file
        # ========================= #
        (self.videoIDs,
         self.videoLabels,
         self.videoSpeakers,
         self.videoSentences,
         self.trainVid, self.testVid) = pickle.load(open(label_path, "rb"), encoding='latin1')

        # All video IDs (train + test)
        self.vids = sorted(list(self.trainVid | self.testVid))

        for ii, vid in enumerate(self.vids):
            uids = self.videoIDs[vid]         # List of utterance names
            labels = self.videoLabels[vid]    # Corresponding labels
            speakers = self.videoSpeakers[vid]# Corresponding speakers (F/M)

            self.max_len = max(self.max_len, len(uids))  # Record max sequence length

            # Initialize modality storage structure for current video
            self.videoAudioHost[vid] = []
            self.videoTextHost[vid] = []
            self.videoVisualHost[vid] = []
            self.videoAudioGuest[vid] = []
            self.videoTextGuest[vid] = []
            self.videoVisualGuest[vid] = []
            self.videoLabelsNew[vid] = []
            self.videoSpeakersNew[vid] = []

            for ii, uid in enumerate(uids):
                # Fill features for corresponding speakers into host (F) and guest (M)
                self.videoAudioHost[vid].append(name2audio[uid]['F'])
                self.videoTextHost[vid].append(name2text[uid]['F'])
                self.videoVisualHost[vid].append(name2video[uid]['F'])
                self.videoAudioGuest[vid].append(name2audio[uid]['M'])
                self.videoTextGuest[vid].append(name2text[uid]['M'])
                self.videoVisualGuest[vid].append(name2video[uid]['M'])
                self.videoLabelsNew[vid].append(labels[ii])
                self.videoSpeakersNew[vid].append(speakermap[speakers[ii]])

            # Convert to numpy array (easier for subsequent processing)
            self.videoAudioHost[vid] = np.array(self.videoAudioHost[vid])
            self.videoTextHost[vid] = np.array(self.videoTextHost[vid])
            self.videoVisualHost[vid] = np.array(self.videoVisualHost[vid])
            self.videoAudioGuest[vid] = np.array(self.videoAudioGuest[vid])
            self.videoTextGuest[vid] = np.array(self.videoTextGuest[vid])
            self.videoVisualGuest[vid] = np.array(self.videoVisualGuest[vid])
            self.videoLabelsNew[vid] = np.array(self.videoLabelsNew[vid])
            self.videoSpeakersNew[vid] = np.array(self.videoSpeakersNew[vid])

    # ========================= #
    # Override __getitem__ method
    # Returns multimodal features and labels etc. for the video at the corresponding index
    # ========================= #
    def __getitem__(self, index):
        vid = self.vids[index]
        return torch.FloatTensor(self.videoAudioHost[vid]),\
               torch.FloatTensor(self.videoTextHost[vid]),\
               torch.FloatTensor(self.videoVisualHost[vid]),\
               torch.FloatTensor(self.videoAudioGuest[vid]),\
               torch.FloatTensor(self.videoTextGuest[vid]),\
               torch.FloatTensor(self.videoVisualGuest[vid]),\
               torch.FloatTensor(self.videoSpeakersNew[vid]),\
               torch.FloatTensor([1]*len(self.videoLabelsNew[vid])),\
               torch.LongTensor(self.videoLabelsNew[vid]),\
               vid

    # Total length of dataset
    def __len__(self):
        return len(self.vids)

    # Return feature dimension for each modality
    def get_featDim(self):
        print(f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim

    # Return max sequence length (number of utterances)
    def get_maxSeqLen(self):
        print(f'max seqlen: {self.max_len}')
        return self.max_len

    # Custom collate_fn, used for DataLoader to automatically pad sequences of different lengths
    def collate_fn(self, data):
        datnew = []
        dat = pd.DataFrame(data)  # Convert batch data to DataFrame, columns correspond to fields

        for i in dat:  # Iterate through each column (field)
            if i <= 5: 
                datnew.append(pad_sequence(dat[i]))           # Multimodal input feature padding (preserve order)
            elif i <= 8:
                datnew.append(pad_sequence(dat[i], True))     # Mask and label padding (reverse order pad)
            else:
                datnew.append(dat[i].tolist())                # Video name, no processing needed

        return datnew