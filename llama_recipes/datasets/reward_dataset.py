# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset

B_INST, E_INST = "[INST]", "[/INST]"

class RewardDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        if partition == 'train':
            self.ann = json.load(open(dataset_config.train_data_path))
        else:
            self.ann = json.load(open(dataset_config.valid_data_path))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        example = B_INST + ' ' + ann['question'] + ' ' + E_INST
        example = self.tokenizer.encode(example)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        example_mask = example.ge(0)
        example[~example_mask] = 0

        return {
            "input_ids": example.tolist(),
            "labels": [ann['label']],
            "attention_mask":example_mask.tolist(),
        }
