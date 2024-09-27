# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

@dataclass
class time_aware_cls_ce:
    dataset: str = "time_aware_cls_ce"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "training_data/time_aware_cls_ce_train.json"
    valid_data_path: str = "training_data/time_aware_cls_ce_valid.json"

@dataclass
class knowledge_aware_cls_ce:
    dataset: str = "knowledge_aware_cls_ce"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "training_data/knowledge_aware_cls_ce_train.json"
    valid_data_path: str = "training_data/knowledge_aware_cls_ce_valid.json"

@dataclass
class self_aware_cls_ce_llama2_7b_chat:
    dataset: str = "self_aware_cls_ce_llama2_7b_chat"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "training_data/self_aware_cls_ce_llama2_7b_chat_train.json"
    valid_data_path: str = "training_data/self_aware_cls_ce_llama2_7b_chat_valid.json"

@dataclass
class self_aware_cls_ce_llama2_13b_chat:
    dataset: str = "self_aware_cls_ce_llama2_13b_chat"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "training_data/self_aware_cls_ce_llama2_13b_chat_train.json"
    valid_data_path: str = "training_data/self_aware_cls_ce_llama2_13b_chat_valid.json"

@dataclass
class intent_aware_cls_ce:
    dataset: str = "intent_aware_cls_ce"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "training_data/intent_aware_cls_ce_train.json"
    valid_data_path: str = "training_data/intent_aware_cls_ce_valid.json"