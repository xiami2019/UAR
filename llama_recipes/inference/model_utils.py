# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaConfig, LlamaForSequenceClassification, AutoModel
from transformers import MistralForCausalLM, MistralConfig, MistralForSequenceClassification, AutoConfig

# Function to load the main model for text generation
def load_model(model_name, quantization):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

# Loading the model from config to load FSDP checkpoints into that
def load_llama_from_config(config_path):
    model_config = LlamaConfig.from_pretrained(config_path) 
    model = LlamaForCausalLM(config=model_config)
    return model
    
# Loading the model from config to load FSDP checkpoints into that
def load_llama_reward_from_config(config_path, num_labels):
    model_config = LlamaConfig.from_pretrained(config_path) 
    model_config.num_labels = num_labels
    model = LlamaForSequenceClassification(config=model_config)
    return model

def load_mistral_from_config(HF_model_path_or_name):
    model_config = MistralConfig.from_pretrained(HF_model_path_or_name)
    model = MistralForCausalLM(config=model_config)
    return model

def load_mistral_reward_from_config(config_path, num_labels):
    model_config = MistralConfig.from_pretrained(config_path)
    model_config.num_labels = num_labels
    model = MistralForSequenceClassification(config=model_config)
    return model

def load_auto_model_from_config(config_path):
    model_config = AutoConfig.from_pretrained(config_path) 
    model = AutoModel.from_config(model_config)
    return model