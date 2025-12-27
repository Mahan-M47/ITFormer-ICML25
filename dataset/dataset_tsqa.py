#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time Series Question Answering Dataset.
Handles loading and preprocessing of time series data and question-answer pairs.
"""
import sys
from transformers import PretrainedConfig, AutoTokenizer
from transformers import AutoProcessor
import torch
import os
import json
from torch.utils.data import Dataset
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import h5py
import re
import ast
from models.TimeLanguageModel import TLMConfig
from accelerate import Accelerator

# Get accelerator instance for main process checks
accelerator = Accelerator()


class DatasetTSQA(Dataset):
    def __init__(self, dataset_path, tokenizer, ts_pad_num):  # assuming dataset is a CSV file

        super().__init__()
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.ts_pad_num = ts_pad_num
        
        # Key fix: Ensure vocab_size is correct
        self.vocab_size = len(self.tokenizer)
        if accelerator.is_main_process:
            accelerator.print(f"📊 Vocab size: {self.vocab_size}")
        
        # Ensure tokenizer settings are consistent
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Validate special tokens
        self._validate_special_tokens()
        self._build_index()

    def _validate_special_tokens(self):
        """Validate that all special token IDs are within valid range."""
        special_tokens = {
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'bos_token_id': getattr(self.tokenizer, 'bos_token_id', None),
            'unk_token_id': getattr(self.tokenizer, 'unk_token_id', None),
        }
        
        if accelerator.is_main_process:
            accelerator.print("🔍 Validating special tokens:")
        for name, token_id in special_tokens.items():
            if token_id is not None:
                if token_id >= self.vocab_size or token_id < 0:
                    if accelerator.is_main_process:
                        accelerator.print(f"❌ {name} = {token_id} out of range [0, {self.vocab_size})")
                    # Fix invalid special tokens
                    if name == 'pad_token_id':
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                        if accelerator.is_main_process:
                            accelerator.print(f"🔧 Fixed: pad_token_id -> {self.tokenizer.pad_token_id}")
                else:
                    if accelerator.is_main_process:
                        accelerator.print(f"✅ {name} = {token_id}")

    def _validate_token_ids(self, token_ids, context=""):
        """Validate token IDs for validity.
        
        Args:
            token_ids: List of token IDs to validate
            context: Context string for error messages
            
        Returns:
            List of validated token IDs
        """
        if not isinstance(token_ids, list):
            return token_ids
            
        valid_ids = []
        for i, token_id in enumerate(token_ids):
            if token_id < 0 or token_id >= self.vocab_size:
                if accelerator.is_main_process:
                    accelerator.print(f"⚠️ {context} position {i}: invalid token_id {token_id}, replacing with unk_token")
                # Replace with unk_token, if not available use eos_token
                replacement = getattr(self.tokenizer, 'unk_token_id', self.tokenizer.eos_token_id)
                valid_ids.append(replacement)
            else:
                valid_ids.append(token_id)
        return valid_ids

    def _build_index(self):
        """Build dataset index by loading and processing data files."""
        self.datasset = pd.read_csv(self.dataset_path)
        print(f'Dataset Columns {self.datasset.columns}')
        self.datasset = self.datasset.to_dict(orient="records")  # list of dicts


    def __len__(self):
        """Return dataset length."""
        return len(self.datasset)


    def _create_chat_input(self, question):
        """Unified chat input creation method."""
        question = f'{question}\nTimeseries:\n<ts>' 
        messages = [
            {"role": "system", "content": 'You are a helpful assistant.'},
            {"role": "user", "content": question}
        ]
        
        try:
            # Use a safer tokenization method
            chat_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            # Replace time series placeholder
            chat_text = chat_text.replace('<ts>', '<|image_pad|>' * self.ts_pad_num)
            return chat_text
        except Exception as e:
            if accelerator.is_main_process:
                accelerator.print(f"❌ Chat template error: {e}")
            # Fallback to a simple format
            return f"You are a helpful assistant.\nuser\n{question}\nassistant\n"

    def _safe_tokenize(self, text, add_special_tokens=True):
        """Safe tokenization, ensure results are within valid range."""
        try:
            # Add more tokenization parameters
            result = self.tokenizer(
                text, 
                add_special_tokens=add_special_tokens,
                padding=False,
                truncation=False,
                return_tensors=None
            )
            token_ids = result['input_ids']
            
            # Validate token_ids
            token_ids = self._validate_token_ids(token_ids, f"tokenize: {text[:50]}...")
            return token_ids
            
        except Exception as e:
            if accelerator.is_main_process:
                accelerator.print(f"❌ Tokenization error for text: {text[:100]}...")
                accelerator.print(f"Error: {e}")
            # Return a safe default value
            return [self.tokenizer.eos_token_id]

    def __getitem__(self, idx):
        try:
            sample = self.datasset[idx]
            # sample = self.add_adaptive_prompt(sample)

            # =========================== Mode 3: Inference/Evaluation ===========================
            # Create query_ids: only the original question text, no other information
            original_question = sample['Question']
            query_ids = self._safe_tokenize(original_question, add_special_tokens=False)
            
            # Create input_ids: includes time series placeholder
            q_text = self._create_chat_input(sample['Question'])
            q_input_ids = self._safe_tokenize(q_text, add_special_tokens=False)
            
            a_text = sample['Answer']
            if not a_text.endswith(self.tokenizer.eos_token):
                a_text += self.tokenizer.eos_token
            a_input_ids = self._safe_tokenize(a_text, add_special_tokens=False)

            # Validate results
            query_ids = self._validate_token_ids(query_ids, f"infer_query_sample_{idx}")
            q_input_ids = self._validate_token_ids(q_input_ids, f"infer_q_sample_{idx}")
            a_input_ids = self._validate_token_ids(a_input_ids, f"infer_a_sample_{idx}")

            ts = ast.literal_eval(sample['Series'])
            # ts = ts[:60] * 10
            ts = [ts]
            
            returned_dict = {
                'form': 'default',
                'stage': 3,
                'query_ids': query_ids,  # Only contains the original question text
                'input_ids': q_input_ids,
                'labels': a_input_ids,
                'ts_values': torch.tensor(ts, dtype=torch.float),
                'index': sample['index']
            }

            return returned_dict
            
        except Exception as e:
            if accelerator.is_main_process:
                accelerator.print(f"❌ Error processing sample {idx}: {e}")


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Ensure tokenizer settings are correct
        if self.tokenizer.padding_side != 'left':
            if accelerator.is_main_process:
                accelerator.print("⚠️  Warning: Setting tokenizer.padding_side to 'left' for decoder-only model")
            self.tokenizer.padding_side = 'left'
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len_inputs = max(len(feature['input_ids']) for feature in features)
        max_len_labels = max(len(feature['labels']) for feature in features)
        max_len_querys = max(len(feature['query_ids']) for feature in features)    
        input_ids = []
        attention_mask = []
        labels = []
        ts_values = []
        stages = []
        index = []
        query_ids = []
        for feature in features:
            input_len = len(feature['input_ids'])
            label_len = len(feature['labels'])
            query_ids_len = len(feature['query_ids'])
            # Left padding is correct (keep original logic)
            padded_input = [self.tokenizer.pad_token_id] * (max_len_inputs - input_len) + feature['input_ids']
            input_ids.append(padded_input)
            
            # Corresponding attention mask
            attention_mask.append([0] * (max_len_inputs - input_len) + [1] * input_len)
            
            # Labels also left-padded
            padded_labels = [self.tokenizer.pad_token_id] * (max_len_labels - label_len) + feature['labels']  # Use -100 to ignore pad positions in loss
            labels.append(padded_labels)
            
            # query_ids also left-padded
            padded_query_ids = [self.tokenizer.pad_token_id] * (max_len_querys - query_ids_len) + feature['query_ids']
            query_ids.append(padded_query_ids)

            ts_values.append(feature['ts_values'])
            stages.append(feature['stage'])
            index.append(feature['index'])


        return {
            'query_ids': torch.tensor(query_ids, dtype=torch.long),
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'ts_values': torch.stack(ts_values, dim=0),
            'stage': torch.tensor(stages, dtype=torch.int8),
            'index': torch.tensor(index, dtype=torch.int32)
        }



