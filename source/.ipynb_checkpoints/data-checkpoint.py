import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizerFast, BertModel
from transformers import PreTrainedTokenizerFast



from torch.nn import MultiheadAttention

import pandas as pd
import string

from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import spacy
import re
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from tqdm import tqdm

import numpy as np

import time

from spacy.training import Alignment
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

import multiprocessing


def create_custom_tokenizer(nlp, special_tokens):
    # Lấy phần tử tokenizer mặc định của spaCy
    infix_re = compile_infix_regex(nlp.Defaults.infixes)
    
    # Tạo một custom tokenizer từ spaCy mặc định
    custom_tokenizer = Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)

    # Thêm các token đặc biệt vào bộ tokenizer
    special_case = {token: [{spacy.symbols.ORTH: token}] for token in special_tokens}
    for token in special_tokens:
        custom_tokenizer.add_special_case(token, special_case[token])
    
    return custom_tokenizer


def find_sentence_containing_span(text, spans, target_start, target_end):
    for i, (start, end) in enumerate(spans):
        if start <= target_start < end or start < target_end <= end:
            return text[start:end], (start, end)
    return None, None

def get_token_for_char(tokens, char_idx):
    for i, token in enumerate(tokens):
        if char_idx > token.idx:
            continue
        if char_idx == token.idx:
            return i, token
        if char_idx < token.idx:
            return i - 1, tokens[i - 1]
    return len(tokens) - 1, tokens[len(tokens) - 1]


def get_context_by_window(text, tokens, e1_start, e1_end, e2_start, e2_end, ws):
    start = min(e1_start, e2_start)
    end = max(e1_end, e2_end)
    
    start_token, _ = get_token_for_char(tokens, start)
    end_token, _ = get_token_for_char(tokens, end)

    # print('compare windowsize:', end_token - start_token > ws)
    
    window = 0
    if end_token - start_token > ws:
        window = ws // 4
        start_1_token = max(0, start_token - window)
        end_1_token = min(start_token + window, len(tokens) - 1)
        
        start_2_token = max(0, end_token - window)
        end_2_token = min(end_token + window, len(tokens) - 1)

        start_1 = tokens[start_1_token].idx
        end_1 = tokens[end_1_token].idx + len(tokens[end_1_token])
        
        start_2 = tokens[start_2_token].idx
        end_2 = tokens[end_2_token].idx + len(tokens[end_2_token])

        text_1 = text[start_1:end_1]
        text_2 = text[start_2:end_2]
        
        mid = text[end_1:start_2]
        context = "\n".join([text_1, text_2])
        
        if start_1 <= e1_start and e1_end <= end_1: # (e1_start, e1_end) in (start_1, end_1) and (e2_start, e2_end) in (start_2, end_2)
            e1_start = e1_start - start_1
            e1_end =  e1_end - start_1
            
            e2_start =  e2_start - start_1 - (len(mid) - 1)
            e2_end =  e2_end - start_1 - (len(mid) - 1)
        else: # (e1_start, e1_end) in (start_2, end_2) and (e2_start, e2_end) in (start_1, end_1)
            e2_start = e2_start - start_1
            e2_end =  e2_end - start_1
            
            e1_start =  e1_start - start_1 - (len(mid) - 1)
            e1_end =  e1_end - start_1 - (len(mid) - 1)
        
        return context, (e1_start, e1_end, e2_start, e2_end)
        
    else:
        window = (ws-(end_token - start_token)) // 2

        start_token -= window
        start_token = max(0, start_token)
        
        end_token += window
        end_token = min(end_token, len(tokens) - 1)
        
        start = tokens[start_token].idx
        end = tokens[end_token].idx + len(tokens[end_token])
        context = text[start:end]

        e1_start = e1_start - start
        e1_end = e1_end - start
        e2_start = e2_start - start
        e2_end = e2_end - start

        return context, (e1_start, e1_end, e2_start, e2_end)

def add_event_tokens(text, span):
    event1_start, event1_end, event2_start, event2_end = span
    tag_start1, tag_start2, tag_end1, tag_end2 = "[E1] ", "[E2] ", " [/E1]", " [/E2]"
    if event1_start > event2_start and event1_end > event2_start:
    # Hoán đổi giá trị nếu event1_end lớn hơn event2_end
        event1_start, event1_end, event2_start, event2_end = event2_start, event2_end, event1_start, event1_end
        tag_start1, tag_end1, tag_start2, tag_end2 = tag_start2, tag_end2, tag_start1, tag_end1
    
    text = text[:event2_end] + tag_end2 + text[event2_end:]
    text = text[:event2_start] + tag_start2 + text[event2_start:]
    text = text[:event1_end] + tag_end1 + text[event1_end:]
    text = text[:event1_start] + tag_start1 + text[event1_start:]
    return text



def extract_tokens_batch(texts, tokenizer):
    """
    Tự động xử lý token extraction cho cả BERT và RoBERTa.
    - BERT: loại bỏ '##'
    - RoBERTa: loại bỏ 'Ġ' và ký tự byte đặc biệt
    """
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Tokenizer phải là kiểu HuggingFace tokenizer"

    # Mã hóa không thêm special tokens để thấy rõ token gốc
    encodings = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        padding=False,
        truncation=False
    )

    all_tokens = []
    for ids in encodings["input_ids"]:
        toks = tokenizer.convert_ids_to_tokens(ids)

        if "roberta" in tokenizer.name_or_path.lower() or "phobert" in tokenizer.name_or_path.lower():
            # RoBERTa-style cleanup
            toks = [
                tok.replace("Ġ", "")
                   .replace("Ċ", "\n")
                   .replace("ĉ", "")
                   .strip()
                for tok in toks
            ]
        else:
            # BERT-style cleanup
            toks = [tok.replace("##", "") for tok in toks]

        toks = [t for t in toks if t != ""]
        all_tokens.append(toks)

    return all_tokens

def extract_tokens(texts, tokenizer, spacy_nlp):
    """
    Trích xuất song song token BERT/RoBERTa và spaCy.
    Tự động xác định loại tokenizer.
    """
    texts = list(texts)
    n_process = max(1, multiprocessing.cpu_count() - 1)  # chừa 1 core

    # Tokenize bằng transformer
    all_bert_tokens = extract_tokens_batch(texts, tokenizer)

    # Tokenize bằng spaCy (nhanh và batch song song)
    all_spacy_tokens = list(spacy_nlp.pipe(texts, batch_size=64, n_process=n_process))

    return all_bert_tokens, all_spacy_tokens



def alignment_tokens_from_BERT_to_spaCy(all_bert_tokens, all_spacy_tokens, special_tokens):
    bert_to_spacy_mappings = []
    max_alignment_length = 0
    all_bert_special_positions = []
    all_spacy_special_positions = []

    for bert_tokens, spacy_tokens in zip(all_bert_tokens, all_spacy_tokens):
        spacy_tokens = [tok.text for tok in spacy_tokens]
        # print('bert_tokens:',bert_tokens)
        # print('spacy_tokens:',spacy_tokens)
        # Tạo alignment giữa token BERT và spaCy
        alignment = Alignment.from_strings(bert_tokens, spacy_tokens)
        alignment_data = alignment.x2y.data
        bert_to_spacy_mappings.append(alignment_data)
        # max_alignment_length = max(max_alignment_length, len(alignment_data))

        # Tìm các token đặc biệt trong bert_tokens và spacy_tokens (chỉ tính token thực)
        bert_special_pos = [bert_tokens.index(tok) if tok in bert_tokens else -1 for tok in special_tokens]
        spacy_special_pos = [spacy_tokens.index(tok) if tok in spacy_tokens else -1 for tok in special_tokens]

        all_bert_special_positions.append(bert_special_pos)
        all_spacy_special_positions.append(spacy_special_pos)

    return bert_to_spacy_mappings, all_bert_special_positions, all_spacy_special_positions


def _build_spacy_mask_row(bert2spacy, num_spacy, valid_len, max_len):
    """
    bert2spacy: list[int] ánh xạ BERT(pure, không chứa CLS/SEP/PAD) -> spaCy idx
    num_spacy:  số token spaCy của sample
    valid_len:  số token hợp lệ trong input_ids (đã gồm CLS và SEP), = attention_mask.sum()
    max_len:    MAX_LENGTH
    """
    # content tokens = (valid_len - 2)  (bỏ CLS và SEP)
    content_len = max(int(valid_len) - 2, 0)
    # số BERT token thực sự có mặt sau truncation
    num_eff = min(content_len, len(bert2spacy))

    mask = torch.zeros((num_spacy, max_len), dtype=torch.bool)
    # ghi 1 vào các cột 1..num_eff (dịch +1 vì cột 0 là CLS)
    for j in range(num_eff):
        s_idx = bert2spacy[j]
        if 0 <= s_idx < num_spacy:
            mask[s_idx, 1 + j] = 1
    return mask


def _build_entity_masks_from_positions(special_pos, valid_len, max_len):
    """
    special_pos: tuple(e1_start, e1_end, e2_start, e2_end) theo chỉ số BERT 'pure' (không có CLS)
                 và hiểu theo quy tắc 'nằm giữa' => (start, end) là vị trí token đánh dấu, không lấy.
    valid_len:   attention_mask.sum()  (bao gồm CLS và SEP)
    -> Trả về 2 mask [L] bool, đã dịch +1 (bỏ CLS), tự động cắt theo SEP/truncation.
    """
    L = max_len
    sep_idx = int(valid_len) - 1
    content_len = max(int(valid_len) - 2, 0)

    e1_mask = torch.zeros(L, dtype=torch.bool)
    e2_mask = torch.zeros(L, dtype=torch.bool)

    e1_s, e1_e, e2_s, e2_e = special_pos  # indices theo BERT 'pure'
    # Khoảng lấy là (start+1) .. (end-1), với slice [start+1 : end], rồi dịch +1 vì CLS
    # Đồng thời cắt vào [1, sep_idx) (vì SEP ở sep_idx, không lấy)
    s1 = max(e1_s + 1, 0)
    e1 = min(e1_e, content_len)
    if e1 > s1:
        e1_mask[1 + s1 : 1 + e1] = 1  # cột 0 (CLS) luôn 0

    s2 = max(e2_s + 1, 0)
    e2 = min(e2_e, content_len)
    if e2 > s2:
        e2_mask[1 + s2 : 1 + e2] = 1

    # Bảo đảm SEP & PAD không được bật
    if sep_idx >= 0:
        e1_mask[sep_idx:] = 0
        e2_mask[sep_idx:] = 0
    return e1_mask, e2_mask


def custom_collate_fn(batch):
    B = len(batch)
    L = batch[0]['input_ids'].size(0)
    max_M = max(item['spacy_mask'].size(0) for item in batch)
    has_token_type_ids =  'token_type_ids' in batch[0]
    spacy_masks = torch.zeros((B, max_M, L), dtype=torch.bool)
    
    for i, item in enumerate(batch):
        M = item['spacy_mask'].size(0)
        spacy_masks[i, :M, :] = item['spacy_mask']
    if has_token_type_ids:
        collated = {
            'input_ids': torch.stack([b['input_ids'] for b in batch]),
            'token_type_ids': torch.stack([b['token_type_ids'] for b in batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
            'entity_mask_e1': torch.stack([b['entity_mask_e1'] for b in batch]),  # [B, L]
            'entity_mask_e2': torch.stack([b['entity_mask_e2'] for b in batch]),  # [B, L]
            'spacy_masks': spacy_masks,                                            # [B, M, L]
            'spacy_tokens': [b['spacy_tokens'] for b in batch],
            'labels': torch.tensor([b['labels'] for b in batch], dtype=torch.long),
        }
    else:
        collated = {
            'input_ids': torch.stack([b['input_ids'] for b in batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
            'entity_mask_e1': torch.stack([b['entity_mask_e1'] for b in batch]),  # [B, L]
            'entity_mask_e2': torch.stack([b['entity_mask_e2'] for b in batch]),  # [B, L]
            'spacy_masks': spacy_masks,                                            # [B, M, L]
            'spacy_tokens': [b['spacy_tokens'] for b in batch],
            'labels': torch.tensor([b['labels'] for b in batch], dtype=torch.long),
        }
    return collated


class CustomTextDatasetTDD(Dataset):
    def __init__(self, df, window_size, tokenizer_bert, spacy_nlp, MAX_LENGTH, special_tokens, device=torch.device("cpu")):
        self.df = df.copy()
        self.tokenizer = tokenizer_bert
        self.spacy_nlp = spacy_nlp
        self.MAX_LENGTH = MAX_LENGTH
        self.window_size = window_size
        self.special_tokens = special_tokens['additional_special_tokens']
        self.device = device
        self.has_token_type_ids = 'token_type_ids' in self.tokenizer.model_input_names
        
        # Labels
        self.labels = list(df['label_encoded'])
        
        # Chuẩn bị spaCy docs
        n_process = max(1, multiprocessing.cpu_count() - 1)  # chừa 1 core
        # Gán docs vào df
        self.df['tokens'] = list(spacy_nlp.pipe(self.df['text'].tolist(), batch_size=16, n_process=n_process))

        def create_marked_text(row, ws):
            text = row['text']
            tokens = row['tokens']
            e1_start, e1_end = row['entity1_start'], row['entity1_end']
            e2_start, e2_end = row['entity2_start'], row['entity2_end']
        
            context, span = get_context_by_window(text, tokens, e1_start, e1_end, e2_start, e2_end, ws)
            
            marked_text = add_event_tokens(context, span)
            
            marked_e1 = f"[E1] {row['entity1_text']} [/E1]"
            marked_e2 = f"[E2] {row['entity2_text']} [/E2]"
            if marked_e1 in marked_text and marked_e2 in marked_text:
                return marked_text
            else:
                print('context:', context)
                print('span', span)
                print('marked_text',marked_text)
                print("Error row:", row['document_id'])
                print()
                return "Error"

        
        # Marked text
        self.df['marked_text'] = self.df.apply(
            lambda row: create_marked_text(row, window_size), axis=1
        )

        # Tokenize cho BERT
        inputs = self.tokenizer(
            list(self.df['marked_text']),
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']

        if self.has_token_type_ids:
            self.token_type_ids = inputs['token_type_ids']

        
        # Token strings
        self.all_bert_tokens, self.all_spacy_tokens = extract_tokens(self.df['marked_text'], tokenizer_bert, spacy_nlp)

        # Alignment + special positions
        self.bert_to_spacy_mappings, self.all_bert_special_positions, self.all_spacy_special_positions = alignment_tokens_from_BERT_to_spaCy(
            self.all_bert_tokens, self.all_spacy_tokens, self.special_tokens
        )

        # Precompute spaCy masks + entity masks
        self.spacy_masks = []
        self.entity_masks = []
        
        L = self.input_ids.size(1)  # MAX_LENGTH
        
        for i, (bert2spacy, spacy_toks, special_pos) in enumerate(
            zip(self.bert_to_spacy_mappings, self.all_spacy_tokens, self.all_bert_special_positions)
        ):
            valid_len = int(self.attention_mask[i].sum().item())  # gồm CLS & SEP
        
            # 1) spaCy mask [M, L], 0 tại CLS/SEP/PAD
            sp_mask = _build_spacy_mask_row(
                bert2spacy=bert2spacy,
                num_spacy=len(spacy_toks),
                valid_len=valid_len,
                max_len=L
            )
            self.spacy_masks.append(sp_mask)
        
            # 2) entity masks [L], 0 tại CLS/SEP/PAD
            e1_mask, e2_mask = _build_entity_masks_from_positions(
                special_pos=special_pos,
                valid_len=valid_len,
                max_len=L
            )
            self.entity_masks.append((e1_mask, e2_mask))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        e1_mask, e2_mask = self.entity_masks[idx]
        if self.has_token_type_ids:
            return {
                    'input_ids': self.input_ids[idx],
                    'token_type_ids': self.token_type_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'entity_mask_e1': e1_mask,
                    'entity_mask_e2': e2_mask,
                    'spacy_mask': self.spacy_masks[idx],   # [M, L]
                    'spacy_tokens': self.all_spacy_tokens[idx],
                    'labels': self.labels[idx],
                    }
        else:
            return {
                    'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'entity_mask_e1': e1_mask,
                    'entity_mask_e2': e2_mask,
                    'spacy_mask': self.spacy_masks[idx],   # [M, L]
                    'spacy_tokens': self.all_spacy_tokens[idx],
                    'labels': self.labels[idx],
                    }

class CustomTextDatasetI2B2(Dataset):
    def __init__(self, df, window_size, tokenizer_bert, spacy_nlp, MAX_LENGTH, special_tokens, device=torch.device("cpu")):
        self.df = df.copy()
        self.tokenizer = tokenizer_bert
        self.spacy_nlp = spacy_nlp
        self.MAX_LENGTH = MAX_LENGTH
        self.window_size = window_size
        self.special_tokens = special_tokens['additional_special_tokens']
        self.device = device
        self.has_token_type_ids = 'token_type_ids' in self.tokenizer.model_input_names
        
        # Labels
        self.labels = list(df['label_encoded'])
        
        # Chuẩn bị spaCy docs
        doc_texts = self.df.drop_duplicates('document_id')[['document_id', 'text']]
        n_process = max(1, multiprocessing.cpu_count() - 1)  # chừa 1 core
        docs = list(spacy_nlp.pipe(doc_texts['text'].tolist(), batch_size=16, n_process=n_process))
        doc_tokens_map = dict(zip(doc_texts['document_id'], docs))

        def create_marked_text(row, doc_tokens_map, ws):
            text = row['text']
            tokens = doc_tokens_map[row['document_id']]  # lấy sẵn tokens theo doc_id
            e1_start, e1_end = row['entity1_start'], row['entity1_end']
            e2_start, e2_end = row['entity2_start'], row['entity2_end']
            
            context, span = get_context_by_window(text, tokens, e1_start, e1_end, e2_start, e2_end, ws)
            marked_text = add_event_tokens(context, span)
        
            marked_e1 = f"[E1] {row['entity1_text']} [/E1]"
            marked_e2 = f"[E2] {row['entity2_text']} [/E2]"
            if marked_e1 in marked_text and marked_e2 in marked_text:
                return marked_text
            else:
                print("Error row:", row['document_id'])
                return "Error"
        
        
        # Marked text
        self.df['marked_text'] = self.df.apply(
            lambda row: create_marked_text(row, doc_tokens_map, window_size), axis=1
        )

        # Tokenize cho BERT
        inputs = self.tokenizer(
            list(self.df['marked_text']),
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']

        if self.has_token_type_ids:
            self.token_type_ids = inputs['token_type_ids']

        
        # Token strings
        self.all_bert_tokens, self.all_spacy_tokens = extract_tokens(self.df['marked_text'], tokenizer_bert, spacy_nlp)

        # Alignment + special positions
        self.bert_to_spacy_mappings, self.all_bert_special_positions, self.all_spacy_special_positions = alignment_tokens_from_BERT_to_spaCy(
            self.all_bert_tokens, self.all_spacy_tokens, self.special_tokens
        )

        # Precompute spaCy masks + entity masks
        self.spacy_masks = []
        self.entity_masks = []
        
        L = self.input_ids.size(1)  # MAX_LENGTH
        
        for i, (bert2spacy, spacy_toks, special_pos) in enumerate(
            zip(self.bert_to_spacy_mappings, self.all_spacy_tokens, self.all_bert_special_positions)
        ):
            valid_len = int(self.attention_mask[i].sum().item())  # gồm CLS & SEP
        
            # 1) spaCy mask [M, L], 0 tại CLS/SEP/PAD
            sp_mask = _build_spacy_mask_row(
                bert2spacy=bert2spacy,
                num_spacy=len(spacy_toks),
                valid_len=valid_len,
                max_len=L
            )
            self.spacy_masks.append(sp_mask)
        
            # 2) entity masks [L], 0 tại CLS/SEP/PAD
            e1_mask, e2_mask = _build_entity_masks_from_positions(
                special_pos=special_pos,
                valid_len=valid_len,
                max_len=L
            )
            self.entity_masks.append((e1_mask, e2_mask))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        e1_mask, e2_mask = self.entity_masks[idx]
        if self.has_token_type_ids:
            return {
                    'input_ids': self.input_ids[idx],
                    'token_type_ids': self.token_type_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'entity_mask_e1': e1_mask,
                    'entity_mask_e2': e2_mask,
                    'spacy_mask': self.spacy_masks[idx],   # [M, L]
                    'spacy_tokens': self.all_spacy_tokens[idx],
                    'labels': self.labels[idx],
                    }
        else:
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'entity_mask_e1': e1_mask,
                'entity_mask_e2': e2_mask,
                'spacy_mask': self.spacy_masks[idx],   # [M, L]
                'spacy_tokens': self.all_spacy_tokens[idx],
                'labels': self.labels[idx],
                }

# Hàm tạo dataloader
def create_dataloader(df, spacy_nlp, tokenizer_bert, special_tokens, batch_size, window_size, MAX_LENGTH, shuffle=True, is_I2B2=False, is_TDD=False, is_TBD=False, is_MATRES=False):
    if is_I2B2 == True:
        dataset = CustomTextDatasetI2B2(df, window_size, tokenizer_bert, spacy_nlp, MAX_LENGTH, special_tokens)
    
    if is_TDD == True or is_TBD == True or is_MATRES==True:
        dataset = CustomTextDatasetTDD(df, window_size, tokenizer_bert, spacy_nlp, MAX_LENGTH, special_tokens)


    print('Created dataset!')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)
    print('Created dataloader!')
    return dataloader


























