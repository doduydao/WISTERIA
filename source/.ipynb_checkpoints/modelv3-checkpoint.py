import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizerFast, BertModel
from torch.nn import MultiheadAttention

import pandas as pd
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import warnings


import spacy
import re
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np

import time
from torch.cuda.amp import autocast, GradScaler

from spacy.training import Alignment
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
import re



# --- Softmax an toàn cho AMP (fp16) ---
def _safe_masked_softmax(scores, mask, dim):
    """
    scores: [...], mask: bool (True = hợp lệ)
    """
    orig_dtype = scores.dtype
    scores = scores.float()
    scores = scores.masked_fill(~mask, -1e4)              # tránh overflow fp16
    probs  = F.softmax(scores, dim=dim)
    return probs.to(orig_dtype)

# --- Word attention pooling: [B,L,H] x [B,M,L] -> [B,M,H] ---
def masked_word_attention_pooling(embeddings, spacy_masks):
    """
    embeddings: [B, L, H]
    spacy_masks: [B, M, L] (bool), 1 tại các BERT token thuộc spaCy token
    return: [B, M, H]  (giữ padding theo M)
    """
    B, L, H = embeddings.size()
    assert spacy_masks.size(0) == B and spacy_masks.size(2) == L
    m = spacy_masks.float()                                              # [B, M, L]

    # query = mean theo mask = (m @ emb) / count
    # sum_masked: [B, M, H]
    sum_masked = torch.einsum('bml,blh->bmh', m, embeddings)
    counts     = m.sum(dim=2, keepdim=True).clamp(min=1.0)               # [B, M, 1]
    query      = sum_masked / counts                                     # [B, M, H]

    # scores = emb · query  -> [B, M, L]
    scores = torch.einsum('blh,bmh->bml', embeddings, query)

    # softmax an toàn theo mask (dim=2 trên L)
    weights = _safe_masked_softmax(scores, spacy_masks, dim=2)           # [B, M, L]

    # weighted sum -> word_embs: [B, M, H]
    word_embs = torch.einsum('bml,blh->bmh', weights, embeddings)

    # Nếu hàng (từng spaCy token) hoàn toàn rỗng (do truncation), ép output = 0 để tránh rò nhiễu
    row_has_any = spacy_masks.any(dim=2).unsqueeze(-1).float()           # [B, M, 1]
    word_embs   = word_embs * row_has_any

    # Dọn NaN/Inf phòng hờ
    word_embs = torch.nan_to_num(word_embs, nan=0.0, posinf=0.0, neginf=0.0)
    return word_embs

# --- Entity pooling: [B,L,H] x [B,L] -> [B,H] (einsum, không tạo 4D) ---
def extract_entity_embeddings(embeddings, e1_mask, e2_mask):
    """
    embeddings: [B, L, H]
    e*_mask:    [B, L] (bool)
    return: e1_embs, e2_embs  ([B, H])
    """
    def pool(embs, mask_bool):
        m = mask_bool.float()                                            # [B, L]

        # query = mean theo mask
        sum_masked = torch.einsum('bl,blh->bh', m, embs)                 # [B, H]
        counts     = m.sum(dim=1, keepdim=True).clamp(min=1.0)           # [B, 1]
        query      = sum_masked / counts                                 # [B, H]

        # scores = emb · query  -> [B, L]
        scores  = torch.einsum('blh,bh->bl', embs, query)
        weights = _safe_masked_softmax(scores, mask_bool, dim=1)         # [B, L]

        out = torch.einsum('bl,blh->bh', weights, embs)                  # [B, H]

        # nếu mask rỗng -> ra 0
        has_any = mask_bool.any(dim=1, keepdim=True).float()             # [B,1]
        out = out * has_any

        # dọn NaN/Inf
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return pool(embeddings, e1_mask), pool(embeddings, e2_mask)

# --- Hàm tổng hợp giữ nguyên interface ---
def apply_entity_and_spacy_masks(embeddings, entity_mask_e1, entity_mask_e2, spacy_masks):
    e1_embs, e2_embs = extract_entity_embeddings(embeddings, entity_mask_e1, entity_mask_e2)
    word_embs = masked_word_attention_pooling(embeddings, spacy_masks)   # [B, M, H] (M đã pad)
    return e1_embs, e2_embs, word_embs

# ============================================================
# Positional Encoding
# ============================================================
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, dim)

    def forward(self, x):
        B, M, H = x.size()
        pos_ids = torch.arange(0, M, device=x.device).unsqueeze(0).expand(B, M)
        return x + self.pos_emb(pos_ids)


# ============================================================
# Submodules
# ============================================================
class ResidualLayer(nn.Module):
    def __init__(self, dim, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(p)

    def forward(self, x, sublayer_out):
        return self.ln(x + self.drop(sublayer_out))


class CrossAttention(nn.Module):
    """Multi-head cross-attention: Q (B,Nq,H), K/V (B,M,H)"""
    def __init__(self, dim, num_heads=8, p=0.1, batch_first=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=p, batch_first=batch_first)
        self.res = ResidualLayer(dim, p)

    def forward(self, q, kv, kv_mask=None):
        out, attn = self.mha(q, kv, kv, key_padding_mask=kv_mask)  # attn: (B, Nq, M)
        out = self.res(q, out)
        return out, attn


class LightContextEncoder(nn.Module):
    """2-layer TransformerEncoder để refine word_embs"""
    def __init__(self, dim, num_heads=8, num_layers=2, p=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=p,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x, mask_bool):
        return self.enc(x, src_key_padding_mask=mask_bool)


class Biaffine(nn.Module):
    """Biaffine scorer: e1,e2 -> logits"""
    def __init__(self, dim, num_classes):
        super().__init__()
        self.U = nn.Parameter(torch.empty(num_classes, dim, dim))
        nn.init.xavier_uniform_(self.U)
        self.W = nn.Linear(2 * dim, num_classes)
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, e1, e2):
        t1 = torch.einsum("bd,cdh,bh->bc", e1, self.U, e2)  # (B,C)
        t2 = self.W(torch.cat([e1, e2], dim=-1))           # (B,C)
        return t1 + t2 + self.b


def extract_entity_tokens(spacy_tokens):
    e1_tokens = []
    e2_tokens = []

    for i in range(len(spacy_tokens)):
        e1t = []
        e2t = []
        e1_do_append = False
        e2_do_append = False
        for token in spacy_tokens[i]:
            if e1_do_append:
                e1t.append(token)
                
            if token.text == '[E1]':
                e1_do_append = True
            if token.text == '[/E1]':
                if len(e1t) > 0:
                    e1t.pop()
                else:
                    continue
                e1_do_append = False
            
            if e2_do_append:
                e2t.append(token)
            if token.text == '[E2]':
                e2_do_append = True
            if token.text == '[/E2]':
                if len(e2t) > 0:
                    e2t.pop()
                else:
                    continue
                e2_do_append = False
        e1_tokens.append(e1t)
        e2_tokens.append(e2t)
    return e1_tokens, e2_tokens


class TemporalRelationModelV2_VagueHead(nn.Module):
    def __init__(self,
                 bert,
                 tokenizer,
                 hidden_ffn_dim,
                 num_heads=8,
                 num_layers=8,
                 dropout=0.1,
                 num_classes=3,
                 device=torch.device("cpu")):
        super().__init__()
        self.bert = bert.to(device)
        self.tokenizer = tokenizer
        self.input_dim = bert.config.hidden_size
        self.device = device
        self.num_classes = num_classes

        # Positional + context encoder
        self.pos_enc = LearnablePositionalEncoding(self.input_dim)
        self.ctx_enc = LightContextEncoder(self.input_dim, num_heads=num_heads, num_layers=num_layers, p=dropout)

        # Projector cho h_pair
        self.projector = nn.Linear(self.input_dim * 2, self.input_dim)

        # Cross-attention
        self.cross_e1 = CrossAttention(self.input_dim, num_heads=num_heads, p=dropout)
        self.cross_e2 = CrossAttention(self.input_dim, num_heads=num_heads, p=dropout)
        self.cross_pair = CrossAttention(self.input_dim, num_heads=num_heads, p=dropout)

        # Label embeddings + gate
        self.label_emb = nn.Parameter(torch.randn(num_classes, self.input_dim))
        nn.init.xavier_uniform_(self.label_emb)
        self.gate_fc = nn.Linear(self.input_dim * 2, 1)

        # Heads chính (giữ nguyên)
        self.biaffine = Biaffine(self.input_dim, num_classes)
        self.context_head = nn.Sequential(
            nn.Linear(self.input_dim * 3, hidden_ffn_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_ffn_dim, num_classes)
        )

        # Late fusion weight (giữ nguyên)
        self.logit_mixer = nn.Parameter(torch.tensor([0.5, 0.5]))

        # --- Head phụ: nhận diện VAGUE ---
        self.vague_head = nn.Sequential(
            nn.Linear(self.input_dim * 5, hidden_ffn_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_ffn_dim, 1)
        )

        # LayerNorm
        self.ln_e = nn.LayerNorm(self.input_dim)
        self.ln_c = nn.LayerNorm(self.input_dim)

    # -------------------------
    # Helper (giữ nguyên)
    # -------------------------
    @staticmethod
    def _masked_topk(attn, valid_mask, k):
        B, M = attn.shape
        k_eff = min(k, M)
        attn_masked = attn.masked_fill(~valid_mask, attn.new_tensor(-1e4))
        topk_vals, topk_idx = torch.topk(attn_masked, k_eff, dim=1)
        return topk_idx, topk_vals

    def _topk_context(self, attn, word_embs, word_valid_mask, k_override=None):
        B, M, H = word_embs.size()
        k = k_override
        if k <= 0:
            denom = word_valid_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
            ctx = (word_embs * word_valid_mask.unsqueeze(-1)).sum(dim=1) / denom.squeeze(-1)
            return ctx, None, None
        k_eff = min(k, M)
        attn_masked = attn.masked_fill(~word_valid_mask, attn.new_tensor(-1e4))
        topk_vals, topk_idx = torch.topk(attn_masked, k_eff, dim=1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, H)
        topk_embs = torch.gather(word_embs, 1, gather_idx)
        weights = F.softmax(topk_vals.float(), dim=1).to(topk_vals.dtype).unsqueeze(-1)
        ctx = torch.sum(weights * topk_embs, dim=1)
        return ctx, topk_idx, topk_vals

    def _label_aware_fusion(self, ctx_pair):
        B, H = ctx_pair.size()
        C = self.num_classes
        label = self.label_emb
        label_exp = label.unsqueeze(0).expand(B, -1, -1)
        ctx_exp = ctx_pair.unsqueeze(1).expand(-1, C, -1)
        z = torch.cat([ctx_exp, label_exp], dim=-1)
        z_flat = z.view(B*C, -1)
        gates = torch.sigmoid(self.gate_fc(z_flat))
        return gates.view(B, C)

    def _late_fusion(self, logits_list):
        w = torch.softmax(self.logit_mixer, dim=0)
        return w[0]*logits_list[0] + w[1]*logits_list[1]

    # -------------------------
    # Forward
    # -------------------------
    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                entity_mask_e1,
                entity_mask_e2,
                spacy_masks,
                spacy_tokens=None,
                top_k=None,
                return_topk_tokens=False,
                return_aux_logits=False):
        # --- Encode ---
        if token_type_ids is None:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # (B,L,H)

        # --- Entities & context ---
        e1_embs, e2_embs, word_embs = apply_entity_and_spacy_masks(embeddings, entity_mask_e1, entity_mask_e2, spacy_masks)
        e1_embs = self.ln_e(e1_embs)
        e2_embs = self.ln_e(e2_embs)

        word_valid_mask = spacy_masks.any(dim=2)
        pad_words = ~word_valid_mask

        word_ctx = self.pos_enc(word_embs)
        word_ctx = self.ctx_enc(word_ctx, pad_words)
        word_ctx = self.ln_c(word_ctx)

        # --- Pair rep ---
        h_pair = self.projector(torch.cat([e1_embs, e2_embs], dim=-1))

        # --- Cross-attn ---
        q_e1, attn_e1 = self.cross_e1(e1_embs.unsqueeze(1), word_ctx, kv_mask=pad_words)
        q_e2, attn_e2 = self.cross_e2(e2_embs.unsqueeze(1), word_ctx, kv_mask=pad_words)
        q_pair, attn_pair = self.cross_pair(h_pair.unsqueeze(1), word_ctx, kv_mask=pad_words)
        attn_e1, attn_e2, attn_pair = attn_e1.squeeze(1), attn_e2.squeeze(1), attn_pair.squeeze(1)

        # --- Context vectors ---
        ctx_e1, idx_e1, val_e1 = self._topk_context(attn_e1, word_ctx, word_valid_mask, k_override=top_k)
        ctx_e2, idx_e2, val_e2 = self._topk_context(attn_e2, word_ctx, word_valid_mask, k_override=top_k)
        ctx_pair, idx_pair, val_pair = self._topk_context(attn_pair, word_ctx, word_valid_mask, k_override=top_k)
        gates = self._label_aware_fusion(ctx_pair)

        # --- Heads ---
        logits_biaff = self.biaffine(e1_embs, e2_embs)
        logits_context = self.context_head(torch.cat([ctx_e1, ctx_e2, ctx_pair], dim=-1))
        logits_context = logits_context * gates
        logits = self._late_fusion([logits_biaff, logits_context])

        # --- Head phụ (VAGUE) ---
        fusion_inp = torch.cat([e1_embs, e2_embs, ctx_e1, ctx_e2, ctx_pair], dim=-1)
        vague_logits = self.vague_head(fusion_inp).squeeze(-1)

        # --- Logging ---
        if return_topk_tokens:
            words = spacy_tokens if spacy_tokens is not None else [[""]] * input_ids.size(0)
            logs = []
            e1_tokens, e2_tokens = extract_entity_tokens(spacy_tokens)
            for i in range(len(words)):
                def safe_pick(idxs):
                    if idxs is None: return []
                    return [(words[i][j.item()] if j.item() < len(words[i]) else "<PAD>") for j in idxs[i].detach().cpu()]
                
                logs.append({
                    "e1_tokens": e1_tokens[i],
                    "e2_tokens": e2_tokens[i],
                    "e1_topk_tokens": safe_pick(idx_e1),
                    "e1_scores": [] if val_e1 is None else val_e1[i].detach().cpu().tolist(),
                    "e2_topk_tokens": safe_pick(idx_e2),
                    "e2_scores": [] if val_e2 is None else val_e2[i].detach().cpu().tolist(),
                    "pair_topk_tokens": safe_pick(idx_pair),
                    "pair_scores": [] if val_pair is None else val_pair[i].detach().cpu().tolist(),
                    "vague_logit": vague_logits[i].item()
                })
            return logits, vague_logits, {top_k: logs}

        if return_aux_logits:
            return logits, vague_logits, [logits_biaff, logits_context]

        return logits, vague_logits