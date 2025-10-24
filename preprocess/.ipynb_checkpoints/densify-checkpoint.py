import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import spacy
import re
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import os
import matplotlib.pyplot as plt
pd.options.mode.copy_on_write = True

# Định nghĩa luật chuyển đổi label
transitivity_rules = {
    ("BEFORE", "BEFORE"): "BEFORE",
    ("AFTER", "AFTER"): "AFTER",
    ("OVERLAP", "OVERLAP"):"OVERLAP",
    ("AFTER", "OVERLAP"): "AFTER",
    ("OVERLAP", "AFTER"): "AFTER",
    ("BEFORE", "OVERLAP"): "BEFORE",
    ("OVERLAP", "BEFORE"): "BEFORE"
}

symmetry_rules = {
    "BEFORE": "AFTER",
    "AFTER": "BEFORE",
    "OVERLAP": "OVERLAP"
}

def add_symmetry(df, symmetry_rules):
    seen_rows = set((row["entity1_id"], row["entity2_id"], row["document_id"], row["label"]) for _, row in df.iterrows())
    # Lọc các hàng có nhãn nằm trong symmetry_rules
    valid_rows = df[df["label"].isin(symmetry_rules)]
    # Áp dụng quy tắc đối xứng để tạo nhãn mới
    valid_rows = valid_rows.assign(
        new_entity1_id=valid_rows["entity2_id"],
        new_entity2_id=valid_rows["entity1_id"],
        new_entity1_start=valid_rows["entity2_start"],
        new_entity2_start=valid_rows["entity1_start"],
        new_entity1_end=valid_rows["entity2_end"],
        new_entity2_end=valid_rows["entity1_end"],
        new_entity1_text=valid_rows["entity2_text"],
        new_entity2_text=valid_rows["entity1_text"],
        new_label=valid_rows["label"].map(symmetry_rules)
    )
    # Tạo hàng mới dưới dạng tuple để kiểm tra trùng lặp
    valid_rows["new_row"] = valid_rows.apply(
        lambda row: (
            row["new_entity1_id"], row["new_entity2_id"], row["document_id"], row["new_label"]
        ),
        axis=1
    )
    # Lọc các hàng chưa có trong seen_rows
    unique_valid_pairs = valid_rows[~valid_rows["new_row"].isin(seen_rows)]
    new_rows_unseen = set(unique_valid_pairs["new_row"].tolist())
    seen_rows.update(new_rows_unseen)
    
    # Tạo DataFrame các hàng mới
    new_rows = unique_valid_pairs[[
        "new_entity1_id", "new_entity2_id",
        "new_entity1_start", "new_entity2_start",
        "new_entity1_end", "new_entity2_end",
        "new_entity1_text", "new_entity2_text",
        "document_id", "text", "new_label"
    ]].rename(columns={
        "new_entity1_id": "entity1_id",
        "new_entity2_id": "entity2_id",
        "new_entity1_start": "entity1_start",
        "new_entity2_start": "entity2_start",
        "new_entity1_end": "entity1_end",
        "new_entity2_end": "entity2_end",
        "new_entity1_text": "entity1_text",
        "new_entity2_text": "entity2_text",
        "new_label": "label"
    })
    
    print("Generated:", len(new_rows), "new symmetry relations")
    df = pd.concat([df, new_rows], ignore_index=True)
    return df

def add_transtivity(df, transitivity_rules):
    seen_rows = set((row["entity1_id"], row["entity2_id"], row["document_id"], row["label"]) for _, row in df.iterrows())
    new_rows = []
    grouped = df.groupby('document_id')
    for group_id, group in grouped:
        # Merge để tìm các cặp hàng thỏa mãn entity2_id của hàng trước = entity1_id của hàng sau
        merged = pd.merge(group, group, left_on="entity2_id", right_on="entity1_id", suffixes=("_1", "_2"))
        # Loại bỏ các dòng có entity1_id_1 = entity2_id_2
        merged = merged[merged["entity1_id_1"] != merged["entity2_id_2"]]
        merged = merged[merged["entity1_id_1"] != merged["entity2_id_2"]]
        # print("merged:", len(merged))
        # Bỏ các dòng mà entity1_start_1 == entity2_start_2 và entity1_end_1 == entity2_end_2
        merged = merged[~((merged["entity1_start_1"] == merged["entity2_start_2"]) & 
                          (merged["entity1_end_1"] == merged["entity2_end_2"]))]

        
        # Lọc các cặp hàng thỏa mãn quy tắc chuyển tiếp
        merged["label_pair"] = list(zip(merged["label_1"], merged["label_2"]))
        valid_pairs = merged[merged["label_pair"].isin(transitivity_rules)]
        # print("valid_pairs:", len(valid_pairs))
        # Tạo các hàng mới dựa trên quy tắc chuyển tiếp
        valid_pairs["new_label"] = valid_pairs["label_pair"].map(transitivity_rules)
    
        # Tạo hàng mới dưới dạng tuple để kiểm tra trùng lặp
        valid_pairs["new_row"] = valid_pairs.apply(
                lambda row: (
                    str(row["entity1_id_1"]), str(row["entity2_id_2"]), row["document_id_1"], row["new_label"]),
                axis=1
            )
    
        unique_valid_pairs = valid_pairs[~valid_pairs["new_row"].isin(seen_rows)]
        # print(f"group: {group_id}; items: {len(group)}, unique_valid_pairs: {len(unique_valid_pairs)}")
        new_rows_unseen = set(unique_valid_pairs["new_row"].tolist())
        seen_rows.update(new_rows_unseen)
    
        # Tạo DataFrame các hàng mới
        new_rows_group = unique_valid_pairs[[
            "entity1_id_1","entity2_id_2",
            "entity1_start_1", "entity2_start_2",
            "entity1_end_1", "entity2_end_2",
            "entity1_text_1", "entity2_text_2",
            "document_id_1", "text_1", "new_label"
        ]].rename(columns={
            "entity1_id_1": "entity1_id",
            "entity2_id_2": "entity2_id",
            "entity1_start_1": "entity1_start",
            "entity2_start_2": "entity2_start",
            "entity1_end_1": "entity1_end",
            "entity2_end_2": "entity2_end",
            "entity1_text_1": "entity1_text",
            "entity2_text_2": "entity2_text",
            "document_id_1": "document_id",
            "text_1": "text",
            "new_label": "label"
        })
        # print("-----")

        new_rows.append(new_rows_group)
    new_rows_df = pd.concat(new_rows, ignore_index=True)
    print("Generated:", len(new_rows_df), "new transitivity relations")
    df = pd.concat([df, new_rows_df], ignore_index=True)
    return df


def data_augmentation(df, transitivity_rules, symmetry_rules):
    previous_len_df = len(df)
    path = 0
    while path!=1:
        # print("before_add_symmetry:", len(df))
        df = add_symmetry(df, symmetry_rules)
        # print("after_add_symmetry:", len(df))
        # df = add_transtivity(df, transitivity_rules)
        # print("after_add_transtivity:", len(df))
        current_len_df = len(df)
        path+=1
        if current_len_df != previous_len_df:
            
            previous_len_df = current_len_df
        else:
            break
    return df

if __name__ == '__main__':
    # Load dataframe (replace with actual data)
    data_source = "../data/i2b2_2012/data_processed/"
    # data_source =  "/Users/doduydao/daodd/PycharmProjects/TRE/data/small_test/data_processed/"
    data_train_path = data_source + "train_merged_full.csv"
    data_test_path = data_source + "test_merged_full.csv"
    
    train_df = pd.read_csv(data_train_path)
    test_df = pd.read_csv(data_test_path)
    print("train_df:", train_df.shape)
    print("test_df:", test_df.shape)
    #Augmentation for training set
    print('Augmentation for training set')
    print('Before:', train_df.shape)
    new_train_df = data_augmentation(train_df, transitivity_rules, symmetry_rules)
    print('After:', new_train_df.shape)
    print()
    aug_train = data_source + "only_symmetry_train_merged_full.csv"
    new_train_df.to_csv(aug_train, index=False)
