import os
import pickle

import torch
import numpy as np

from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader

from transformers import BertTokenizerFast, BertModel

def load_cache_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        features = data['features']
        labels = data['labels']
        tag_ids = data['tag_ids']
    return features, labels, tag_ids

# Hàm attention pooling
def attention_pooling(span_embs):
    query = span_embs.mean(dim=0)
    scores = torch.matmul(span_embs, query)
    weights = torch.softmax(scores, dim=0).unsqueeze(1)
    return (span_embs * weights).sum(dim=0)

# Hàm gộp word-level embedding từ subword
def get_word_level_embs(embeddings, word_ids, tokens, entity1_tokens, entity2_tokens):
    word_vectors = []
    current_word_id = None
    current_vecs = []
    current_token_pieces = []

    for i, word_id in enumerate(word_ids):
        if word_id == -1:
            continue
            
        token = tokens[i]
        if word_id != current_word_id:
            if current_vecs:
                word_vector = attention_pooling(torch.stack(current_vecs, dim=0))
                word_text = "".join([t.replace("##", "") if t.startswith("##") else f" {t}" for t in current_token_pieces]).strip()
                word_vectors.append((word_text, word_vector))
            current_vecs = [embeddings[i]]
            current_token_pieces = [token]
            current_word_id = word_id
        else:
            current_vecs.append(embeddings[i])
            current_token_pieces.append(token)

    if current_vecs:
        word_vector = attention_pooling(torch.stack(current_vecs, dim=0))
        word_text = "".join([t.replace("##", "") if t.startswith("##") else f" {t}" for t in current_token_pieces]).strip()
        word_vectors.append((word_text, word_vector))

    # Lọc các token đặc biệt và dấu câu
    skip_tokens = ["[CLS]", "[SEP]", "[E1]", "[/E1]", "[E2]", "[/E2]"] + entity1_tokens + entity2_tokens
    
    return [(w, v) for w, v in word_vectors if w not in skip_tokens and w not in string.punctuation]


def extract_event_word_embeddings(embeddings, tokens, word_ids):
    e1_start_idx = tokens.index("[E1]") + 1
    e1_end_idx = tokens.index("[/E1]") - 1
    e2_start_idx = tokens.index("[E2]") + 1
    e2_end_idx = tokens.index("[/E2]") - 1

    e1_embedding = attention_pooling(embeddings[e1_start_idx:e1_end_idx + 1])
    e2_embedding = attention_pooling(embeddings[e2_start_idx:e2_end_idx + 1])

    entity1_tokens = tokens[e1_start_idx:e1_end_idx + 1]
    entity2_tokens = tokens[e2_start_idx:e2_end_idx + 1]

    word_vec_pairs = get_word_level_embs(embeddings, word_ids, tokens, entity1_tokens, entity2_tokens)
    words, word_embs = zip(*word_vec_pairs)
    return (e1_embedding, e2_embedding, word_embs, words)

def extract_event_embeddings(embeddings, tokens, word_ids):
    e1_start_idx = tokens.index("[E1]") + 1
    e1_end_idx = tokens.index("[/E1]") - 1
    e2_start_idx = tokens.index("[E2]") + 1
    e2_end_idx = tokens.index("[/E2]") - 1

    e1_embedding = attention_pooling(embeddings[e1_start_idx:e1_end_idx + 1])
    e2_embedding = attention_pooling(embeddings[e2_start_idx:e2_end_idx + 1])
    return torch.cat((e1_embedding, e2_embedding), dim=0)


def create_embedding(features, labels, tag_ids, batch_size, bert, tokenizer, only_entity=True):
    input_ids, token_type_ids, attention_mask, word_ids_tensor = features[0], features[1], features[2], features[3]
    
    dataset = TensorDataset(input_ids, token_type_ids, attention_mask, word_ids_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    features = []
 
    for batch in tqdm(dataloader, desc="Processing Batches"):
        input_ids_batch, token_type_ids_batch, attention_mask_batch, word_ids_batch = batch
        embeddings_batch = bert(input_ids=input_ids_batch.to(device),
                                token_type_ids=token_type_ids_batch.to(device),
                                attention_mask=attention_mask_batch.to(device)).last_hidden_state.detach().cpu()
        
        for i in range(embeddings_batch.size(0)):
            if only_entity:
                features.append(extract_event_embeddings(embeddings_batch[i],
                                                         tokenizer.convert_ids_to_tokens(input_ids_batch[i]),
                                                         word_ids_batch[i]))
            else:
                features.append(extract_event_word_embeddings(embeddings_batch[i], 
                                                          tokenizer.convert_ids_to_tokens(input_ids_batch[i]), 
                                                          word_ids_batch[i]))
    features = torch.cat(features, dim=0)

    return {'features': features, 'labels': torch.tensor(labels), 'tag_ids': tag_ids}

def script_merged_context(path_in, path_out, batch_size, only_entity, bert, tokenizer, device):
    features, labels, tag_ids = load_cache_data(path_in)
    bert = bert.to(device)
    
    data = create_embedding(features, labels, tag_ids, batch_size, bert, tokenizer, only_entity)
    
    with open(path_out, 'wb') as f:
        pickle.dump(data, f)
    print(path_out, "is saved.")



if __name__ == '__main__':
    train_save_path_in = 'dataset_cache/merged_context/train_merged_BERT.pkl'
    validation_save_path_in = 'dataset_cache/merged_context/validation_merged_BERT.pkl'
    test_save_path_in = 'dataset_cache/merged_context/test_merged_BERT.pkl'

    train_save_path_out = 'embeddings/BERT/merged_context/train_merged_only_entity.pkl'
    validation_save_path_out = 'embeddings/BERT/merged_context/validation_merged_only_entity.pkl'
    test_save_path_out = 'embeddings/BERT/merged_context/test_merged_only_entity.pkl'
    
    # Hyperparameters
    bert_model_name = "bert-base-uncased"
    # bert_model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
    # bert_model_name = "emilyalsentzer/Bio_ClinicalBERT"
    # bert_model_name = "medicalai/ClinicalBERT"
    # bert_model_name = "dmis-lab/biobert-v1.1"
    # Device setup
    device = torch.device("mps")


    # Tokenizer và DataLoader
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    bert = BertModel.from_pretrained(bert_model_name)
    
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special_tokens)
    bert.resize_token_embeddings(len(tokenizer))

    batch_size = 16
    only_entity = True
    
    # script_merged_context(train_save_path_in, train_save_path_out, batch_size, only_entity)
    script_merged_context(validation_save_path_in, validation_save_path_out, batch_size, only_entity, bert, tokenizer, device)
    script_merged_context(test_save_path_in, test_save_path_out, batch_size, only_entity, bert, tokenizer, device)
    