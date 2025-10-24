import os
import pickle
import numpy as np
from tqdm import tqdm

from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

pd.options.mode.copy_on_write = True

def load_cache_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        features = data['features']
        labels = data['labels']
        tag_ids = data['tag_ids']
    return features, labels, tag_ids


class FeaturesDataset(Dataset):
    def __init__(self, features, labels, tag_ids):
        self.features = features
        self.labels = labels
        self.tag_ids = tag_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.tag_ids[index]

def create_dataloader(path, batch_size=32, shuffle=False):
    features, labels, tag_ids = load_cache_data(path)
    dataset = FeaturesDataset(features, labels, tag_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

class BERT_Encoder(nn.Module):
    def __init__(self, tokenizer_bert, bert_model, device):
        super(BERT_Encoder, self).__init__()
        self.tokenizer_bert = tokenizer_bert
        self.bert = bert_model.to(device)
        self.device = device
        
    def get_entity_embeddings(self, last_hidden_state, input_ids, start_token, end_token):
        # get position of start_token và end_token
        start_positions = (input_ids == self.tokenizer_bert.convert_tokens_to_ids(start_token)).nonzero(as_tuple=True)
        end_positions = (input_ids == self.tokenizer_bert.convert_tokens_to_ids(end_token)).nonzero(as_tuple=True)
        
        if start_positions[0].size(0) == 0 or end_positions[0].size(0) == 0:
            raise ValueError(f"Token {start_token} or {end_token} not found in input_ids.")
        
        # take embedding from [start_token] to [end_token]
        embeddings = []
        for batch_idx in range(len(start_positions[0])):
            start_idx = start_positions[1][batch_idx]
            end_idx = end_positions[1][batch_idx]
            embeddings.append(last_hidden_state[batch_idx, start_idx:end_idx + 1, :])
        
        return embeddings
    
    def process_entity_embeddings(self, embeddings):
        return torch.stack([torch.mean(embed, dim=0).detach() for embed in embeddings])  # Mean pooling
        

    def forward(self, ids, marks):
        input_ids_e1s = ids[0].to(self.device)
        input_ids_e2s = ids[1].to(self.device)
        attention_mask_e1s = marks[0].to(self.device)
        attention_mask_e2s = marks[1].to(self.device)
        last_hidden_state_e1 = self.bert(input_ids=input_ids_e1s, attention_mask=attention_mask_e1s).last_hidden_state # (batch_size, seq_len, hidden_size)
        last_hidden_state_e2 = self.bert(input_ids=input_ids_e2s, attention_mask=attention_mask_e2s).last_hidden_state # (batch_size, seq_len, hidden_size)
        
        e1_embeddings = self.get_entity_embeddings(last_hidden_state_e1, input_ids_e1s, "[E1]", "[/E1]")
        e2_embeddings = self.get_entity_embeddings(last_hidden_state_e2, input_ids_e2s, "[E2]", "[/E2]")
        
        e1_embeded = self.process_entity_embeddings(e1_embeddings)
        e2_embeded = self.process_entity_embeddings(e2_embeddings)

        return e1_embeded, e2_embeded

class TemporalDataSet:
    def __init__(self, dataloader, tokenizer_bert, bert_model, device):
        self.dataloader = dataloader
        self.bert_encoder = BERT_Encoder(tokenizer_bert, bert_model, device)
    
    def save_to_pickle(self, features, labels, tag_ids, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump({'features': features, 'labels': labels, 'tag_ids':tag_ids}, f)
        print(file_name, "is saved.")
        
    def create_embeddings(self, save_path):
        features = []
        labels = []
        tag_ids = []
        for x, batch_labels, batch_tag_ids in tqdm(self.dataloader, desc="Processing Batches"):
            e1_embeded, e2_embeded = self.bert_encoder(x[0], x[1])
            feature = torch.cat((e1_embeded, e2_embeded), dim=1)
            features.append(feature)
            labels.append(batch_labels)
            e1_ids = np.array(batch_tag_ids[0])
            e2_ids = np.array(batch_tag_ids[1])
            doc_ids = batch_tag_ids[2].numpy()
            tag_ids_batches = np.column_stack((e1_ids, e2_ids, doc_ids))
            tag_ids.append(tag_ids_batches)
        tag_ids = np.vstack(tag_ids)
        features = torch.cat(features, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        print("Saving dataset")
        self.save_to_pickle(features, labels, tag_ids, file_name=save_path)


def script_separate_context(path_in, path_out, batch_size, tokenizer, bert, device):
    dataloader = create_dataloader(path_in, batch_size=batch_size)
    dataset = TemporalDataSet(dataloader, tokenizer, bert, device)
    train_dataset.create_embeddings(path_out)

if __name__ == '__main__':
    machine = int(input("local: 0, server: 1 --- Choose (0 or 1):"))
    mode = int(input("merged_marked_context: 0, separate_marked_context: 1 --- Choose (0 or 1):"))

    if machine == 0:
        root_code = "/Users/doduydao/daodd/PycharmProjects/phd-dao-do/TRE/"
        root_data = "/Users/doduydao/daodd/PycharmProjects/TRE/data/"
        
    else:
        root_code = "~/daodd/phd-dao-do/TRE/"
        root_data = "/data/ddao/TRE/data/"
    
    
    train_path_in = 'dataset_cache/separate_context/train_merged.pkl'
    validation_path_in = 'dataset_cache/separate_context/validation_merged.pkl'
    test_path_in = 'dataset_cache/separate_context/test_merged.pkl'

    train_path_out = 'embeddings/BERT/separate_context/train_merged.pkl'
    validation_path_out = 'embeddings/BERT/separate_context/validation_merged.pkl'
    test_path_out = 'embeddings/BERT/separate_context/test_merged.pkl'

   
    # Hyperparameters
    bert_model_name = "bert-base-uncased"
    # bert_model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
    # bert_model_name = "emilyalsentzer/Bio_ClinicalBERT"
    # bert_model_name = "medicalai/ClinicalBERT"
    # bert_model_name = "dmis-lab/biobert-v1.1"
    # Device setup
    device = torch.device("mps")
    
    # Tokenizer và DataLoader
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert = BertModel.from_pretrained(bert_model_name)
    
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special_tokens)
    bert.resize_token_embeddings(len(tokenizer))

    batch_size = 16
    script_separate_context(train_path_in, train_path_out, batch_size, tokenizer, bert, device)
    script_separate_context(validation_path_in, validation_path_out, batch_size, tokenizer, bert, device)
    script_separate_context(test_path_in, test_path_out, batch_size, tokenizer, bert, device)