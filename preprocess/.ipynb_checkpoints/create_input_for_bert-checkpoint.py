import os
import re
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizerFast

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


# Tokenizer và DataLoader
nlp_spacy = English()
nlp_spacy.add_pipe("senter")
nlp_spacy.initialize()
tokenizer_spacy = Tokenizer(nlp_spacy.vocab)

bert_model_name = "bert-base-uncased"
tokenizer_bert = BertTokenizerFast.from_pretrained(bert_model_name)
special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
tokenizer_bert.add_special_tokens(special_tokens)

def get_token_for_char(tokens, char_idx):
    for i, token in enumerate(tokens):
        if char_idx > token.idx:
            continue
        if char_idx == token.idx:
            return i, token
        if char_idx < token.idx:
            return i - 1, tokens[i - 1]
    return len(tokens) - 1, tokens[len(tokens) - 1]

def get_context_by_window_for_each_entity(text, tokens, e1_start, e1_end, e2_start, e2_end, ws):
    start = min(e1_start, e2_start)
    end = max(e1_end, e2_end)
    
    start_token, _ = get_token_for_char(tokens, start)
    end_token, _ = get_token_for_char(tokens, end)

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
            # print("This case")
            e1_start = e1_start - start_1
            e1_end =  e1_end - start_1
            
            e2_start =  e2_start - start_1 - (len(mid) - 1)
            e2_end =  e2_end - start_1 - (len(mid) - 1)
        else: # (e1_start, e1_end) in (start_2, end_2) and (e2_start, e2_end) in (start_1, end_1)
            e2_start = e2_start - start_1
            e2_end =  e2_end - start_1
            
            e1_start =  e1_start - start_1 - (len(mid) - 1)
            e1_end =  e1_end - start_1 - (len(mid) - 1)
        
        # return context, (e1_start, e1_end, e2_start-len(mid), e2_end-len(mid))
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


def add_entity_tokens(text, span):
    entity1_start, entity1_end, entity2_start, entity2_end = span
    
    tag_start1, tag_start2, tag_end1, tag_end2 = "[E1]", "[E2]", "[/E1]", "[/E2]"
    text = text[:entity1_start] + tag_start1 + text[entity1_start:]
    if entity1_end >= entity1_start:
        entity1_end += len(tag_start1)
    if entity2_start >= entity1_start:
        entity2_start += len(tag_start1)
    if entity2_end >= entity1_start:
        entity2_end += len(tag_start1)
    if entity1_start >= entity1_start:
        entity1_start += len(tag_start1)

    text = text[:entity1_end] + tag_end1 + text[entity1_end:]
    if entity1_start > entity1_end:
        entity1_start += len(tag_end1)
    if entity2_start > entity1_end:
        entity2_start += len(tag_end1)
    if entity2_end > entity1_end:
        entity2_end += len(tag_end1)

    if max(entity1_start, entity2_start) < min(entity1_end, entity2_end):
        return text, entity1_start, entity1_end, entity1_start, entity1_end

    text = text[:entity2_start] + tag_start2 + text[entity2_start:]
    if entity1_start >= entity2_start:
        entity1_start += len(tag_start2)
    if entity1_end >= entity2_start:
        entity1_end += len(tag_start2)
    if entity2_end >= entity2_start:
        entity2_end += len(tag_start2)
    if entity2_start >= entity2_start:
        entity2_start += len(tag_start2)

    text = text[:entity2_end] + tag_end2 + text[entity2_end:]
    if entity1_start >= entity2_end:
        entity1_start += len(tag_end2)
    if entity1_end >= entity2_end:
        entity1_end += len(tag_end2)
    if entity2_start >= entity2_end:
        entity2_start += len(tag_end2)
    return text


def create_marked_text(row, ws):
    text = row['text']
    e1_start = row['entity1_start']
    e1_end = row['entity1_end']
    e2_start = row['entity2_start']
    e2_end = row['entity2_end']
    e1_id = row['entity1_id']
    e2_id = row['entity2_id']
    doc_id = row['document_id']
    tokens = tokenizer_spacy(text)
    
    context, span = get_context_by_window_for_each_entity(text, tokens, e1_start, e1_end, e2_start, e2_end, ws)
    
    marked_text = add_entity_tokens(context, span)

    marked_e1 = "[E1]"+row['entity1_text']+"[/E1]"
    marked_e2 = "[E2]"+row['entity2_text']+"[/E2]"
    if marked_e1 in marked_text and marked_e2 in marked_text:
        return marked_text 
    else:
        print("Error row:",row)
        return "Error"

def create_cache_dataset_with_marked_context_merged(df, save_path, window_size, MAX_LENGTH):
    prefix = '/'.join(save_path.split('/')[:-1])
    if os.path.exists(prefix) is False:
        raise ValueError(prefix, "is not existed.")
    print("Number of item in dataset:", len(df))
    
    print("Creating marked text as merged context:...")
    df['marked_text'] = df.apply(lambda row: create_marked_text(row, window_size), axis=1)
    print("Created marked text!")
    
    inputs = tokenizer_bert(list(df['marked_text']),
                            add_special_tokens=True, 
                            max_length=MAX_LENGTH, 
                            padding="max_length", 
                            truncation=True, 
                            return_tensors="pt")
    
    input_ids = inputs['input_ids'] #[N, MAX_LENGTH]
    token_type_ids = inputs['token_type_ids'] #[N, MAX_LENGTH]
    attention_mask = inputs['attention_mask'] #[N, MAX_LENGTH]
    word_ids_arr = np.array([inputs.word_ids(i) for i in range(len(df['marked_text']))], dtype=object)
    word_ids_arr[word_ids_arr == None] = -1
    word_ids_tensor = torch.tensor(word_ids_arr.astype(np.int64), dtype=torch.long)
    
    features = [input_ids, token_type_ids, attention_mask, word_ids_tensor]

    tag_ids = [list(df['entity1_id']), list(df['entity2_id']), list(df['document_id'])]

    labels = list(df['label_encoded'])

    print("Saving dataset at:", save_path)
    save_to_pickle(features, labels, tag_ids, file_name=save_path)
    

def save_to_pickle(features, labels, tag_ids, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump({'features': features, 'labels': labels, 'tag_ids':tag_ids}, f)
        print(file_name, "is saved.")

# === 2. Dataset Class ===
class TemporalRelationDataset:
    def __init__(self, dataframe,
                 nlp_spacy,
                 tokenizer_spacy, 
                 tokenizer_bert, 
                 max_length, 
                 window_size=128):
        self.dataframe = dataframe
        self.nlp = nlp_spacy
        self.tokenizer_spacy = tokenizer_spacy
        self.tokenizer_bert = tokenizer_bert
        
        self.MAX_LENGTH = max_length
        self.window_size = window_size

        
    def __len__(self):
        return len(self.dataframe)
    
    def make_ids_for_bert(self, index):
        row = self.dataframe.iloc[index]
        text = row['text']
        e1_start = row['entity1_start']
        e1_end = row['entity1_end']
        e2_start = row['entity2_start']
        e2_end = row['entity2_end']
        e1_id = row['entity1_id']
        e2_id = row['entity2_id']
        doc_id = row['document_id']

        marked_text_e1, marked_text_e2 = self.create_marked_text(text, e1_start, e1_end, e2_start, e2_end)

        # Token hóa
        encoding_e1 = self.tokenizer_bert(marked_text_e1, max_length=self.MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
        encoding_e2 = self.tokenizer_bert(marked_text_e2, max_length=self.MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
        
        input_ids_e1 = encoding_e1['input_ids'].squeeze(0)
        attention_mask_e1 = encoding_e1['attention_mask'].squeeze(0)

        input_ids_e2 = encoding_e2['input_ids'].squeeze(0)
        attention_mask_e2 = encoding_e2['attention_mask'].squeeze(0)

        input_ids = [input_ids_e1, input_ids_e2]
        attention_mask = [attention_mask_e1, attention_mask_e2]
        label = torch.tensor(row['label_encoded'], dtype=torch.long)
        tag_id = [e1_id, e2_id, doc_id]
        
        item = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label,
                'tag_id': tag_id
                }

        return item

    def get_spans_of_sentences(self, text):
        spans = []
        start = 0
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        spans = []
        start = 0
        for i, sentence in enumerate(sentences):
            match = re.search(re.escape(sentence), text)
            if match:
                spans.append(match.span())
            else:
                print(f"Not found sentence {sentence}")
        return spans
    
    def find_sentence_containing_span(self, text, spans, target_start, target_end):
        for i, (start, end) in enumerate(spans):
            if start <= target_start < end or start < target_end <= end:
                return text[start:end], (start, end)
        return None, None
    
    def get_token_for_char(self, tokens, char_idx):
        for i, token in enumerate(tokens):
            if char_idx > token.idx:
                continue
            if char_idx == token.idx:
                return i, token
            if char_idx < token.idx:
                return i - 1, tokens[i - 1]
        return len(tokens) - 1, tokens[len(tokens) - 1]
        
    def get_context_by_window(self, text, tokens, e_start, e_end):
        start_token, _ = self.get_token_for_char(tokens, e_start)
        end_token, _ = self.get_token_for_char(tokens, e_end)
        start_token -= (self.window_size - (end_token - start_token)) // 2
        end_token += (self.window_size - (end_token - start_token)) // 2
        end_token += max(0, -start_token)
        start_token = max(0, start_token)
        end_token = min(end_token, len(tokens) - 1)
        start = tokens[start_token].idx
        end = tokens[end_token].idx + len(tokens[end_token])
        return text[start:end], (start, end)
    
    def create_marked_text(self, text, e1_start, e1_end, e2_start, e2_end):
        tokens = self.tokenizer_spacy(text)
        spans = self.get_spans_of_sentences(text)
        context_e1, span_context_e1 = self.find_sentence_containing_span(text, spans, e1_start, e1_end)
        context_e2, span_context_e2 = self.find_sentence_containing_span(text, spans, e2_start, e2_end)
    
        if context_e1 is None or span_context_e1 is None or len(self.tokenizer_spacy(context_e1)) > self.window_size:
            context_e1, span_context_e1 = self.get_context_by_window(text, tokens, e1_start, e1_end)
    
        if context_e2 is None or span_context_e2 is None or len(self.tokenizer_spacy(context_e2)) > self.window_size:
            context_e2, span_context_e2 = self.get_context_by_window(text, tokens, e2_start, e2_end)
        
        e1_start_in_context = e1_start - span_context_e1[0]
        e1_end_in_context = e1_end - span_context_e1[0]
        e2_start_in_context = e2_start - span_context_e2[0]
        e2_end_in_context = e2_end - span_context_e2[0]
    
        marked_text_e1 = (context_e1[:e1_start_in_context] 
                          + "[E1] " 
                          + context_e1[e1_start_in_context:e1_end_in_context] 
                          + " [/E1]" 
                          + context_e1[e1_end_in_context:])
    
        marked_text_e2 = (context_e2[:e2_start_in_context] 
                          + "[E2] " 
                          + context_e2[e2_start_in_context:e2_end_in_context] 
                          + " [/E2]" 
                          + context_e2[e2_end_in_context:])
    
        return marked_text_e1, marked_text_e2

    def create_cache_dataset(self, save_path):
        prefix = '/'.join(save_path.split('/')[:-1])
        if os.path.exists(prefix) is False:
            raise ValueError(prefix, "is not existed.")
        
        features = []
        labels =  []
        tag_ids = []
        dataset_length = len(self.dataframe)
        print("Number of item in dataset:", dataset_length)
        print("Make embeddings")
        for i in tqdm(range(dataset_length), desc="Processing tasks:"):
            item = self.make_ids_for_bert(i)
            y = item['label'].numpy()
            tag_id = item['tag_id']
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            features.append([input_ids, attention_mask])
            labels.append(y)
            tag_ids.append(tag_id)
            
        print("Saving dataset at:", save_path)   
        save_to_pickle(features, labels, tag_ids, file_name=save_path) 

def load_raw_data(paths):
    if len(paths) == 1:
        data_path = paths[0]
        # Load dataframe (replace with actual data)
        train_df = pd.read_csv(data_path)
       
        print("Load original data:")
        print("Data:", train_df.shape)
       
        # Mã hóa nhãn
        label_encoder = LabelEncoder()
        train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
        num_classes = len(label_encoder.classes_)
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(f"Label Mapping: {label_mapping}")
    
        print("Split data:")
        test_size = 0.15
        validation_size = 0.15
        random_state = 42
        
        print("Data to train set and test set with test_size:", test_size)
        train_df, test_df = train_test_split(train_df, test_size=validation_size, random_state=random_state)
        print("Train:", len(train_df))
        print("Test", len(test_df))
    
        
        print("Train set split to train set and validation set with validation_size:", validation_size)
        train_df, valid_df = train_test_split(train_df, test_size=validation_size, random_state=random_state)
        print("Train:", len(train_df))
        print("Validation", len(valid_df))

        return train_df, valid_df, test_df
    
    if len(paths) == 2:
        data_train_path, data_test_path = paths[0], paths[1]
        # Load dataframe (replace with actual data)
        train_df = pd.read_csv(data_train_path)
        test_df = pd.read_csv(data_test_path)
        print("Load original data:")
        print("train_df:", train_df.shape)
        print("test_df:", test_df.shape)
        
        # Mã hóa nhãn
        label_encoder = LabelEncoder()
        train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
        test_df['label_encoded'] = label_encoder.transform(test_df['label'])  # Ánh xạ giống tập huấn luyện
        num_classes = len(label_encoder.classes_)
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(f"Label Mapping: {label_mapping}")
    
        print("Split data:")
        validation_size = 0.15
        random_state = 42
        
        print("Train to train set and validation set with validation_size:", validation_size)
        train_df, valid_df = train_test_split(train_df, test_size=validation_size, random_state=random_state)
        print("Train:", len(train_df))
        print("Validation", len(valid_df))
    
        return train_df, valid_df, test_df
     
    if len(paths) == 3:
        data_train_path, data_dev_path, data_test_path = paths[0], paths[1], paths[2]
         # Load dataframe (replace with actual data)
        train_df = pd.read_csv(data_train_path)
        valid_df = pd.read_csv(data_dev_path)
        test_df = pd.read_csv(data_test_path)
        print("Load original data:")
        print("train_df:", train_df.shape)
        print("test_df:", test_df.shape)
        
        # Mã hóa nhãn
        label_encoder = LabelEncoder()
        train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
        test_df['label_encoded'] = label_encoder.transform(test_df['label'])  # Ánh xạ giống tập huấn luyện
        valid_df['label_encoded'] = label_encoder.transform(valid_df['label'])  # Ánh xạ giống tập huấn luyện
        num_classes = len(label_encoder.classes_)
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(f"Label Mapping: {label_mapping}")
    
        print("Train:", len(train_df))
        print("Validation", len(valid_df))
        print("Test:", len(test_df))
    
        return train_df, valid_df, test_df

def script_i2b2(root_data, mode):
    data_source = root_data + "raw_data/i2b2_2012/data_processed/"
    data_train_path = data_source + "train_merged_full.csv"
    data_test_path = data_source + "test_merged_full.csv"

    train_df, valid_df, test_df =  load_raw_data([data_train_path, data_test_path])

    window_size, MAX_LENGTH, shuffle = 128, 256, False
    # Tokenizer và DataLoader
    nlp_spacy = English()
    nlp_spacy.add_pipe("senter")
    nlp_spacy.initialize()
    tokenizer_spacy = Tokenizer(nlp_spacy.vocab)
    
    bert_model_name = "bert-base-uncased"
    tokenizer_bert = BertTokenizerFast.from_pretrained(bert_model_name)
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer_bert.add_special_tokens(special_tokens)
    
    if mode == 0:
        train_save_path = root_data + 'dataset_cache/i2b2/merged_context/train_merged_BERT.pkl'
        validation_save_path = root_data + 'dataset_cache/i2b2/merged_context/validalidation_merged_BERT.pkl'
        test_save_path = root_data + 'dataset_cache/i2b2/merged_context/test_merged_BERT.pkl'
        create_cache_dataset_with_marked_context_merged(train_df, train_save_path, window_size, MAX_LENGTH)
        create_cache_dataset_with_marked_context_merged(valid_df, validation_save_path, window_size, MAX_LENGTH)
        create_cache_dataset_with_marked_context_merged(test_df, test_save_path, window_size, MAX_LENGTH)  
    else:
        train_save_path = root_data + 'dataset_cache/i2b2/separate_context/train_merged_BERT.pkl'
        validation_save_path = root_data + 'dataset_cache/i2b2/separate_context/validation_merged_BERT.pkl'
        test_save_path = root_data + 'dataset_cache/i2b2/separate_context/test_merged_BERT.pkl'
        
        dataset = TemporalRelationDataset(train_df, nlp_spacy, tokenizer_spacy, tokenizer_bert, MAX_LENGTH, window_size)
        dataset.create_cache_dataset(train_save_path)

        dataset = TemporalRelationDataset(valid_df, nlp_spacy, tokenizer_spacy, tokenizer_bert, MAX_LENGTH, window_size)
        dataset.create_cache_dataset(validation_save_path)

        dataset = TemporalRelationDataset(test_df, nlp_spacy, tokenizer_spacy, tokenizer_bert, MAX_LENGTH, window_size)
        dataset.create_cache_dataset(test_save_path)
    print("Finished!")

def script_TBD(root_data, mode):
    data_source = root_data + "raw_data/TimeBank-dense/"
    train_data_path = data_source + "train.csv"
    dev_data_path = data_source + "dev.csv"
    test_data_path = data_source + "test.csv"

    train_df, valid_df, test_df =  load_raw_data([train_data_path, dev_data_path, test_data_path])

    window_size, MAX_LENGTH, shuffle = 128, 256, False
    # Tokenizer và DataLoader
    nlp_spacy = English()
    nlp_spacy.add_pipe("senter")
    nlp_spacy.initialize()
    tokenizer_spacy = Tokenizer(nlp_spacy.vocab)
    
    bert_model_name = "bert-base-uncased"
    tokenizer_bert = BertTokenizerFast.from_pretrained(bert_model_name)
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer_bert.add_special_tokens(special_tokens)
    
    if mode == 0:
        train_save_path = root_data + 'dataset_cache/TBD/merged_context/train_BERT.pkl'
        validation_save_path = root_data + 'dataset_cache/TBD/merged_context/validation_BERT.pkl'
        test_save_path = root_data + 'dataset_cache/TBD/merged_context/test_BERT.pkl'
        create_cache_dataset_with_marked_context_merged(train_df, train_save_path, window_size, MAX_LENGTH)
        create_cache_dataset_with_marked_context_merged(valid_df, validation_save_path, window_size, MAX_LENGTH)
        create_cache_dataset_with_marked_context_merged(test_df, test_save_path, window_size, MAX_LENGTH)  
    else:
        train_save_path = root_data + 'dataset_cache/TBD/separate_context/train_BERT.pkl'
        validation_save_path = root_data + 'dataset_cache/TBD/separate_context/validation_BERT.pkl'
        test_save_path = root_data + 'dataset_cache/TBD/separate_context/test_BERT.pkl'
        
        dataset = TemporalRelationDataset(train_df, nlp_spacy, tokenizer_spacy, tokenizer_bert, MAX_LENGTH, window_size)
        dataset.create_cache_dataset(train_save_path)

        dataset = TemporalRelationDataset(valid_df, nlp_spacy, tokenizer_spacy, tokenizer_bert, MAX_LENGTH, window_size)
        dataset.create_cache_dataset(validation_save_path)

        dataset = TemporalRelationDataset(test_df, nlp_spacy, tokenizer_spacy, tokenizer_bert, MAX_LENGTH, window_size)
        dataset.create_cache_dataset(test_save_path)

    print("Finished!")

def script_TB12(root_data, mode):
    data_source = root_data + "raw_data/TimeBank1.2/data/"
    data_path = data_source + "timeml.csv"

    train_df, valid_df, test_df =  load_raw_data([data_path])

    window_size, MAX_LENGTH, shuffle = 128, 256, False
   

    if mode == 0:
        train_save_path = root + 'dataset_cache/TB1.2/merged_context/train_BERT.pkl'
        validation_save_path = root + 'dataset_cache/TB1.2/merged_context/validation_BERT.pkl'
        test_save_path = root + 'dataset_cache/TB1.2/merged_context/test_BERT.pkl'
        create_cache_dataset_with_marked_context_merged(train_df, train_save_path, window_size, MAX_LENGTH)
        create_cache_dataset_with_marked_context_merged(valid_df, validation_save_path, window_size, MAX_LENGTH)
        create_cache_dataset_with_marked_context_merged(test_df, test_save_path, window_size, MAX_LENGTH)  
    else:
        train_save_path = root + 'dataset_cache/TB1.2/separate_context/train_BERT.pkl'
        validation_save_path = root + 'dataset_cache/TB1.2/separate_context/validation_BERT.pkl'
        test_save_path = root + 'dataset_cache/TB1.2/separate_context/test_BERT.pkl'
        
        dataset = TemporalRelationDataset(train_df, nlp_spacy, tokenizer_spacy, tokenizer_bert, MAX_LENGTH, window_size)
        dataset.create_cache_dataset(train_save_path)

        dataset = TemporalRelationDataset(valid_df, nlp_spacy, tokenizer_spacy, tokenizer_bert, MAX_LENGTH, window_size)
        dataset.create_cache_dataset(validation_save_path)

        dataset = TemporalRelationDataset(test_df, nlp_spacy, tokenizer_spacy, tokenizer_bert, MAX_LENGTH, window_size)
        dataset.create_cache_dataset(test_save_path)
    
    print("Finished!")

if __name__ == '__main__':

    machine = int(input("local: 0, server: 1 --- Choose (0 or 1):"))
    mode = int(input("merged_marked_context: 0, separate_marked_context: 1 --- Choose (0 or 1):"))
    script = int(input("i2b2: 0, TBD: 1, TB1.2: 2 --- Choose (0 or 1 or 2):"))
    
    if machine == 0:
        root_data = "/Users/doduydao/daodd/PycharmProjects/TRE/data/"
    else:
        root_data = "/data/ddao/TRE/data/"

    print("root_data:", root_data)
    
    if script == 0:
        script_i2b2(root_data, mode)
    
    elif script == 1:
        script_TBD(root_data, mode)

    elif script == 2:
        script_TB12(root_data, mode)

    else:
        print("Incorrect augments")