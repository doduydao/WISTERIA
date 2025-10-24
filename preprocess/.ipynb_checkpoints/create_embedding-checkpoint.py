from create_embedd_merged_context import script_merged_context
from create_embedd_separate_context import script_separate_context
from transformers import BertTokenizer, BertModel, BertTokenizerFast

if __name__ == '__main__':
    machine = int(input("local: 0, server: 1 --- Choose (0 or 1):"))
    data_name = str(input("<i2b2> <TBD> <TB1.2>:"))
    mode = str(input("<merged_context>, <separate_context>:")

    if machine == 0:
        root_code = "/Users/doduydao/daodd/PycharmProjects/phd-dao-do/TRE/"
        root_data = "/Users/doduydao/daodd/PycharmProjects/TRE/data/"
        
    else:
        root_code = "~/daodd/phd-dao-do/TRE/"
        root_data = "/data/ddao/TRE/data/"

    msg = "bert-base-uncased: 0\n bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12:1\n emilyalsentzer/Bio_ClinicalBERT:2\n medicalai/ClinicalBERT:3\n dmis-lab/biobert-v1.1:4"
    model = int(input(msg))
    match model:
        case 0:
            bert_model_name = "bert-base-uncased"
            folder_name = "/BERT/"
        case 1:
            bert_model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
            folder_name = "/BLUEBERT/"
        case 2:
            bert_model_name = "emilyalsentzer/Bio_ClinicalBERT"
            folder_name = "/BIO_CLINICALBERT/"
        case 3:
            bert_model_name = "medicalai/ClinicalBERT"
            folder_name = "/CLINICALBERT/"
        case 4:
            bert_model_name = "dmis-lab/biobert-v1.1"
            folder_name = "/BIOBERT/"
        case _:
            print("None model")
    
    train_path_in = 'dataset_cache/' + data_name + '/'+ mode + '/train_merged.pkl'
    validation_path_in = 'dataset_cache/' + data_name + '/'+ mode + '/validation_merged.pkl'
    test_path_in = 'dataset_cache/' + data_name + '/'+ mode + '/test_merged.pkl'

    train_path_out = 'embeddings/' + data_name + folder_name + mode + '/train_merged.pkl'
    validation_path_out = 'embeddings' + data_name + folder_name + mode + '/validation_merged.pkl'
    test_path_out = 'embeddings' + data_name + folder_name + mode + '/test_merged.pkl'

   
    # Device setup
    device = torch.device("mps")
    
    # Tokenizer và DataLoader
    
    bert = BertModel.from_pretrained(bert_model_name)
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    
    batch_size = 16

    if mode == 0:
        only_entity = int(input("words and entities: 0, only entities: 1 --- Choose (0 or 1):"))
        tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
        tokenizer.add_special_tokens(special_tokens)
        bert.resize_token_embeddings(len(tokenizer))
        
        
        script_merged_context(train_path_in, train_path_out, batch_size, only_entity==1, bert, tokenizer, device)
        script_merged_context(validation_path_in, validation_path_out, batch_size, only_entity==1, bert, tokenizer, device)
        script_merged_context(test_path_in, test_path_out, batch_size, only_entity==1, bert, tokenizer, device)
    
    if mode == 1:
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        tokenizer.add_special_tokens(special_tokens)
        bert.resize_token_embeddings(len(tokenizer))
        
        script_separate_context(train_path_in, train_path_out, batch_size, tokenizer, bert, device)
        script_separate_context(validation_path_in, validation_path_out, batch_size, tokenizer, bert, device)
        script_separate_context(test_path_in, test_path_out, batch_size, tokenizer, bert, device)