from os import listdir
import pandas as pd
from TBPreprocess.TimeMLParser import parse, make_data_structure, clean_text_and_update_spans


def are_intervals_disjoint(interval1, interval2):
    a, b = interval1
    c, d = interval2
    return b < c or d < a

def get_only_event_id_dict(updated_entity_dict):
    event_dict = dict()
    for k, v in updated_entity_dict.items():
        if 'e' in k:
            event_dict[v['id']] = v
    return event_dict

def load_data(data_folder, map_data):
    rows = []
    no_file = 0
    file_error_non_type_relations = set()
    count_relations = 0
    count_error_relations = 0
    count_none_type_relations = 0

    for file in listdir(data_folder):
        if not file.endswith('.tml'):
            continue
        no_file += 1


        with open(data_folder + '/' + file, 'r') as open_file:
            file_string = open_file.read().replace(' & ', '   ').replace('L&D', 'LnD').replace('&', 'n')
        file_id = ".".join(file.split(".")[:-1])

        # if file_id != "PRI19980121.2000.2591":
        #     continue

        doc_data = map_data[map_data['doc_id'] == file_id]
        # print(file_id, ":", doc_data.shape[0])
        # print(doc_data)
        metadata, raw_text, links, instances, signals, events, timexs = parse(file_string, only_tlinks=True)
        tlinks, entity_dict, signal_dict = make_data_structure(links, signals, events, timexs)
        cleaned_text, updated_entity_dict, updated_signal_dict = clean_text_and_update_spans(raw_text, entity_dict, signal_dict)
        tlinks = doc_data.values
        count_relations += len(tlinks)
        updated_entity_dict = get_only_event_id_dict(updated_entity_dict)

        for link in tlinks:
            label = link[3]
            if link[1] not in updated_entity_dict or link[2] not in updated_entity_dict:
                count_error_relations += 1
                continue
            start_entity = updated_entity_dict[link[1]]
            end_entity = updated_entity_dict[link[2]]
            if start_entity is None or end_entity is None:
                file_error_non_type_relations.add(file)
                count_error_relations += 1
                continue

            entity1_id = start_entity['id']
            entity1_start = start_entity['span'][0]
            entity1_end = start_entity['span'][1]
            entity1_text = start_entity['text']

            entity2_id = end_entity['id']
            entity2_start = end_entity['span'][0]
            entity2_end = end_entity['span'][1]
            entity2_text = end_entity['text']

            if entity1_id == entity2_id:
                count_error_relations += 1
                continue

            if entity1_start == entity2_start or entity1_end == entity2_end:
                count_error_relations += 1
                continue

            if not are_intervals_disjoint((entity1_start, entity1_end), (entity2_start, entity2_end)):
                count_error_relations += 1
                continue


            rows.append([entity1_id, entity2_id,
                         entity1_start, entity2_start,
                         entity1_end, entity2_end,
                         entity1_text, entity2_text,
                         file_id, cleaned_text, label])

    print("files:", no_file)
    # print("Total relations (ignore errors):", count_relations)
    print(f"Processed relations: {len(rows)}")

    df = pd.DataFrame(rows, columns=["entity1_id", "entity2_id",
                                     "entity1_start", "entity2_start",
                                     "entity1_end", "entity2_end",
                                     "entity1_text", "entity2_text",
                                     "document_id", "text", "label"])
    return df



if __name__ == '__main__':

    # data_root = "/Users/doduydao/daodd/PycharmProjects/TRE/data/raw_data/TDDiscourse/"
    # train_map_path = data_root + "TDDMan/TDDManTrain.tsv"
    # dev_map_path = data_root + "TDDMan/TDDManDev.tsv"
    # test_map_path = data_root + "TDDMan/TDDManTest.tsv"
    #
    # TRAINNING_DATA_DIR = "/Users/doduydao/daodd/PycharmProjects/TRE/data/raw_data/TBD/original_TB12/train"
    # TEST_DATA_DIR = "/Users/doduydao/daodd/PycharmProjects/TRE/data/raw_data/TBD/original_TB12/test"
    # DEV_DATA_DIR = "/Users/doduydao/daodd/PycharmProjects/TRE/data/raw_data/TBD/original_TB12/dev"
    #
    # train_map_data = pd.read_csv(train_map_path, delimiter="\t", header=None, names=['doc_id', 'e1_id', 'e2_id', 'label'])
    # dev_map_data = pd.read_csv(dev_map_path, delimiter="\t", header=None, names=['doc_id', 'e1_id', 'e2_id', 'label'])
    # test_map_data = pd.read_csv(test_map_path, delimiter="\t", header=None, names=['doc_id', 'e1_id', 'e2_id', 'label'])
    # print(train_map_data)
    # print(dev_map_data)
    # print(test_map_data)
    #
    #
    # train_df = load_data(TRAINNING_DATA_DIR, train_map_data)
    # train_df.to_csv(data_root + 'TDDManProcessed/train.csv', index=False)
    # print()
    #
    #
    # dev_df = load_data(DEV_DATA_DIR, dev_map_data)
    # dev_df.to_csv(data_root + 'TDDManProcessed/dev.csv', index=False)
    # print()
    #
    # test_df = load_data(TEST_DATA_DIR, test_map_data)
    # test_df.to_csv(data_root + 'TDDManProcessed/test.csv', index=False)
    # print()


    data_root = "/Users/doduydao/daodd/PycharmProjects/TRE/data/raw_data/TDDiscourse/"
    train_map_path = data_root + "TDDAuto/TDDAutoTrain.tsv"
    dev_map_path = data_root + "TDDAuto/TDDAutoDev.tsv"
    test_map_path = data_root + "TDDAuto/TDDAutoTest.tsv"

    TRAINNING_DATA_DIR = "/Users/doduydao/daodd/PycharmProjects/TRE/data/raw_data/TBD/original_TB12/train"
    TEST_DATA_DIR = "/Users/doduydao/daodd/PycharmProjects/TRE/data/raw_data/TBD/original_TB12/test"
    DEV_DATA_DIR = "/Users/doduydao/daodd/PycharmProjects/TRE/data/raw_data/TBD/original_TB12/dev"

    train_map_data = pd.read_csv(train_map_path, delimiter="\t", header=None, names=['doc_id', 'e1_id', 'e2_id', 'label'])
    dev_map_data = pd.read_csv(dev_map_path, delimiter="\t", header=None, names=['doc_id', 'e1_id', 'e2_id', 'label'])
    test_map_data = pd.read_csv(test_map_path, delimiter="\t", header=None, names=['doc_id', 'e1_id', 'e2_id', 'label'])
    print(train_map_data)
    print(dev_map_data)
    print(test_map_data)


    train_df = load_data(TRAINNING_DATA_DIR, train_map_data)
    train_df.to_csv(data_root + 'TDDAutoProcessed/train.csv', index=False)
    print()


    dev_df = load_data(DEV_DATA_DIR, dev_map_data)
    dev_df.to_csv(data_root + 'TDDAutoProcessed/dev.csv', index=False)
    print()

    test_df = load_data(TEST_DATA_DIR, test_map_data)
    test_df.to_csv(data_root + 'TDDAutoProcessed/test.csv', index=False)
    print()
