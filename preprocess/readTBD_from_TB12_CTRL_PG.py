from os import listdir
import pandas as pd
from TBPreprocess.TimeMLParser import parse, make_data_structure, clean_text_and_update_spans
import json

def are_intervals_disjoint(interval1, interval2):
    a, b = interval1
    c, d = interval2
    return b < c or d < a


def load_data(data_folder, map_data):
    rows = []
    no_file = 0
    file_error_non_type_relations = set()
    count_relations = 0
    count_error_relations = 0
    count_none_type_relations = 0
    no_relations = 0
    for file in listdir(data_folder):
        if not file.endswith('.tml'):
            continue
        no_file += 1

        # if file != "APW19980227.0487.tml":
        #     continue

        with open(data_folder + '/' + file, 'r') as open_file:
            file_string = open_file.read().replace(' & ', '   ').replace('L&D', 'LnD').replace('&', 'n')
        file_id = ".".join(file.split(".")[:-1])

        doc_data = map_data[map_data['doc_id']==file_id]
        print(file_id, ":", doc_data.shape[0])
        metadata, raw_text, links, instances, signals, events, timexs = parse(file_string, only_tlinks=True)
        tlinks, entity_dict, signal_dict = make_data_structure(links, signals, events, timexs)
        cleaned_text, updated_entity_dict, updated_signal_dict = clean_text_and_update_spans(raw_text, entity_dict, signal_dict)

        count_relations += len(tlinks)
        no_relations += doc_data.shape[0]

        # for link_id, link in tlinks.items():
        #     label = link['label']
        #     start_entity = updated_entity_dict[link['start_node']]
        #     end_entity = updated_entity_dict[link['end_node']]
        #
        #     if start_entity is None or end_entity is None:
        #         file_error_non_type_relations.add(file)
        #         count_error_relations += 1
        #         continue
        #     if label == 'NONE':
        #         count_none_type_relations += 1
        #         label = "VAGUE"
        #
        #     entity1_id = start_entity['id']
        #     entity1_start = start_entity['span'][0]
        #     entity1_end = start_entity['span'][1]
        #     entity1_text = start_entity['text']
        #
        #     entity2_id = end_entity['id']
        #     entity2_start = end_entity['span'][0]
        #     entity2_end = end_entity['span'][1]
        #     entity2_text = end_entity['text']
        #
        #     if entity1_id == entity2_id:
        #         count_error_relations += 1
        #         continue
        #
        #     if entity1_start == entity2_start or entity1_end == entity2_end:
        #         count_error_relations += 1
        #         continue
        #
        #     if not are_intervals_disjoint((entity1_start, entity1_end), (entity2_start, entity2_end)):
        #         count_error_relations += 1
        #         continue
        #
        #
        #     rows.append([entity1_id, entity2_id,
        #                  entity1_start, entity2_start,
        #                  entity1_end, entity2_end,
        #                  entity1_text, entity2_text,
        #                  file_id, cleaned_text, label])


    print("files:", no_file)
    print("file_error_non_type_relations:", len(file_error_non_type_relations), ":", file_error_non_type_relations)
    print("Total relations (ignore errors):", count_relations)
    print(f"Error relations: {count_error_relations}")
    print(f"None type relations: {count_none_type_relations}")
    print(f"Processed relations: {len(rows)}")
    print("no_relations", no_relations)
    df = pd.DataFrame(rows, columns=["entity1_id", "entity2_id",
                                     "entity1_start", "entity2_start",
                                     "entity1_end", "entity2_end",
                                     "entity1_text", "entity2_text",
                                     "document_id", "text", "label"])
    return df

def get_relations(data):
    relations = []
    for k, v in data.items():
        for k1, v1 in v.items():
            for rel in v1['relation']:
                r = "\t".join([k, rel[2], rel[5], rel[6]])
                relations.append(r)
    print(len(relations))
    text = "doc_id\te1_id\te2_id\tlabel\n"+"\n".join(relations)
    return text



if __name__ == '__main__':

    # data_root = "/data/ddao/TRE/data/raw_data/TBD/"
    data_root = "/Users/doduydao/daodd/PycharmProjects/TRE/data/raw_data/TBD/"
    # map_path = data_root + "TBD.txt"
    #
    #
    # TRAINNING_DATA_DIR = data_root + "original_TB12/train"
    # TEST_DATA_DIR = data_root + "original_TB12/test"
    # DEV_DATA_DIR = data_root + "original_TB12/dev"

    # map_data = pd.read_csv(map_path, delimiter="\t")
    # print(map_data.shape)

    # APW19980227 = map_data[map_data['doc_id'] == "AP900815-0044"]
    # print(APW19980227.shape[0])

    data_path = "/Users/doduydao/daodd/PycharmProjects/TRE/references/CTRL-PG/data/tbd/all_context_small/"


    train_json_path = data_path + "train.json"
    test_json_path = data_path + "test.json"
    dev_json_path = data_path + "dev.json"
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_json = json.load(f)
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_json = json.load(f)
    with open(dev_json_path, 'r', encoding='utf-8') as f:
        dev_json = json.load(f)

    train_path_out = data_root + "original_TB12/train.txt"
    dev_path_out = data_root + "original_TB12/dev.txt"
    test_path_out = data_root + "original_TB12/test.txt"

    with open(train_path_out, 'w', encoding='utf-8') as f:
        f.write(get_relations(train_json))

    with open(dev_path_out, 'w', encoding='utf-8') as f:
        f.write(get_relations(dev_json))

    with open(test_path_out, 'w', encoding='utf-8') as f:
        f.write(get_relations(test_json))

