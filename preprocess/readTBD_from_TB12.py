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

def find_tlink_TB12(e1, e2, tlinks_TB12, entities_TB12):
    tlink = None

    for lid, link in tlinks_TB12.items():
        # print(lid, link)
        start_node = link['start_node']
        end_node = link['end_node']
        e_start = entities_TB12[start_node]
        e_end = entities_TB12[end_node]
        if e1['id'] == e_start['id'] and e2['id'] == e_end ['id']:
            tlink = link
            break
    return tlink


def load_data(data_folder, map_data, have_signal=False):

    rows = []
    no_file = 0
    file_error_non_type_relations = set()
    count_relations = 0
    count_error_relations = 0
    count_none_type_relations = 0

    count_signal = 0

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
        print(file_id, ":", doc_data.shape[0])
        # print(doc_data)
        metadata, raw_text, links, instances, signals, events, timexs = parse(file_string, only_tlinks=True)

        tlinks_TB12, entity_dict, signal_dict = make_data_structure(links, signals, events, timexs)


        cleaned_text, updated_entity_dict_TB12, updated_signal_dict_TB12 = clean_text_and_update_spans(raw_text, entity_dict, signal_dict)


        tlinks = doc_data.values
        count_relations += len(tlinks)
        updated_entity_dict = get_only_event_id_dict(updated_entity_dict_TB12)

        for link in tlinks:
            label = link[3]
            start_entity = updated_entity_dict[link[1]]
            end_entity = updated_entity_dict[link[2]]


            if start_entity is None or end_entity is None:
                file_error_non_type_relations.add(file)
                count_error_relations += 1
                continue

            if label == 'VAGUE':
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

            if have_signal:
                signal_id = ""
                signal_text = ""
                signal_start = ""
                signal_end = ""
                tlink = find_tlink_TB12(start_entity, end_entity, tlinks_TB12, updated_entity_dict_TB12)
                if tlink is not None:
                    signal_id = tlink['signal']
                    if signal_id != "":
                        print(updated_signal_dict_TB12[signal_id])
                        count_signal += 1
                        signal = updated_signal_dict_TB12[signal_id]
                        signal_id = signal['id']
                        signal_text = signal['text']
                        signal_start = signal['span'][0]
                        signal_end = signal['span'][1]


                rows.append([entity1_id, entity2_id,
                             entity1_start, entity2_start,
                             entity1_end, entity2_end,
                             entity1_text, entity2_text,
                             file_id, cleaned_text, label, signal_id, signal_text, signal_start, signal_end])
            else:
                rows.append([entity1_id, entity2_id,
                             entity1_start, entity2_start,
                             entity1_end, entity2_end,
                             entity1_text, entity2_text,
                             file_id, cleaned_text, label])
    print("files:", no_file)
    # print("Total relations (ignore errors):", count_relations)
    print(f"Processed relations: {len(rows)}")
    print(f"Signals: {count_signal}")
    if have_signal:
        df = pd.DataFrame(rows, columns=["entity1_id", "entity2_id",
                                         "entity1_start", "entity2_start",
                                         "entity1_end", "entity2_end",
                                         "entity1_text", "entity2_text",
                                         "document_id", "text", "label", "signal_id", "signal_text", "signal_start",
                                         "signal_end"])
    else:
        df = pd.DataFrame(rows, columns=["entity1_id", "entity2_id",
                                         "entity1_start", "entity2_start",
                                         "entity1_end", "entity2_end",
                                         "entity1_text", "entity2_text",
                                         "document_id", "text", "label"])
    return df



if __name__ == '__main__':

    # data_root = "/data/ddao/TRE/data/raw_data/TBD/"
    data_root = "/Users/doduydao/daodd/PycharmProjects/TRE/data/raw_data/TBD/"

    train_map_path = data_root + "original_TB12/train.txt"
    dev_map_path = data_root + "original_TB12/dev.txt"
    test_map_path = data_root + "original_TB12/test.txt"

    TRAINNING_DATA_DIR = data_root + "original_TB12/train"
    TEST_DATA_DIR = data_root + "original_TB12/test"
    DEV_DATA_DIR = data_root + "original_TB12/dev"

    train_map_data = pd.read_csv(train_map_path, delimiter="\t")
    dev_map_data = pd.read_csv(dev_map_path, delimiter="\t")
    test_map_data = pd.read_csv(test_map_path, delimiter="\t")

    # train_df = load_data(TRAINNING_DATA_DIR, train_map_data)
    # train_df.to_csv(data_root + 'processed/train_no_vague.csv', index=False)
    print()


    # dev_df = load_data(DEV_DATA_DIR, dev_map_data)
    # dev_df.to_csv(data_root + 'processed/dev_no_vague.csv', index=False)
    # print()

    test_df = load_data(TEST_DATA_DIR, test_map_data, have_signal=True)
    # test_df.to_csv(data_root + 'processed/test_have_signal.csv', index=False)
    # print()

    test_sample = test_df[test_df['document_id'] == 'CNN19980213.2130.0155']

    entity_e80_all = test_sample[
        (test_sample['entity1_id'] == 'e80') & (test_sample['entity2_id'] == 'e82')
        ]
    print(entity_e80_all)


    # print(test_sample.iloc[298])
    # print(test_sample)






