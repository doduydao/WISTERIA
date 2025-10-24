from os import listdir
import xml.etree.ElementTree as ET
import pandas as pd


def are_intervals_disjoint(interval1, interval2):
    a, b = interval1
    c, d = interval2
    return b <= c or d <= a


def load_data(data_folder):
    input_examples = []
    rows = []
    no_file = 0

    file_error_non_type_relations = set()
    count_relations = 0
    count_none_relations = 0
    count_sectime_relations = 0
    count_E_E_relations = 0
    count_E_T_relations = 0
    count_T_T_relations = 0
    for file in listdir(data_folder):
        if not file.endswith('.xml'):
            continue
        # print("Filename:", file)
        no_file+=1
        with open(data_folder + '/' + file, 'r') as open_file:
            file_string = open_file.read().replace(' & ', '   ').replace('L&D', 'LnD').replace('&', 'n')
        root = ET.fromstring(file_string)
        file_id = file.split(".")[0]
        text = root[0].text
        annotations = root[1]
        entities = {}
        links = []
        for a in annotations:
            if a.tag == 'EVENT':
                entities[a.attrib['id']] = a.attrib
            if a.tag == 'TIMEX3':
                entities[a.attrib['id']] = a.attrib
            if a.tag == 'TLINK':
                links.append(a.attrib)
        count_relations += len(links)

        for link in links:
            link_id = link['id'].upper()
            # if "SECTIME" in link_id:
            #     count_sectime_relations+=1
            #     continue
            label = link["type"].upper()
            entity1_text = link['fromText']
            entity2_text = link['toText']
            entity1_id = link['fromID']
            entity2_id = link['toID']


            if link['fromID'] not in entities \
                or link['toID'] not in entities \
                or link['type'] == '' \
                or link['toID'] == link['fromID']:
                # count_none_relations.add(file)
                count_none_relations += 1
                continue
            entity1_start = int(entities[link['fromID']]['start'])
            entity1_end = int(entities[link['fromID']]['end'])
            entity2_start = int(entities[link['toID']]['start'])
            entity2_end = int(entities[link['toID']]['end'])

            if entity1_start == entity2_start or entity1_end == entity2_end:
                count_none_relations += 1
                continue

            if not are_intervals_disjoint((entity1_start, entity1_end), (entity2_start, entity2_end)):
                count_none_relations += 1
                continue

            if 'E' in entity1_id.upper() and 'E' in entity2_id.upper():
                count_E_E_relations += 1
            elif ('E' in entity1_id.upper() and 'T' in entity2_id.upper()) or ('T' in entity1_id.upper() and 'E' in entity2_id.upper()):
                count_E_T_relations += 1
            else:
                count_T_T_relations += 1

            replace_dict = {'SIMULTANEOUS': 'OVERLAP', 'DURING': 'OVERLAP', 'BEFORE_OVERLAP': 'BEFORE',
                            'ENDED_BY': 'BEFORE', 'BEGUN_BY': 'AFTER'}
            if label in replace_dict:
                label = replace_dict[label]


            rows.append([entity1_id, entity2_id,
                         entity1_start, entity2_start,
                         entity1_end, entity2_end,
                         entity1_text, entity2_text,
                         file_id, text, label])

    print("files:", no_file)
    print("Total relations:", count_relations)
    print("Total sectime relations:",count_sectime_relations)
    print(f"None relations: {count_none_relations}")
    print(f"Processed relations: {len(rows)}")
    print(f"count_E_E_relations: {count_E_E_relations}")
    print(f"count_E_T_relations: {count_E_T_relations}")
    print(f"count_T_T_relations: {count_T_T_relations}")

    df = pd.DataFrame(rows, columns=["entity1_id", "entity2_id",
                                     "entity1_start", "entity2_start",
                                     "entity1_end", "entity2_end",
                                     "entity1_text", "entity2_text",
                                     "document_id", "text", "label"])
    return df

if __name__ == '__main__':
    TRAINNING_DATA_DIR = "/Users/doduydao/daodd/PycharmProjects/TRE/data/i2b2_2012/2012-07-15.original-annotation.release"
    TEST_DATA_DIR = "/Users/doduydao/daodd/PycharmProjects/TRE/data/i2b2_2012/ground_truth/merged_xml"

    train_df = load_data(TRAINNING_DATA_DIR)
    test_df = load_data(TEST_DATA_DIR)

    data_dir = '/Users/doduydao/daodd/PycharmProjects/TRE/data/i2b2_2012/data_processed/'
    train_df.to_csv(data_dir+'train_merged_full.csv', index=False)

    test_df.to_csv(data_dir+'test_merged_full.csv', index=False)

    print("Training")
    label_counts = train_df["label"].value_counts()
    # Tạo phân phối (tỷ lệ phần trăm)
    label_distribution = label_counts / label_counts.sum()

    # Hiển thị thống kê và phân phối
    print("Number of label:", label_counts.sum())
    print(label_counts)
    print("\nLabel distribution:")
    print(label_distribution)

    print("Test")
    label_counts = test_df["label"].value_counts()
    # Tạo phân phối (tỷ lệ phần trăm)
    label_distribution = label_counts / label_counts.sum()

    # Hiển thị thống kê và phân phối
    print("Number of label:", label_counts.sum())
    print(label_counts)
    print("\nLabel distribution:")
    print(label_distribution)


    # TRAINNING_DATA_DIR = "/Users/doduydao/daodd/PycharmProjects/TRE/data/small_test/cut"
    # train_df = load_data(TRAINNING_DATA_DIR)
    # data_dir = '/Users/doduydao/daodd/PycharmProjects/TRE/data/small_test/data_processed/'
    # train_df.to_csv(data_dir+'1.csv', index=False)