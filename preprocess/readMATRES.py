import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
from os import listdir


def are_intervals_disjoint(interval1, interval2):
    a, b = interval1
    c, d = interval2
    return b < c or d < a


def get_verb_span(row):
    # Đảm bảo mỗi phần cách nhau đúng 1 khoảng trắng
    before = str(row['before']).rstrip() + ' '
    verb = str(row['verb']).strip()
    after = ' ' + str(row['after']).lstrip()

    # Tính vị trí bắt đầu/kết thúc của verb trong chuỗi mới
    full_text = before + verb + after
    start = len(before)
    end = start + len(verb)
    return pd.Series([full_text, start, end], index=['text', 'verb_start', 'verb_end'])


def process_raw_data(data):
    df = data[['after', 'before', 'docid', 'eventid', 'verb']].copy()
    df[['text', 'verb_start', 'verb_end']] = df.apply(get_verb_span, axis=1)
    df = df[['docid', 'eventid', 'verb', 'verb_start', 'verb_end', 'text']].copy()

    result = {}

    for _, row in df.iterrows():
        docid = row['docid']
        eventid = row['eventid']

        # Tạo dict chứa thông tin chi tiết
        event_info = {
            'verb': row['verb'],
            'verb_start': row['verb_start'],
            'verb_end': row['verb_end'],
            'text': row['text']
        }

        # Khởi tạo nếu docid chưa có
        if docid not in result:
            result[docid] = {}

        # Gán event_info vào eventid
        result[docid][eventid] = event_info
    return result


def merge_data(df, data_map):
    data_list = df.to_dict(orient='records')
    rows = []
    count_error_relations = 0



    for item in data_list:
        doc_id = item['doc_id']
        e1_id = item['e1_eiid']
        e2_id = item['e2_eiid']
        e1_text = item['e1_text']
        e2_text = item['e2_text']
        label = item['label']

        e1_start = None
        e1_end = None
        e2_start = None
        e2_end = None
        e1_context = None
        e2_context = None
        # Bổ sung e1_start, e1_end
        if doc_id in data_map:
            doc = data_map[doc_id]

            for e_id, e_info in doc.items():
                if e1_text == e_info['verb']:
                    e1_start = e_info['verb_start']
                    e1_end = e_info['verb_end']
                    e1_context = e_info['text']
                if e2_text == e_info['verb']:
                    e2_start = e_info['verb_start']
                    e2_end = e_info['verb_end']
                    e2_context = e_info['text']
        else:
            print('No', doc_id, 'in datamap')
        if e1_start is None or e2_start is None or e1_end is None or e2_end is None:
            count_error_relations += 1
            # print(item)
            continue
        else:
            if e1_context == e2_context:
                text = e1_context
            else:
                text = "\n".join([e1_context, e2_context])
                e2_start = e2_start + len(e1_context) + 1
                e2_end = e2_end + len(e1_context) + 1
            rows.append([e1_id, e2_id,
                         e1_start, e2_start,
                         e1_end, e2_end,
                         e1_text, e2_text,
                         doc_id, text, label])

    print(f"Processed relations: {len(rows)}")
    print(f"Not found relations: {count_error_relations}")
    df = pd.DataFrame(rows, columns=["entity1_id", "entity2_id",
                                     "entity1_start", "entity2_start",
                                     "entity1_end", "entity2_end",
                                     "entity1_text", "entity2_text",
                                     "document_id", "text", "label"])

    print('Post-process:')
    print('Remove duplicate entity:')
    df = df[df["entity1_start"] != df["entity2_start"]].reset_index(drop=True)
    print('Finished!')
    print('Length of data:', len(df))
    return df


def split_raw_aquaint_platinum(df,  AQ_doc_ids, PL_doc_ids):
    raw_aquaint, raw_platinum = None, None
    # Lọc các dòng với docid thuộc AQ_doc_ids
    raw_aquaint = df[df['docid'].isin(AQ_doc_ids)].copy()

    # Lọc các dòng với docid thuộc PL_doc_ids
    raw_platinum = df[df['docid'].isin(PL_doc_ids)].copy()

    return raw_aquaint, raw_platinum

if __name__ == '__main__':
    # data_root = "/data/ddao/TRE/data/raw_data/TBD/"
    data_root = "/Users/doduydao/daodd/PycharmProjects/TRE/data/raw_data/MATRES/"

    aquaint = data_root + "aquaint.txt"
    platnium = data_root + "platinum.txt"
    timebank = data_root + "timebank.txt"

    aquaint_data = pd.read_csv(aquaint, delimiter="\t", header=None,
                               names=['doc_id', 'e1_text', 'e2_text', 'e1_eiid', 'e2_eiid', 'label'])
    timebank_data = pd.read_csv(timebank, delimiter="\t", header=None,
                                names=['doc_id', 'e1_text', 'e2_text', 'e1_eiid', 'e2_eiid', 'label'])


    # AQTB = pd.concat([aquaint_data, timebank_data], ignore_index=True)
    platnium_data = pd.read_csv(platnium, delimiter="\t", header=None, names=['doc_id', 'e1_text', 'e2_text', 'e1_eiid', 'e2_eiid', 'label'])

    print("timebank_data:", timebank_data.shape[0])
    print("aquaint_data:", aquaint_data.shape[0])
    print("Platinum:", platnium_data.shape[0])

    AQ_doc_ids = set(aquaint_data['doc_id'])
    PL_doc_ids = set(platnium_data['doc_id'])
    # print(AQ_doc_ids)

    raw_aquaint_platinum_path = data_root + "rawdata/AQ_Platinum_all.csv"
    raw_timebank_path = data_root + "rawdata/TBDense_all_new.csv"
    raw_timebank_remaining_path = data_root + "rawdata/TB_remaining_147docs.csv"
    raw_timebank_intersection_path = data_root + "rawdata/TBDense_intersection.csv"
    #
    raw_timebank = pd.read_csv(raw_timebank_path, delimiter=",")
    raw_timebank_remaining_data = pd.read_csv(raw_timebank_remaining_path, delimiter=",")
    # raw_timebank_intersection_data = pd.read_csv(raw_timebank_intersection_path, delimiter=",")

    # Gộp lại
    raw_timebank_merged = pd.concat([raw_timebank, raw_timebank_remaining_data])
    # Loại bỏ trùng lặp theo tất cả các cột
    raw_timebank_merged = raw_timebank_merged.drop_duplicates()
    # print(raw_timebank_merged.shape)

    raw_aquaint_platinum_data = pd.read_csv(raw_aquaint_platinum_path, delimiter=",")
    raw_aquaint, raw_platinum = split_raw_aquaint_platinum(raw_aquaint_platinum_data, AQ_doc_ids, PL_doc_ids)

    timebank_process = process_raw_data(raw_timebank_merged)
    aquaint_process = process_raw_data(raw_aquaint)
    raw_platinum = process_raw_data(raw_platinum)

    # for d, v in timebank_process.items():
    #     for v_id, v_info in v.items():
    #         print(v_id, v_info['verb'], v_info['verb_start'], v_info['verb_end'])
    #     break
    print()
    TB_df = merge_data(timebank_data, timebank_process)
    print()
    AQ_df = merge_data(aquaint_data, aquaint_process)
    print()
    PL_df = merge_data(platnium_data, raw_platinum)

    # print(TB_df)
    # print(AQ_df)
    # print(PL_df)
    train = pd.concat([TB_df, AQ_df])
    test = PL_df
    train.to_csv('/Users/doduydao/daodd/PycharmProjects/TRE/data/raw_data/MATRES/data_processed/'+ 'train.csv', index=False)
    test.to_csv('/Users/doduydao/daodd/PycharmProjects/TRE/data/raw_data/MATRES/data_processed/' + 'test.csv', index=False)