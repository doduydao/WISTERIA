from os import listdir
import pandas as pd
from TBPreprocess.TimeMLParser import parse
from TBPreprocess.Event import Event
from inputExample import InputExample

def remake_span(entities, text):
    stop_loop = False
    while stop_loop != True:
        status = False
        for i in range(len(text)):
            for entity_id in range(len(entities)):
                tag = entities[entity_id].raw_tag
                part_tmp_text = text[:i]
                if tag in part_tmp_text:
                    start_id = i - len(tag)
                    end_id = start_id + len(entities[entity_id].stem)
                    if isinstance(entities[entity_id], Event):
                        entities[entity_id].event_start = start_id
                        entities[entity_id].event_end = end_id
                    else:
                        entities[entity_id].timex_start = start_id
                        entities[entity_id].timex_end = end_id
                    text = text.replace(tag, entities[entity_id].stem)
                    status = True
                    break
            if status:
                break
            if i == len(text) - 1:
                stop_loop = True
                break
    return entities, text

def find_entities(entities, id):
    entity = None
    for entity in entities:
        if isinstance(entity, Event):
            if entity.eiid == id:
                return entity
        else:
           if entity.tID == id:
               return entity
    return entity


def load_data(data_folder):
    input_examples = []
    rows = []
    no_file = 0
    file_error_non_type_relations = set()
    count_relations = 0
    count_error_relations = 0
    for file in listdir(data_folder):
        if not file.endswith('.tml'):
            continue
        no_file += 1

        # if file != "APW19980227.0487.tml":
        #     continue
        print(file)
        with open(data_folder + '/' + file, 'r') as open_file:
            file_string = open_file.read().replace(' & ', '   ').replace('L&D', 'LnD').replace('&', 'n')
        file_id = file.split(".")[0]
        metadata, raw_text, links, instances, signals, events, timexs = parse(file_string)
        entities = events + timexs
        entities, text = remake_span(entities, raw_text)
        count_relations += len(links)
        for link in links:
            label = link.rel_type
            start_entity = find_entities(entities, link.start_node)
            end_entity = find_entities(entities, link.start_node)

            if start_entity is None or end_entity is None or label == 'NONE':
                file_error_non_type_relations.add(file)
                count_error_relations += 1
                continue

            if isinstance(start_entity, Event):
                entity1_id = start_entity.eid
                entity1_start = start_entity.event_start
                entity1_end = start_entity.event_end
                entity1_text = start_entity.stem
            else:
                entity1_id = start_entity.tID
                entity1_start = start_entity.timex_start
                entity1_end = start_entity.timex_end
                entity1_text = start_entity.stem

            if isinstance(end_entity, Event):
                entity2_id = end_entity.eid
                entity2_start = end_entity.event_start
                entity2_end = end_entity.event_end
                entity2_text = end_entity.stem
            else:
                entity2_id = end_entity.tID
                entity2_start = end_entity.timex_start
                entity2_end = end_entity.timex_end
                entity2_text = end_entity.stem

            # print(link)
            rows.append([entity1_id, entity2_id,
                         entity1_start, entity2_start,
                         entity1_end, entity2_end,
                         entity1_text, entity2_text,
                         file_id, text, label])

            # input_examples.append(InputExample(entity1_id, entity2_id,
            #                                    entity1_start, entity2_start,
            #                                    entity1_end, entity2_end,
            #                                    entity1_text, entity2_text,
            #                                    file_id, text, label))
    print("files:", no_file)
    print("file_error_non_type_relations:", len(file_error_non_type_relations), ":", file_error_non_type_relations)
    print("Total relations (ignore errors):", count_relations)
    print(f"Error relations: {count_error_relations}")
    print(f"Processed relations: {len(rows)}")
    df = pd.DataFrame(rows, columns=["entity1_id", "entity2_id",
                                     "entity1_start", "entity2_start",
                                     "entity1_end", "entity2_end",
                                     "entity1_text", "entity2_text",
                                     "document_id", "text", "label"])
    return df
if __name__ == '__main__':

    # extra_DATA_DIR = "/Users/doduydao/daodd/PycharmProjects/TRE/data/TimeBank1.2/data/extra"
    timeml_DATA_DIR = "/Users/doduydao/daodd/PycharmProjects/TRE/data/TimeBank1.2/data/timeml"
    path = "/Users/doduydao/daodd/PycharmProjects/TRE/data/TimeBank1.2/data/"
    train_df = load_data(extra_DATA_DIR)
    train_df.to_csv(path + 'extra.csv', index=False)

    test_df = load_data(timeml_DATA_DIR)
    test_df.to_csv(path + 'timeml.csv', index=False)
