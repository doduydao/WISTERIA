# Written by: vfern124
# Last updated: asing118

# Using regular expressions to parse the TimeML text
import re
from TBPreprocess.Instance import Instance
from TBPreprocess.Event import Event
from TBPreprocess.Signal import Signal
from TBPreprocess.Link import Link
from TBPreprocess.TimeX import TimeX

# from Instance import Instance
# from Event import Event
# from Signal import Signal
# from Link import Link
# from TimeX import TimeX

import copy


# Helper function for parsing
def __parse_phrase(to_parse) -> str:
    to_parse = '>' + to_parse + '<'
    parse_pattern = re.compile(r'>([^<]+)<')
    parse_matches = parse_pattern.finditer(to_parse)
    phrase = ''
    for match_parses in parse_matches:
        phrase += match_parses.group(1)
    return phrase.replace("\n", " ").replace("  ", " ")

def get_span(text, sub_text):
    print(text, sub_text)
    start = text.find(sub_text)  # Tìm chỉ số bắt đầu
    if start == -1:
        return None  # Trả về None nếu không tìm thấy
    end = start + len(sub_text)  # Chỉ số kết thúc
    return (start, end)


def parse_events(time_ml_text) -> list[Event]:
    """
    Searches a given text for possible Event objects
    :param time_ml_text: a str tagged with Time ML Events
    :return: a list of Event objects
    """

    event_pattern = re.compile(r'<EVENT([^>]+)>((?:(?!</EVENT)[\S\s])+)</EVENT>')
    event_matches = event_pattern.finditer(time_ml_text)

    events = []
    for match_event in event_matches:
        eid_pattern = re.compile(r'eid="e(\d+)"')
        eid = int(eid_pattern.findall(match_event.group(1))[0])

        e_class_pattern = re.compile(r'class="([^"]+)"')
        e_class = e_class_pattern.findall(match_event.group(1))[0]

        stem_pattern = re.compile(r'stem="([^"]+)"')
        stem_temp = stem_pattern.findall(match_event.group(1))
        if len(stem_temp) != 0:
            stem = stem_temp[0]
        else:
            stem = match_event.group(2)
        span = match_event.span()
        event_start = span[0]
        event_end = span[1]
        raw_tag = match_event.group()
        events += [Event(eid, e_class, event_start, event_end, stem, raw_tag=raw_tag)]


    return events


def parse_signals(time_ml_text) -> list[Signal]:
    """
    Searches a given text for possible Signal objects
    :param time_ml_text: a str tagged with Time ML Signals
    :return: a list of Signal objects
    """
    signal_pattern = re.compile(r'<SIGNAL sid="s(\d+)">((?:(?!</SIGNAL)[\S\s])+)</SIGNAL>')
    signal_matches = signal_pattern.finditer(time_ml_text)

    signals = []
    for signal in signal_matches:
        id = int(signal.group(1))
        string = __parse_phrase(signal.group(2))
        start = signal.span()[0]
        end = signal.span()[1]
        signals += [Signal(id, string, start, end)]

    return signals


def parse_instances(time_ml_text) -> list[Instance]:
    """
    Searches a given text for possible Instance objects
    :param time_ml_text: a str tagged with Time ML Instances
    :return: a list of Instance objects
    """
    event_list = parse_events(time_ml_text)
    signals = parse_signals(time_ml_text)

    instance_pattern = re.compile(r'<MAKEINSTANCE([^>]+)/>')
    instance_matches = instance_pattern.finditer(time_ml_text)
    
    instances = []
    for match_instance in instance_matches:
        eiid_pattern = re.compile(r'eiid="ei([^"]+)"')
        eiid = int(eiid_pattern.findall(match_instance.group(1))[0])

        eventID_pattern = re.compile(r'eventID="e([^"]+)"')
        eventID = int(eventID_pattern.findall(match_instance.group(1))[0])
        event = None
        for e in event_list:
            if e.eid == eventID:
                event = e
                break

        tense_pattern = re.compile(r'tense="([^"]+)"')
        tense = ''
        if "tense" in match_instance.group():
            tense = tense_pattern.findall(match_instance.group(1))[0]

        aspect_pattern = re.compile(r'aspect="([^"]+)"')
        aspect = ''
        if "aspect" in match_instance.group():
            aspect = aspect_pattern.findall(match_instance.group(1))[0]

        pol_pattern = re.compile(r'polarity="([^"]+)"')
        pol = ''
        if "polarity" in match_instance.group():
            pol = pol_pattern.findall(match_instance.group(1))[0]

        signal_pattern = re.compile(r'signalID="s(\d+)"')
        signal_temp = signal_pattern.findall(match_instance.group(1))
        signal = None
        if len(signal_temp) != 0:
            signal_id = int(signal_temp[0])

            for s in signals:
                if s.signal_id == signal_id:
                    signal = s
                    break

        card_pattern = re.compile(r'cardinality="([^"]+)"')
        card_temp = card_pattern.findall(match_instance.group(1))
        cardinality = None
        if len(card_temp) != 0:
            cardinality = card_temp[0]

        pos_pattern = re.compile(r'pos="([^"]+)"')
        pos = ''
        if "pos" in match_instance.group():
            pos = pos_pattern.findall(match_instance.group(1))[0]
        if pos == 'UNKNOWN' or pos == '':
            pos = 'OTHER'
        mod_pattern = re.compile(r'modality="[^"]+"')
        mod_temp = mod_pattern.findall(match_instance.group(1))
        modality = None
        if len(mod_temp) != 0:
            modality = mod_temp[0]
        if signal is None:
            instances += [
                Instance(eiid, event.get_id_str(), event.event_class, tense, aspect, pos, pol, modality, "",
                         cardinality)]

        else:
            instances += [Instance(eiid, event.get_id_str(), event.event_class, tense, aspect, pos,
                                   pol, modality, signal.get_id_str(), cardinality)]

    return instances


def parse_timex(time_ml_text) -> list[TimeX]:
    """
    Searches a given text for possible Timex objects
    :param time_ml_text: a str tagged with Time ML Timexes
    :return: a list of Timex objects
    """

    timex_pattern = re.compile(r'<TIMEX3([^>]+)>((?:(?!</TIMEX3)[\S\s])+)</TIMEX3>')
    timex_matches = timex_pattern.finditer(time_ml_text)
    timexes = []
    for match_timex in timex_matches:
        raw_tag = match_timex.group()
        phrase = __parse_phrase(match_timex.group(2))
        stem = phrase
        tid_pattern = re.compile(r'tid="t(\d+)"')
        tid = int(tid_pattern.findall(match_timex.group(1))[0])

        type_pattern = re.compile(r'type="([^"]+)"')
        type_temp = type_pattern.findall(match_timex.group(1))
        type = 'NONE'
        if len(type_temp):
            type = type_temp[0]

        value_pattern = re.compile(r'value="([^"]+)"')
        value = value_pattern.findall(match_timex.group(1))[0]

        mod_pattern = re.compile(r'mod="([^"]+)"')
        mod_temp = mod_pattern.findall(match_timex.group(1))
        mod = 'NONE'
        if len(mod_temp):
            mod = mod_temp[0]

        temporal_pattern = re.compile(r'temporalFunction="([^"]+)"')

        tmp = temporal_pattern.findall(match_timex.group(1))
        if len(tmp) != 0:
            temporal = bool(tmp[0])
        else:
            temporal = False

        func_doc_pattern = re.compile(r'functionInDocument="([^"]+)"')
        func_doc_temp = func_doc_pattern.findall(match_timex.group(1))
        func_doc = 'NONE'
        if len(func_doc_temp):
            func_doc = func_doc_temp[0]

        anchor_pattern = re.compile(r'anchorTimeID="t(\d+)"')
        anchor_temp = anchor_pattern.findall(match_timex.group(1))
        anchor = None
        if len(anchor_temp):
            anchor = int(anchor_temp[0])

        quant_pattern = re.compile(r'quant="[^"]*"')
        quant_temp = quant_pattern.findall(match_timex.group(1))
        quant = None
        if len(quant_temp):
            quant = quant_temp[0]

        freq_pattern = re.compile(r'freq="([^"]+)"')
        freq_temp = freq_pattern.findall(match_timex.group(1))
        freq = None
        if len(freq_temp):
            freq = freq_temp[0]

        begin_point_pattern = re.compile(r'beginPoint="t([^"]+)"')
        begin_point_temp = begin_point_pattern.findall(match_timex.group(1))
        begin_point = None
        if len(begin_point_temp):
            begin_point = int(begin_point_temp[0])

        end_point_pattern = re.compile(r'endPoint="t([^"]+)"')
        end_point_temp = end_point_pattern.findall(match_timex.group(1))
        end_point = None
        if len(end_point_temp):
            end_point = int(end_point_temp[0])
        # text = match_timex.group()
        span = match_timex.span()
        # start_timex, end_timex = get_span(text, stem)
        # timex_start = start_timex + span[0]
        # timex_end = end_timex + span[0]
        timex_start = span[0]
        timex_end = span[1]
        timexes += [TimeX(tid, value, temporal, phrase, timex_start, timex_end, type, mod, func_doc,
                          anchor, quant, freq, begin_point, end_point, stem, raw_tag)]
    return timexes


def parse_links(time_ml_text, instances, signals, timexes) -> list[Link]:
    """
    Searches a given text for possible Link objects
    :param time_ml_text: a str tagged with Time ML Links
    :return: a list of Link objects
    """
    # instances = parse_instances(time_ml_text)
    # signals = parse_signals(time_ml_text)
    # timexes = parse_timex(time_ml_text)

    link_pattern = re.compile(r'<([AST]LINK)([^>]+)/>')
    link_matches = link_pattern.finditer(time_ml_text)

    links = []
    
    for match_link in link_matches:
        tag = match_link.group(1)

        link_id_pattern = re.compile(r'lid="l([\d]+)"')
        link_id = int(link_id_pattern.findall(match_link.group(2))[0])

        rel_type_pattern = re.compile(r'relType="([^"]+)"')
        rel_type = rel_type_pattern.findall(match_link.group(2))[0]

        start_pattern = re.compile(r'timeID="t(\d+)"')
        start_temp = start_pattern.findall(match_link.group(2))
        start = None
        if len(start_temp) != 0:
            for t in timexes:
                if t.tID == int(start_temp[0]):
                    start = t
                    break
        else:
            start_pattern = re.compile(r'eventInstanceID="ei(\d+)"')
            start_temp = start_pattern.findall(match_link.group(2))
            if len(start_temp) != 0:
                for i in instances:
                    if i.event_instance_id == int(start_temp[0]):
                        start = i
                        break
        
        related_pattern = re.compile(r'relatedToTime="t(\d+)"')
        related_temp = related_pattern.findall(match_link.group(2))
        related = None
        if len(related_temp) != 0:
            for t in timexes:
                if t.tID == int(related_temp[0]):
                    related = t
                    break
        else:
            related_pattern = re.compile(r'(subordinatedEventInstance|relatedToEventInstance)="ei(\d+)"')
            related_temp = related_pattern.findall(match_link.group(2))
            if len(related_temp) != 0:
                for i in instances:
                    if i.event_instance_id == int(related_temp[0][1]):
                        related = i
                        break

        signal_pattern = re.compile(r'signalID="s(\d+)"')
        signal_temp = signal_pattern.findall(match_link.group(2))
        signal = None
        if len(signal_temp) != 0:
            for s in signals:
                if s.signal_id == int(signal_temp[0]):
                    signal = s
                    break

        origin_pattern = re.compile(r'origin="([^"]+)"')
        origin_temp = origin_pattern.findall(match_link.group(2))
        origin = None
        if len(origin_temp) != 0:
            origin = origin_temp[0].upper()


        if start is None or related is None:
            if start is None:
                print(f"Error: Link {link_id}", match_link.group(), "start:", start)
            else:
                print(f"Error: Link {link_id}", match_link.group(), "related:", related)
            continue

        if signal is None:
            
            links += [
                Link(link_id, tag, rel_type, start.get_id_str(), related.get_id_str(), "", origin)]
        else:
            links += [Link(link_id, tag, rel_type, start.get_id_str(), related.get_id_str(), signal.get_id_str(), origin)]
    return links

def remove_redundants(time_ml_text):
    time_ml_text_copy = time_ml_text
    # remove header, timeML_tag, instances, Tlink
    time_ml_text_copy = re.sub('<\?xml.*\?>', r'', time_ml_text_copy)  # xml_version_pattern
    time_ml_text_copy = re.sub(r'<MAKEINSTANCE([^>]+)/>', r'', time_ml_text_copy)  # instance_pattern
    time_ml_text_copy = re.sub(r'<([AST]LINK)([^>]+)/>', r'', time_ml_text_copy)  # link_pattern
    time_ml_text_copy = re.sub(r'<([AST]LINK)([^>]+)/>', r'', time_ml_text_copy)  # link_pattern

    time_ml_text_copy = re.sub(r'<TimeML([^>]+)>', r'', time_ml_text_copy)  # TimeML_tag_1_pattern
    time_ml_text_copy = re.sub(r'</TimeML>', r'', time_ml_text_copy)  # TimeML_tag_2_pattern
    time_ml_text_copy = re.sub(r'<EXTRAINFO>((?:(?!</EXTRAINFO)[\S\s])+)</EXTRAINFO>', r'',
                               time_ml_text_copy)  # DOCID_pattern

    # extract from <DOCID>, <DCT>, <TITLE>, <TEXT>, <EVENT>, <TIMEX3>, <SIGNAL>
    time_ml_text_copy = re.sub(r'<DOCID>([\S\s]+)</DOCID>', r'\1', time_ml_text_copy)  # DOCID_pattern
    time_ml_text_copy = re.sub(r'<DCT>([\S\s]+)</DCT>', r'\1', time_ml_text_copy)  # DCT_pattern
    time_ml_text_copy = re.sub(r'<TITLE>([\S\s]+)</TITLE>', r'\1', time_ml_text_copy)  # TITLE_pattern
    time_ml_text_copy = re.sub(r'<TEXT>([\S\s]+)</TEXT>', r'\1', time_ml_text_copy)  # TEXT_pattern

    # remove empty lines
    time_ml_text_copy = "\n".join([ll.rstrip() for ll in time_ml_text_copy.splitlines() if ll.strip()]).strip()
    time_ml_text_copy = time_ml_text_copy.replace("\n", " ").replace("  ", " ")
    return time_ml_text_copy

def parse_raw_text(time_ml_text) -> str:
    time_ml_text_copy = remove_redundants(time_ml_text)

    time_ml_text_copy = re.sub(r'<EVENT([^>]+)>((?:(?!</EVENT)[\S\s])+)</EVENT>', r'\2', time_ml_text_copy) # event_pattern
    time_ml_text_copy = re.sub(r'<TIMEX3([^>]+)>((?:(?!</TIMEX3)[\S\s])+)</TIMEX3>', r'\2', time_ml_text_copy) # timex_pattern
    time_ml_text_copy = re.sub(r'<SIGNAL sid="s(\d+)">((?:(?!</SIGNAL)[\S\s])+)</SIGNAL>', r'\2', time_ml_text_copy) # signal_pattern

    # remove empty lines
    time_ml_text_copy = "\n".join([ll.rstrip() for ll in time_ml_text_copy.splitlines() if ll.strip()])

    return time_ml_text_copy

def parse_text_add_labels(time_ml_text) -> str:
    time_ml_text_copy = time_ml_text
    # remove header, timeML_tag, instances, Tlink
    time_ml_text_copy = re.sub('<\?xml.*\?>', r'', time_ml_text_copy) # xml_version_pattern
    time_ml_text_copy = re.sub(r'<MAKEINSTANCE([^>]+)/>', r'', time_ml_text_copy) # instance_pattern
    time_ml_text_copy = re.sub(r'<([AST]LINK)([^>]+)/>', r'', time_ml_text_copy) # link_pattern
    time_ml_text_copy = re.sub(r'<([AST]LINK)([^>]+)/>', r'', time_ml_text_copy) # link_pattern

    time_ml_text_copy = re.sub(r'<TimeML([^>]+)>', r'', time_ml_text_copy) # TimeML_tag_1_pattern
    time_ml_text_copy = re.sub(r'</TimeML>', r'', time_ml_text_copy) #TimeML_tag_2_pattern
    time_ml_text_copy = re.sub(r'<EXTRAINFO>((?:(?!</EXTRAINFO)[\S\s])+)</EXTRAINFO>', r'', time_ml_text_copy) # DOCID_pattern

    # extract from <DOCID>, <DCT>, <TITLE>, <TEXT>, <EVENT>, <TIMEX3>, <SIGNAL>
    time_ml_text_copy = re.sub(r'<DOCID>([\S\s]+)</DOCID>', r'\1', time_ml_text_copy) # DOCID_pattern
    time_ml_text_copy = re.sub(r'<DCT>([\S\s]+)</DCT>', r'\1', time_ml_text_copy) # DCT_pattern
    time_ml_text_copy  = re.sub(r'<TITLE>([\S\s]+)</TITLE>', r'\1', time_ml_text_copy) #TITLE_pattern
    time_ml_text_copy = re.sub(r'<TEXT>([\S\s]+)</TEXT>', r'\1', time_ml_text_copy) #TEXT_pattern

    time_ml_text_copy = re.sub(r'<EVENT([^>]*)eid="(e\d+)"([^>]*)>((?:(?!</EVENT)[\S\s])+)</EVENT>', r'<span id="\2" class="ui red label">\4</span>', time_ml_text_copy) # event_pattern
    time_ml_text_copy = re.sub(r'<TIMEX3([^>]*)tid="(t[0-9]+)"([^>]*)>((?:(?!</TIMEX3)[\S\s])+)</TIMEX3>', r'<span id="\2" class="ui blue label">\4</span>', time_ml_text_copy) # timex_pattern
    time_ml_text_copy = re.sub(r'<SIGNAL sid="s(\d+)">((?:(?!</SIGNAL)[\S\s])+)</SIGNAL>', r'\2', time_ml_text_copy) # signal_pattern

    # print("time_ml_text_copy:",time_ml_text_copy)
    # remove empty lines
    time_ml_text_copy = "\n".join([ll.rstrip() for ll in time_ml_text_copy.splitlines() if ll.strip()])

    if '</s>' in time_ml_text_copy:
        time_ml_text_copy = re.sub('\n+', r' ', time_ml_text_copy)
        time_ml_text_copy = "\n".join([ll.rstrip() for ll in time_ml_text_copy.split('</s>') if ll.strip()])

        # remove <s> </s>
        time_ml_text_copy = re.sub('<s>', r'', time_ml_text_copy)
        # print("time_ml_text_copy:",time_ml_text_copy)
        time_ml_text_copy = re.sub('</s>', r'', time_ml_text_copy)
    # time_ml_text_copy = re.sub(' +', r' ', time_ml_text_copy)
    return time_ml_text_copy

def parse_metadata(time_ml_text) -> list[str]:
    # First find where the text begins
    text_match = re.compile(r'<TEXT>').finditer(time_ml_text)
    stop = 0
    for match in text_match:
        stop = int(match.span()[0])

    # Then collect any matches before the text, which correspond to the metadata
    meta_pattern = re.compile(r'<(\w+)>((?:(?!</\1>)[\S\s])+)(</\1>)')
    meta_matches = meta_pattern.finditer(time_ml_text[0: stop:])

    data = []
    for match_data in meta_matches:
        phrase = __parse_phrase(match_data.group(2))
        data += [[match_data.group(1), phrase]]

    return data

def merge_eiid_to_eid(instances, events):
    new_events = []

    for instance in instances:
        new_event = None
        instance_json = instance.to_json()
        # print(instance_json)
        for event in events:
            event_json = event.to_json()
            if instance_json['event']['id'] == event_json['id']:
                new_event = copy.deepcopy(event)
                new_event.set_eiid(instance_json['id'])
                break
        # print(new_event.to_json())
        new_events.append(new_event)
        # print()
    return new_events

def change_eid_to_eiid(text_added_labels, events):
    for event in events:
        event_json = eval(event.to_json())
        # print(event_json)
        eid = "id=\"" + event_json['id']+ "\""
        eiid = "id=\"" + event_json['eiid']+ "\""
        text_added_labels = text_added_labels.replace(eid, eiid)
    # print(text_added_labels)
    return text_added_labels

def parse_for_visualizing(time_ml_text) -> (str, str, list[Link], list[TimeX], list[Instance]):
    """
    Searches a given text for TimeML Tags
    :param time_ml_text: a str annotated in TimeML
    :return: the metadata, text_with_labels, and then a list for each category: links, timexes, instances, signals, and events
    """
    instances = parse_instances(time_ml_text)
    events = parse_events(time_ml_text)
    events = merge_eiid_to_eid(instances, events)
    text_added_labels = parse_text_add_labels(time_ml_text)
    text_added_labels = change_eid_to_eiid(text_added_labels, events)
    signals = parse_signals(time_ml_text)
    timex = parse_timex(time_ml_text)
    links = parse_links(time_ml_text, instances, signals, timex)
    return parse_metadata(time_ml_text), text_added_labels, links, \
               timex + instances, signals, events, timex


def parse(time_ml_text, only_tlinks=False):
    """
    Searches a given text for TimeML Tags
    :param time_ml_text: a str annotated in TimeML
    :return: the metadata, raw text, and then a list for each category: links, timexes, instances, signals, and events
    """
    time_ml_text_copy = remove_redundants(time_ml_text)
    instances = parse_instances(time_ml_text)
    events = merge_eiid_to_eid(instances, parse_events(time_ml_text_copy))
    signals = parse_signals(time_ml_text_copy)
    timex = parse_timex(time_ml_text_copy)
    links = parse_links(time_ml_text, instances, signals, timex)
    tlinks = []
    if only_tlinks:
        for link in links:
            if link.link_tag != "TLINK":
                continue
            tlinks.append(link)
        return parse_metadata(time_ml_text), time_ml_text_copy, tlinks, \
            timex + instances, signals, events, timex

    return parse_metadata(time_ml_text), time_ml_text_copy, links, \
        timex + instances, signals, events, timex


def parse_dict(time_ml_text) -> (str, str, list[Link], list[TimeX], list[Instance], list[Event], list[Signal]):
    return {
        'metadata': parse_metadata(time_ml_text),
        'raw_text': parse_raw_text(time_ml_text),
        'links': parse_links(time_ml_text),
        'timexes': parse_timex(time_ml_text),
        'instances': parse_instances(time_ml_text),
        'events': parse_events(time_ml_text),
        'signals': parse_signals(time_ml_text)
    }

def read_file_data(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()


def make_data_structure(links, signals, events, timexs):
    entity_dict = dict()
    for event in events:
        event = event.to_json()
        id = event['id']
        stem = event['stem']
        eiid = event['eiid']
        span = (event['event_start'], event['event_end'])
        entity_dict[eiid] = {'id': id, 'stem': stem, 'span': span}

    for timex in timexs:
        timex = timex.to_json()
        id = timex['id']
        stem = timex['stem']
        span = (timex['timex_start'], timex['timex_end'])
        entity_dict[id] = {'id': id, 'stem': stem, 'span': span}
    # Sắp xếp theo span[0]
    entity_dict = dict(sorted(entity_dict.items(), key=lambda item: item[1]['span'][0]))

    # In ra (nếu muốn)
    # for eiid, info in sorted_entity_dict.items():
    #     print(f"{eiid}: {info}")

    signal_dict = dict()
    for signal in signals:
        signal = signal.to_json()
        id = signal['id']
        stem = signal['signalString']
        span = (signal['start'], signal['end'])
        signal_dict[id] = {'id': id, 'stem': stem, 'span': span}

    # print("\nlinks:")
    tlinks = dict()
    for link in links:
        link = link.to_json()
        id = link['id']
        label = link['temporalRelation']
        start_node = link['start_node']
        end_node = link['end_node']
        signal = link['signal']
        tlinks[id] = {'label': label, "start_node": start_node, "end_node": end_node, "signal": signal}

    # for link, info in tlinks.items():
    #     print(f"{link}: {info}")

    return tlinks, entity_dict, signal_dict


def clean_text_and_update_spans(raw_text, entity_dict, signal_dict):
    merged_dict = {**entity_dict, **signal_dict}

    # Sắp xếp tag_items theo span để xử lý theo thứ tự
    tag_items = sorted(merged_dict.items(), key=lambda item: item[1]['span'][0])

    cleaned_text = ""
    last_idx = 0
    offset = 0  # tổng số ký tự đã xóa
    updated_entity_dict = {}
    updated_signal_dict = {}

    for key, item in tag_items:
        start, end = item['span']

        # Trích nội dung chứa tag
        segment = raw_text[start:end]

        # Loại bỏ tag và giữ lại nội dung giữa các tag
        inner_text = re.sub(r'</?(TIMEX3|EVENT|SIGNAL)[^>]*>', '', segment)

        # Thêm phần trước tag vào cleaned_text
        cleaned_text += raw_text[last_idx:start] + inner_text

        # Cập nhật span mới dựa vào offset
        new_start = len(cleaned_text) - len(inner_text)
        new_end = new_start + len(inner_text)

        if key in entity_dict:
            updated_entity_dict[key] = {'id': item['id'],
                                  'stem': item['stem'],
                                  'text': inner_text,
                                  'span': (new_start, new_end)
                                  }
        else:
            updated_signal_dict[key] = {'id': item['id'],
                                  'stem': item['stem'],
                                  'text': inner_text,
                                  'span': (new_start, new_end)
                                  }

        # Cập nhật chỉ số tiếp theo
        last_idx = end

    # Thêm phần còn lại của raw_text
    cleaned_text += raw_text[last_idx:]

    return cleaned_text, updated_entity_dict, updated_signal_dict

if __name__ == "__main__":
    # data = read_file_data('../pytlex_data/TimeBankCorpus/example_data.tml')
    # output = parse_dict(data)
    #
    # for key in output:
    #     print(key, end=': ')
    #     print(output[key])
    path = "/Users/doduydao/daodd/PycharmProjects/TRE/data/TimeBank-dense/train/PRI19980213.2000.0313.tml"
    with open(path, 'r') as open_file:
        file_string = open_file.read().replace(' & ', '   ').replace('L&D', 'LnD').replace('&', 'n')
    # file_id = file.split(".")[0]
    metadata, raw_text, links, instances, signals, events, timexs = parse(file_string, only_tlinks=True)
    #
    for e in events:
        print(e.to_json())

    for t in timexs:
        print(t.to_json())

    # print("metadata:", metadata)
    # print("\nraw_text:", raw_text)
    tlinks, entity_dict, signal_dict = make_data_structure(links, signals, events, timexs)
    cleaned_text, updated_entity_dict, updated_signal_dict = clean_text_and_update_spans(raw_text, entity_dict, signal_dict)

    # print("\ncleaned_text:", cleaned_text)

    # for k, v in updated_entity_dict.items():
    #     print(k, v)
    #
    # print()
    # for k, v in updated_signal_dict.items():
    #     print(k, v)

    for link_id, link in tlinks.items():
        label = link['label']
        # print(link)
        start_entity = updated_entity_dict[link['start_node']]
        end_entity = updated_entity_dict[link['end_node']]


    # for k, v in updated_entity_dict.items():
    #     span = v['span']
    #     if v['text'] != cleaned_text[span[0]:span[1]]:
    #         print(k, v,"------", cleaned_text[span[0]:span[1]])
