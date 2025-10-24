# Written by: rgarc461
# Last Updated by: asing118


from dataclasses import dataclass, field


@dataclass
class Instance:
    """

    Class to construct Instance objects with attributes of event_instance_id, event, tense, aspect, pos, polarity,
    modality, signal, and cardinality.

    """
    event_instance_id: int
    event: str
    event_class: str
    tense: str
    aspect: str
    pos: str
    polarity: str
    modality: str = field(default=None)
    signal: str = field(default=None)
    cardinality: str = field(default=None)

    # def __post_init__(self):
    #     if self.event_instance_id < 1:
    #         raise ValueError("Event instance id cannot be less than 1.")

    #     if self.event is None:
    #         raise TypeError("Event can not be None")

    #     if type(self.event) is not str:
    #         raise TypeError("Event must be a String")

    #     accepted_tenses = {"PAST", "PRESENT", "FUTURE", "NONE", "INFINITIVE", "PRESPART", "PASTPART"}
    #     if self.tense not in accepted_tenses:
    #         raise TypeError(self.tense + " is not an acceptable tense.")

    #     accepted_aspects = {"PROGRESSIVE", "PERFECTIVE", "PERFECTIVE_PROGRESSIVE", "NONE"}
    #     if self.aspect not in accepted_aspects:
    #         raise TypeError(self.aspect + " is not an acceptable aspect.")

    #     accepted_pos = {"ADJECTIVE", "NOUN", "VERB", "PREPOSITION", "OTHER"}
    #     if self.pos not in accepted_pos:
    #         raise TypeError(self.pos + " is not an acceptable pos.")

    #     accepted_polarity = {"POS", "NEG"}
    #     if self.polarity not in accepted_polarity:
    #         raise TypeError(self.polarity + " is not an acceptable polarity")

    #     if self.modality is not None:
    #         if "modality" in self.modality:
    #             self.quant = self.modality.split("=")[1]
    #         if '"' in self.modality:
    #             self.modality = self.modality.replace('"', '')
    #         self.modality = self.modality.strip()
    #         if len(self.modality) == 0:
    #             raise ValueError("Modality can not be empty string or all whitespace")

    #     if self.signal is not None and type(self.signal) is not str:
    #         raise Exception("signal must be stored as a str")

    def get_id_str(self):
        return "eiid" + str(self.event_instance_id)

    def to_json(self):
        ret = {'id': self.get_id_str(), 'tense': self.tense, 'aspect': self.aspect, 'pos': self.pos, 'polarity': self.polarity}

        if self.modality is not None:
            ret['modality'] = self.modality
        else:
            ret['modality'] = 'null'

        
        if self.cardinality is not None:
            ret['cardinality'] = self.cardinality
            
        else:
            ret['cardinality'] = 'null'
            
        if self.signal is not None or self.signal != "":
            ret['signal'] = self.signal
        else:
            ret['signal'] = '{}'

        # ret['event'] = {"id": self.event, "eventClass":self.event_class}

        ret['event'] = {"id": self.event, "eventClass": self.event_class}
        return ret

    def __hash__(self):
        return hash(self.event_instance_id + hash(self.aspect) + hash(self.cardinality) + hash(self.modality))
