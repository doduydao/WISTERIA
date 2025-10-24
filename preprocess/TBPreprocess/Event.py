class Event:
    def __init__(self,
                 eid,
                 event_class,
                 event_start,
                 event_end,
                 stem=None,
                 eiid=-1,
                 raw_tag=''):
        self.eid = eid
        self.event_class = event_class
        self.event_start = event_start
        self.event_end = event_end
        self.stem = stem
        self.eiid = eiid
        self.raw_tag = raw_tag



    def __post_init__(self):
        if self.eid < 1:
            raise Exception("EventID cannot be less than 1.")

        if self.event_class is None:
            raise Exception("Event Class needs to be defined")

        accepted_event_classes = {"REPORTING", "PERCEPTION", "ASPECTUAL", "I_ACTION", "I_STATE", "STATE", "OCCURRENCE"}
        if self.event_class not in accepted_event_classes:
            raise Exception("That is not a valid Event Class - " + self.event_class)

        if self.stem is not None:
            if ">" in self.stem and "<" in self.stem:
                front = self.stem.split(">")
                back = front[1].split("<")
                self.stem = back[0]
            self.stem = self.stem.strip()
            if len(self.stem) == 0:
                raise Exception("Stem cannot be empty or all whitespace.")

    def get_id_str(self):
        """
        Returns the full string representation of the eventID.
        Each event has to be identified by a unique ID number and String.
        """
        eid_str = "e" + str(self.eid)
        return eid_str

    def get_eiid_str(self):
        """
        Returns the full string representation of the eventID.
        Each event has to be identified by a unique ID number and String.
        """
        eiid_str = "eiid" + str(self.eiid)
        return eiid_str
    def set_eiid(self, eiid):
        """
        Returns the full string representation of the eventID.
        Each event has to be identified by a unique ID number and String.
        """
        if 'eiid' in eiid:
            self.eiid = eiid[4:]


    def to_string(self):
        """
        Returns the Event information as a String.
        """
        event_string = "EVENT: eid = " + self.get_id_str() + ", class = " + str(self.event_class) + ", stem = " \
                       + str(self.stem)
        return event_string

    def to_json(self):
        """
        Returns the JSON (RFC 8259) representation of the event.
        """
        if self.stem is None:
            stem = "None"
        else:
            stem = self.stem
        ret = {"eiid": self.get_eiid_str(),
               "id": self.get_id_str(),
               "eventClass": self.event_class,
               "stem": stem,
               'event_start': self.event_start,
               'event_end': self.event_end}
        return ret
    def __hash__(self):
        return hash(self.eid + hash(self.event_class) + hash(self.stem))
