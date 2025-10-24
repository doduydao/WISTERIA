# Written by: vfern124
# Last updated: asing118

from dataclasses import dataclass, field


# Do not change order of variables, will cause TimeMLParser to explode... Alert vfern124 before making changes
@dataclass
class Link:
    """
    Link class represents an edge between a starting node and a related node.
    """
    link_id: int
    link_tag: str
    rel_type: str
    start_node: str  # Must be Instance or TimeX
    related_to_node: str  # Must be Instance or TimeX
    signal: str = field(default=None)
    origin_type: str = field(default=None)

    # def __post_init__(self):
    #     if self.link_id < 0:
    #         raise Exception("Error: Link ID cannot be less than 0")
    #
    #     accepted_origin_types = {"USER", "MANUALLY", "CLOSURE", None}
    #     if self.link_tag == 'TLINK' and self.origin_type not in accepted_origin_types:
    #         raise Exception("Error: Not acceptable Origin Type - " + self.origin_type)
    #
    #     accepted_link_tags = {"ALINK", "SLINK", "TLINK"}
    #     if self.link_tag not in accepted_link_tags:
    #         raise Exception("Error: Not acceptable Link Tag - l" + str(self.link_id))
    #
    #     accepted_alink_types = {"INITIATES", "CULMINATES", "TERMINATES", "CONTINUES", "REINITIATES"}
    #     if self.link_tag == 'ALINK' and self.rel_type not in accepted_alink_types:
    #         raise Exception("Error: Not acceptable Alink Relation Type - l" + str(self.link_id))
    #
    #     accepted_slink_types = {"MODAL", "EVIDENTIAL", "NEG_EVIDENTIAL", "FACTIVE", "COUNTER_FACTIVE", "CONDITIONAL"}
    #     if self.link_tag == 'SLINK' and self.rel_type not in accepted_slink_types:
    #         raise Exception("Error: Not acceptable Slink Relation Type - l" + str(self.link_id))
    #
    #     accepted_tlink_types = {"BEFORE", "AFTER", "INCLUDES", "IS_INCLUDED", "DURING", "DURING_INV", "SIMULTANEOUS",
    #                             "IAFTER", "IBEFORE", "IDENTITY", "BEGINS", "ENDS", "BEGUN_BY", "ENDED_BY"}
    #     if self.link_tag == 'TLINK' and self.rel_type not in accepted_tlink_types:
    #         raise Exception("Error: Not acceptable Tlink Relation Type - l" + str(self.link_id))
    #
    #     if self.start_node is None:
    #         raise Exception("Error: Start Node cannot be None - l" + str(self.link_id))
    #
    #     if self.related_to_node is None:
    #         raise Exception("Error: Related to Node cannot be None - l" + str(self.link_id))
    #
    #     if ("t" not in self.start_node[0]) and ("eiid" not in self.start_node):
    #         raise Exception("Error: Start Node must be Instance or TimeX - " + self.start_node + " - l" +
    #                         str(self.link_id))
    #
    #     if ("t" not in self.related_to_node[0]) and ("eiid" not in self.related_to_node):
    #         raise Exception("Error: Related to Node must be Instance or TimeX - " + self.related_to_node +
    #                         " - l" + str(self.link_id))
    #
    #     if self.signal is not None and type(self.signal) is not str:
    #         raise Exception("Should only be a str id for Signal")

    def get_id_str(self):
        return 'lid' + str(self.link_id)

    def __hash__(self):
        return hash(str(self.link_id) + self.link_tag + self.start_node + self.related_to_node)

    def __eq__(self, other):
        if self.link_id != other.link_id:
            return False
        if self.link_tag != other.link_tag:
            return False
        if self.origin_type != other.origin_type:
            return False
        if self.start_node != other.start_node:
            return False
        if self.related_to_node != other.related_to_node:
            return False
        if self.signal != other.signal:
            return False
        if self.rel_type != other.rel_type:
            return False

        return True

    def to_json(self):
        if self.origin_type is not None:
            origin_type = self.origin_type
        else:
            origin_type = ""

        ret = {"id": self.get_id_str(),
               "linkTag": self.link_tag,
               "start_node": self.start_node,
               "end_node": self.related_to_node,
               "temporalRelation": self.rel_type,
               "origin": origin_type,
               "signal": self.signal
               }

        return ret
