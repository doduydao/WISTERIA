
class TimeX:
    def __init__(self, tID,
                 value,
                 temporalFunction,
                 phrase,
                 timex_start=None,
                 timex_end=-1,
                 type='NONE',
                 mod='NONE',
                 documentFunction='NONE',
                 anchorID=None,
                 quant=None,
                 freq=None,
                 beginPoint=None,
                 endPoint=None,
                 stem=None,
                 raw_tag=''):
        self.tID = tID
        self.value = value
        self.temporalFunction = temporalFunction
        self.phrase = phrase
        self.timex_start = timex_start
        self.timex_end = timex_end
        self.type = type
        self.mod = mod
        self.documentFunction = documentFunction
        self.anchorID = anchorID
        self.quant = quant
        self.freq = freq
        self.beginPoint = beginPoint
        self.endPoint = endPoint
        self.stem = stem
        self.raw_tag = raw_tag

    # def __post_init__(self):
    #     if self.tID < 0:
    #         raise Exception("Error: Time id can't be less than zero.")

    #     accepted_types = {"DATE", "TIME", "DURATION", "SET", "NONE"}
    #     if self.type not in accepted_types:
    #         raise Exception("Error: Not acceptable TimeX Type - t" + str(self.tID))

    #     accepted_mods = {"BEFORE", "AFTER", "ON_OR_BEFORE", "ON_OR_AFTER", "LESS_THAN", "MORE_THAN", "EQUAL_OR_LESS",
    #                      "EQUAL_OR_MORE", "START", "MID", "END", "APPROX", "NONE"}
    #     if self.mod not in accepted_mods:
    #         raise Exception("Error: Not acceptable TimeX Mod - t" + str(self.tID))

    #     accepted_doc_functions = {"CREATION_TIME", "MODIFICATION_TIME", "PUBLICATION_TIME", "RELEASE_TIME",
    #                               "RECEPTION_TIME", "EXPIRATION_TIME", "NONE"}
    #     if self.documentFunction not in accepted_doc_functions:
    #         raise Exception("Error: Not acceptable TimeX Document Function - t" + str(self.tID))

    #     if self.anchorID is not None:
    #         if self.anchorID < 0:
    #             raise Exception("Error: Anchor id can't be less than zero. - t" + str(self.tID))

    #     if self.beginPoint is not None:
    #         if self.beginPoint < 0:
    #             raise Exception("Error: beginPoint can't be less than zero. - t" + str(self.tID))

    #     if self.endPoint is not None:
    #         if self.endPoint < 0:
    #             raise Exception("Error: endPoint can't be less than zero. - t" + str(self.tID))

    #     # Is this a valid error? Example when type = 'SET' but quant and freq are none:
    #     #   in 'VOA19980303.1600.2745.tml' tID is 't120'
    #     # if self.type == 'SET' and (self.quant is None and self.freq is None):
    #     #     raise Exception("Error: quant or freq can't be None if type is SET - t" + str(self.tID))
    #     if self.quant is not None:
    #         if "quant" in self.quant:
    #             self.quant = self.quant.split("=")[1]
    #         if '"' in self.quant:
    #             self.quant = self.quant.replace('"', '')

    def get_id_str(self):
        return "t" + str(self.tID)

    def get_anchor_id_str(self):
        return "t" + str(self.anchorID)

    def get_begin_point_str(self):
        return "t" + str(self.beginPoint)

    def get_end_point_str(self):
        return "t" + str(self.endPoint)

    def __hash__(self):
        return hash(self.tID + hash(self.type) + hash(self.value))

    def to_json(self):
        t_type = self.type if self.type is not 'NONE' else "NULL"
        # mod = self.mod if self.mod is not 'NONE' else "NULL"
        document_function = self.documentFunction if self.documentFunction is not 'NONE' else "NULL"
        anchor_id = self.get_anchor_id_str() if self.anchorID is not None else "NULL"
        quant = self.quant if self.quant is not None else "NULL"
        freq = self.freq if self.freq is not None else "NULL"
        begin_point = self.get_begin_point_str() if self.beginPoint is not None else "NULL"
        end_point = self.get_end_point_str() if self.endPoint is not None else "NULL"

        ret = {"id":self.get_id_str(),
               "type":t_type,
               "value":self.value,
               "temporalFunction":self.temporalFunction,
               "anchorID":anchor_id,
               "beginPoint":begin_point,
               "endPoint":end_point,
               "quantity":quant,
               "frequency":freq,
               "functionInDocument":document_function,
               "stem": self.stem,
               "timex_start": self.timex_start,
                "timex_end": self.timex_end
                }
        return ret
    
               # f' "phrase":"{self.phrase}",' \
               # f' "mod":"{mod,' \
               



