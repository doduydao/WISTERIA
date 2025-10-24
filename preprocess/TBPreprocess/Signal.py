# Written by: kfont015
# Last updated: vfern124

from dataclasses import dataclass


@dataclass
class Signal:
    signal_id: int
    signal_string: str
    start: int
    end: int

    def __post_init__(self):
        if self.signal_id < 1:
            raise Exception("Signal ID may not be less than 1")

        if len(self.signal_string) == 0:
            raise Exception("Signal string can not be null, empty string or all whitespace")

    def get_id_str(self):
        return "sid{}".format(self.signal_id)

    def to_json(self):
        return {"id": self.get_id_str(),
                "signalString": self.signal_string,
                "start": self.start,
                "end": self.end
                }

    # Custom __str__
    def __str__(self):
        return "Signal: {{id = {}, String = {}}}".format(self.signal_id, self.signal_string)
