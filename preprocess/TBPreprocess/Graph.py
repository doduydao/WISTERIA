import TimeMLParser

class Graph:
    def __init__(self, filepath):
        """
        :param str filepath: the path to a TimeML annotated file to be parsed and analyzed.
        """

        self.nodes = {}
        self.links = {}
        self.signals = {}
        self.events = {}
        self.raw_text = None
        self.type = "Overview"
        self.consistency = None
        self.total_time_points = 0

        if filepath is not None:
            if filepath:
                with open(filepath) as file:
                    self.time_ml_data = file.read()

            self.metadata, self.raw_text, links, nodes, signals, events, timexs = TimeMLParser.parse(self.time_ml_data)

            if len(nodes) == 0:
                raise Exception("No Nodes found in Text")

            self.timexs = {timex.get_id_str(): timex for timex in timexs}
            self.events = {event.get_id_str(): event for event in events}
            self.links = {link.get_id_str(): link for link in links}
            self.nodes = {node.get_id_str(): node for node in nodes}
            if signals is not None:
                self.signals = {signal.get_id_str(): signal for signal in signals}

        else:
            raise Exception("Must supply either a TimeML Annotated File or a TimeML annotated string")

    def to_json(self):
        output = dict()
        output["text"] = self.raw_text
        output["nodes"] = [eval(node.to_json()) for node in self.nodes.values()]
        output["links"] = [eval(link.to_json()) for link in self.links.values()]
        output["events"] = [eval(event.to_json()) for event in self.events.values()]
        output["timexs"] = [eval(timex.to_json()) for timex in self.timexs.values()]
        output["signals"] = [eval(signal.to_json()) for signal in self.signals.values()]
        output["isConsistent"] = True
        output["inconsistentSubGraphs"] = []
        return output

    def __repr__(self):
        ret = "Nodes: "
        for node in self.nodes.values():
            ret += node.to_json()
            ret += ", "

        ret = ret[:-2] + "\n"

        ret += "Links: "
        for link in self.links.values():
            ret += link.to_json()
            ret += ", "

        ret = ret[:-2] + "\n "
        return ret

    def to_string(self):

        ret = "Nodes: "
        for node in self.nodes.values():
            ret += node.to_json()
            ret += ", "

        ret = ret[:-2] + "\n"

        ret += "Links: "
        for link in self.links.values():
            ret += link.to_json()
            ret += ", "

        ret = ret[:-2] + "\n "
        return ret

    def get_data_json(self):
        output = dict()
        output["events"] = [eval(event.to_json()) for event in self.events.values()]
        output["timexs"] = [eval(timex.to_json()) for timex in self.timexs.values()]
        output["signals"] = [eval(signal.to_json()) for signal in self.signals.values()]
        output["links"] = [eval(link.to_json()) for link in self.links.values()]
        return output
