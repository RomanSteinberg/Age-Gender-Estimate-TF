class AbstractNet:
    def __init__(self):
        self.bottleneck_scope = None
        self.age_num_classes = 101
        self.gender_num_classes = 2

    def get_tail(self):
        pass

    def get_head(self, net):
        pass

    def build_model(self):
        net = self.get_tail()
        net = self.get_head(net)
        return net