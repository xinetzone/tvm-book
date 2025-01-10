class ImageNet1kAttr(object):
    def __init__(self):
        self.num_class = 1000
        self.synset = []
        self.classes = []
        self.classes_long = []

        for syn, cls, cls_l in _ILSVRC2012_Attr:
            self.synset.append(syn)
            self.classes.append(cls)
            self.classes_long.append(cls_l)