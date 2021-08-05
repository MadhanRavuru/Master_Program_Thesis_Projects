
""" Dataset parameters """
class Params():
    def __init__(self):
        # network structure parameters
        self.model = 'MobileNetv2_DeepLabv3'
        self.dataset = 'cityscapes'
        self.s = [2, 1, 2, 2, 2, 1, 1]  # stride of each conv stage
        self.t = [1, 1, 6, 6, 6, 6, 6]  # expansion factor t
        self.n = [1, 1, 2, 3, 4, 3, 3]  # number of repeat time
        self.c = [32, 16, 24, 32, 64, 96, 160]  # output channel of each conv stage
        self.output_stride = 16
        self.multi_grid = (1, 2, 4)
        self.aspp = (6, 12, 18)
        self.down_sample_rate = 32  # classic down sample rate

        # dataset parameters
        self.rescale_size = 600
        self.image_size = 256
        self.num_class = 312  # 20 classes for training