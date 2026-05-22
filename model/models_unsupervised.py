import torch.nn as nn
from model.model_fast_1 import FastDVDnet


class UnsupervisedDVDnet(nn.Module):
    def __init__(self, num_input_frames=5):
        super(UnsupervisedDVDnet, self).__init__()
        self.num_input_frames = num_input_frames
        self.fastdvdnet = FastDVDnet(num_input_frames=num_input_frames)


    def forward(self, x ,S ): #,S
        y = self.fastdvdnet(x,S) #,S
        del x
        return y