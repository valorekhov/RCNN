"""cnn: a PyTorch implementation of vanilla CNN. """

import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yandi Xia"

CONFIG = {
    "filter_sizes": [3],
    "num_filters": 250,
    "hid_sizes": [250],
    "dropout_switches": [True]
}


class CnnNet(nn.Module):
    """A vanilla CNN model"""

    def __init__(self, config):
        self.config = config
        for arg in CONFIG:
            self.__setattr__(arg, CONFIG[arg])
        assert len(self.hid_sizes) == len(self.dropout_switches)
        super(CnnNet, self).__init__()
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.init_embedding()
        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             nn.Conv2d(in_channels=1,
                                       out_channels=self.num_filters,
                                       kernel_size=(filter_size, config.d_embed)))
            self.encoders.append(self.__getattr__(enc_attr_name))
        self.hid_layers = []
        ins = len(self.filter_sizes) * self.num_filters
        for i, hid_size in enumerate(self.hid_sizes):
            hid_attr_name = "hid_layer_%d" % i
            self.__setattr__(hid_attr_name, nn.Linear(ins, hid_size))
            self.hid_layers.append(self.__getattr__(hid_attr_name))
            ins = hid_size
        self.logistic = nn.Linear(ins, config.d_out)

    def forward(self, batch):
        """
        :param x:
                input x is in size of [N, C, H, W]
                N: batch size
                C: number of channel, in text case, this is 1
                H: height, in text case, this is the length of the text
                W: width, in text case, this is the dimension of the embedding
        :return:
                a tensor [N, L], where L is the number of classes
        """
        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3
        # lookup table output size [N, H, W=emb_dim]
        print(batch.text.size())
        x = self.embed(batch.text)
        if self.config.fix_emb:
            x = torch.autograd.Variable(x.data)
            if torch.cuda.is_available():
                x = x.cuda()
        # expand x to [N, 1, H, W=emb_dim]
        x = x.unsqueeze(c_idx)
        enc_outs = []
        for encoder in self.encoders:
            enc_ = F.relu(encoder(x))
            k_h = enc_.size()[h_idx]
            k_w = 1
            enc_ = F.max_pool2d(enc_, kernel_size=(k_h, k_w))
            enc_ = enc_.squeeze(w_idx)
            enc_ = enc_.squeeze(h_idx)
            enc_outs.append(enc_)
        # each of enc_outs size [N, C]
        encoding = torch.cat(enc_outs, 1)
        hid_in = encoding
        for hid_layer, do_dropout in zip(self.hid_layers, self.dropout_switches):
            hid_out = F.relu(hid_layer(hid_in))
            if do_dropout:
                hid_out = F.dropout(hid_out, training=self.training)
            hid_in = hid_out
        pred_prob = F.log_softmax(self.logistic(hid_in))
        return pred_prob

    def init_embedding(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
