import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F


class RCNN_Text(nn.Module):

    def __init__(self, args):
        super(RCNN_Text, self).__init__()
        self.args = args

        n_tokens = args.embed_num
        embedding_dim = args.embed_dim
        hidden_dim_1 = 200
        hidden_dim_2 = 100
        n_classes = args.class_num
        Ci = 1
        Co = args.kernel_num

        # We add an additional row of zeros to the embeddings matrix to represent unseen words and the NULL token.
        # embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
        # embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

        # self.document = torch.autograd.Variable(shape=(None,), dtype="int32")
        # self.left_context = torch.autograd.Variable(shape=(None,), dtype="int32")
        # self.right_context = torch.autograd.Variable(shape=(None,), dtype="int32")

        self.embed = nn.Embedding(n_tokens + 1, embedding_dim)

        # self.forward_pass = nn.RNN(input_size=args.batch_size, hidden_size=hidden_dim_1)
        # self.backward_pass = nn.RNN(input_size=args.batch_size, hidden_size=hidden_dim_1, go_backwards=True)

        self.rnn= nn.RNN(input_size=args.batch_size, hidden_size=hidden_dim_1, bidirectional=True)
        self.convolution = nn.Conv2d(Ci, Co, (3, embedding_dim))
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(3 * Co, n_classes)

    def conv_and_pool(x, conv):
        c = conv(x)
        x = F.relu(c).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


    def forward(self, x):
        x = self.embed(x)  # (N,W,D)

        # doc_embedding = self.embed(x)
        # l_embedding = self.embed(self.left_context)
        # r_embedding = self.embed(self.right_context)

        # forward = self.forward_pass(l_embedding)  # See equation (1).
        # backward = self.backward_pass(r_embedding)  # See equation (2).
        # together = merge([forward, doc_embedding, backward], mode="concat", concat_axis=2)  # See equation (3).


        # semantic = TimeDistributed(Dense(hidden_dim_2, activation="tanh"))(together)  # See equation (4).

        # Keras provides its own max-pooling layers, but they cannot handle variable length input
        # (as far as I can tell). As a result, I define my own max-pooling layer here.
        # pool_rnn = Lambda(lambda x: backend.max(x, axis=1), output_shape=(hidden_dim_2,))(semantic)  # See equation (5).

        # output = Dense(n_classes, input_dim=hidden_dim_2, activation="softmax")(
        #    pool_rnn)  # See equations (6) and (7).

        # model = Model(input=[document, left_context, right_context], output=output)
        # model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])

        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x1 = RCNN_Text.conv_and_pool(x, self.convolution)  # (N,Co)
        x = torch.cat(x1, 1)

        x = self.dropout(x)  # (N,len(Ks)*Co)
        return self.fc1(x)  # (N,C)
