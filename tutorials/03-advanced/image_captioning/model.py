import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()   
        resnet = models.resnet152(pretrained=True)                  #resNet
        modules = list(resnet.children())[:-1]                      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)  #nn.Sequential类似于Keras中的贯序模型，它是Module的子类，在构建数个网络层之后会自动调用forward()方法，从而有网络模型生成。
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)   #（*）会把接收到的参数形成一个元组，而（**）则会把接收到的参数存入一个字典
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)   
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)     #维度  [batch,打平]
        features = self.bn(self.linear(features))             #维度  [batch,embed_size]，即lstm的x0
        return features


class DecoderRNN(nn.Module):  #input数据(seq_len,batch_size,input_size)=(,,embed_size)：(每个句子长度，几个句子，每个单词用几维度表示),batch放最前
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20): 
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)                          #(字典单词数，维度)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) #(input_size,hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)   #captions是单词id号列表，[句子数量，单词数量，embed_size]，同上面input数据
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)   # [batch,1,embed_size]+[batch,seq_len,embed_size] 将x0放在xt之前
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)   #pack_padded_sequence填充,和data_loader里的补长操作重复？？？lengths作用？？？
        hiddens, _ = self.lstm(packed)     #hiddens是out，维度[batch,seq_len,h dim]
        outputs = self.linear(hiddens[0])  # hidden维度[1,2,3] , hidden[0]维度[2,3]。 outputs维度[batch,seq_len,vocab_size],变成单词embed长度
        return outputs
    
    def sample(self, features, states=None):    #greedy search，上面是直接用groundtruth当做生成下一个单词的条件
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)                           # inputs:[batch,1,embed_size]
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)，在vocab_size上取最大值，即选择一个概率最大单词
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)，将这个单词当做下个输入
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
