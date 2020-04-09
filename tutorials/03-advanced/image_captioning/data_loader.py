import torch
import torchvision.transforms as transforms
import torch.utils.data as data            #torch.utils.data主要包括以下三个类：Dataset / sampler.Sampler / DataLoader
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):    #coco数据集，Dataset创建数据集，      参数：路径，字典，transform
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""          
    def __init__(self, root, json, vocab, transform=None)
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""          
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.                #图片目录
            json: coco annotation file path.      #路径
            vocab: vocabulary wrapper.            #字典
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())    
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):                 #（index标签，caption，图片）多对1
        """Returns one data pair (image and caption)."""
        coco = self.coco                          #构建coco对象， coco = COCO(json_file)
        vocab = self.vocab
        ann_id = self.ids[index]                  #groundtruth的id
        caption = coco.anns[ann_id]['caption']    #groundtruth
        img_id = coco.anns[ann_id]['image_id']    #图片id
        path = coco.loadImgs(img_id)[0]['file_name']    #coco.loadImgs：根据id号，导入对应的图像信息
        image = Image.open(os.path.join(self.root, path)).convert('RGB')      #os.path.join：路径拼接； image的路径
        
        if self.transform is not None:
            image = self.transform(image)   #图片处理

        # Convert caption (string) to word ids.                      caption → target
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())   #nltk.word_tokenize(text)分词，nltk.FreqDist(words)词频统计
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))                               #将caption头尾加上标志 
        target = torch.Tensor(caption)                               #caption → target
        return image, target                                         #image,target   图片，单词序列

    def __len__(self):                #有__len__(self)函数来获取数据集的长度.
        return len(self.ids)            


def collate_fn(data):     #参数：data，应该是上个返回的image，target
    """Creates mini-batch tensors from the list of tuples (image, caption).
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).          #图片(batch_size, 3, 256, 256)
        targets: torch tensor of shape (batch_size, padded_length).       #target(batch_size, padded_length)，扩长
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)      #list.sort(cmp=None, key=None, reverse=False ) cmp可选参数、key用来进行比较的元素、默认False升序，按target长度排序
    images, captions = zip(*data)                         #zip() 打包为元组的列表，zip(*) 将元组解压为列表

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)                            #图片叠加，到4维，在0位上

    # Merge captions (from tuple of 1D tensor to 2D tensor).   #caption叠加，到2维
    lengths = [len(cap) for cap in captions]                   #每个caption的长度
    targets = torch.zeros(len(captions), max(lengths)).long()    #加上batch的caption，先是0，后补数据
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        #将caption前面有数据的补在targets中，扩展的为0
    return images, targets, lengths                           #将长度不等的caption扩展为长度相同的？

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).                         #(batch_size, 3, 224, 224)
    # captions: a tensor of shape (batch_size, padded_length).                     #(batch_size, padded_length) 扩展的长度，1位是字符串形式
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,              #dataset (Dataset): 加载数据的数据集
                                              batch_size=batch_size,     #batch_size (int, optional): 每批加载多少个样本
                                              shuffle=shuffle,           #shuffle (bool, optional): 设置为“真”时,在每个epoch对数据打乱.（默认：False）
                                              num_workers=num_workers,   #num_workers (int, optional): 用于加载数据的子进程数。0表示数据将在主进程中加载​​。（默认：0）
                                              collate_fn=collate_fn)     #collate_fn (callable, optional): 合并样本列表以形成一个 mini-batch.  #　callable可调用对象
    return data_loader
