import nltk
import pickle
import argparse         #argparse是一个Python模块：命令行选项、参数和子命令解析器
from collections import Counter
from pycocotools.coco import COCO            #pycocotools是微软提供的导入coco信息的库

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}          
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):               
        if not word in self.word2idx:
            self.word2idx[word] = self.idx          #建立 word2idx索引、idx2word索引
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']         
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)                #构建coco对象， coco = pycocotools.coco.COCO(json_file)
    counter = Counter()              #统计词频， 词：频数
    ids = coco.anns.keys()          #？？？
    for i, id in enumerate(ids):     #i id ？？？
        caption = str(coco.anns[id]['caption'])    #获取caption
        tokens = nltk.tokenize.word_tokenize(caption.lower())    #对caption分词
        counter.update(tokens)       #将tokens更新加入到counter中

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))   #百分多少的caption单词已被Tokenized

    # If the word frequency is less than 'threshold', then the word is discarded.   频率小于阙值的单词被遗弃
    words = [word for word, cnt in counter.items() if cnt >= threshold]    #cnt？？，得到大于阙值的单词

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')      #将这4各个词加入字典
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)     #将word加入字典
    return vocab

def main(args):   #参数：caption路径，保存路径，阙值
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)    #args.caption_path、args.threshold？？？？参数？
    vocab_path = args.vocab_path   #定义一个路径，后面将字典保存到这
    with open(vocab_path, 'wb') as f:  #将字典vocab保存到vocab_path文件
        pickle.dump(vocab, f)      #pickle.dump(obj, file, [,protocol])，序列化对象，将对象obj保存到文件file中去。参数protocol是序列化模式，默认是0
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':   #当直接运行build_vocab.py时，下面代码运行；当import时，不运行
    parser = argparse.ArgumentParser()        #创建一个 ArgumentParser 对象；ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息
    parser.add_argument('--caption_path', type=str,     #给一个 ArgumentParse r对象添加程序参数信息   #命名；
                        default='data/annotations/captions_train2014.json',   #当参数未在命令行中出现时使用的值
                        help='path for train annotation file')                #一个此选项作用的简单描述
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()                 #用 parse_args() 方法解析参数
    main(args)
