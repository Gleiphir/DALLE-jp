import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
import io

from tqdm import tqdm

import random

from torch.utils.data.dataset import Dataset

dataset_path = r"merged.txt"

class token_dataset:
    def __init__(self,filepath):
        self.ja_tokenizer = get_tokenizer('spacy', language='ja_core_news_sm')
        self.dataset_fp = filepath

        self.jp_vocab = self.build_vocab(filepath, self.ja_tokenizer)

        self.data = {}
        self.read_fp()

    def read_fp(self):
        with open(self.dataset_fp,encoding='utf-8') as Fp:
            for ln in tqdm(Fp.readlines()):
                idx_, text = ln.split('|')
                idx = int(idx_)
                if not idx in self.data:
                    self.data[idx] = []
                self.data[idx].append(text)

    def build_vocab(self,filepath, tokenizer):
        counter = Counter()
        with io.open(filepath, encoding="utf8") as f:
          for string_ in f:
            counter.update(tokenizer(string_))
        return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    def getRand(self, idx):
        raw = random.choice(self.data[idx])
        return torch.tensor([self.jp_vocab[token] for token in self.ja_tokenizer(raw)],
                                      dtype=torch.long)

    def getAll(self,idx):
        return [[self.jp_vocab[token] for token in self.ja_tokenizer(s)] for s in self.data[idx] ]

    def maxLen(self):
        Max = 0
        for key in self.data:
            for s in self.data[key]:
                if len(self.ja_tokenizer(s)) > Max:
                    Max = len(self.ja_tokenizer(s))
        return Max
