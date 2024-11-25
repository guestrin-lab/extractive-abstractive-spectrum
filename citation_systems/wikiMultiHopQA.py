from dataset import Dataset 
import random
import pandas as pd
import gzip
import json
import sys
from utils import standardize_quotation_marks

class WikiMultiHopQA(Dataset): 

    def __init__(self, seed):
        print('Loading data...')
        data_file = 'data/wikiMultiHop/train.json'
        with open(data_file, mode="rt") as f:
            self.data = json.load(f)
        random.seed(seed)
        self.data = self.data[40:] # Exclude the first 40, which were used to draw few shot examples
        print('Shuffling...')
        random.shuffle(self.data)
        print('Shuffled!')
    
    def __getitem__(self, i):
        question = self.data[i]['question']
        sources = []
        titles = []
        context = self.data[i]['context']
        supporting_facts = self.data[i]['supporting_facts']
        supporting_facts_tag_list = [x[0] for x in supporting_facts]
        for j in range(len(context)):
            curr_tag = context[j][0]
            if (curr_tag in supporting_facts_tag_list):
                constructed_source = ''
                context_item = context[j]
                for sentence in context_item[1]:
                    constructed_source += sentence + ' '
                constructed_source = standardize_quotation_marks(constructed_source)
                sources.append(constructed_source)
                titles.append(context_item[0])
        return {'question': question,
                'id': i,
                'gold_reference': sources, # this is the minimum spanning gold reference
                'urls': titles
                }

    def __len__(self):
        return len(self.data)
    
    def print_item(self, i):
        item = self.__getitem__(i)
        print(item['question'])
        for x in item['gold_reference']:
            print()
            print(x)
