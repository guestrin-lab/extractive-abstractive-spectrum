from dataset import Dataset 
import random
import pandas as pd
import gzip
import json
import sys
import pickle
import os
from text_utils import simplify_nq_example

class NaturalQuestions(Dataset): 

    def __init__(self, seed=0):
        print('Loading data...')
        random.seed(seed)
        processed_nq_fp = "data/natural-questions/v1.0/dev/processed_nq.pkl"
        if (not os.path.exists(processed_nq_fp)):
            data_file = "data/v1.0/dev/nq-dev-00.jsonl.gz"
            with gzip.open(data_file, mode="rt") as f:
                self.data = [json.loads(line) for line in f]
            self.processed_data = []
            processed_data_i = 0
            for i in range(len(self.data)):
                curr_item = self.get_legacy_item(i)
                if (curr_item != None):
                    curr_item['id'] = processed_data_i
                    self.processed_data.append(curr_item)
                    processed_data_i += 1
            with open(processed_nq_fp, 'wb') as f:
                pickle.dump(self.processed_data, f)
        else:
            with open(processed_nq_fp, 'rb') as f:
                self.processed_data = pickle.load(f)
        print('Number of instances in NQ dataset: '+str(len(self.processed_data)))

    def get_one_gold_reference(self, simplified_instance):
        start_idxs = [simplified_instance['annotations'][j]['long_answer']['start_token'] for j in range(5)] # collect from all 5 annotators
        end_idxs = [simplified_instance['annotations'][j]['long_answer']['end_token'] for j in range(5)] 
        gold_start_idx = -1
        j=0
        while ((gold_start_idx==-1) and (j < len(start_idxs))):
            gold_start_idx = start_idxs[j]
            gold_end_idx = end_idxs[j]

            content_tag = simplified_instance["document_text"].split(" ")[gold_start_idx]
            if (content_tag != '<P>'):
                gold_start_idx = -1
                gold_end_idx = -1
            j+=1
        if (gold_start_idx == -1):
            return None
        references = [" ".join(simplified_instance["document_text"].split(" ")[gold_start_idx:gold_end_idx])]
        return references

    def is_P_tag(self, ref):
        return ref.split(" ")[0] == '<P>'
    
    def get_full_reference(self, simplified_instance):
        start_idxs = [simplified_instance['annotations'][j]['long_answer']['start_token'] for j in range(5)] # collect from all 5 annotators
        end_idxs = [simplified_instance['annotations'][j]['long_answer']['end_token'] for j in range(5)]
        gold_start_idx = -1
        j=0
        while ((gold_start_idx==-1) and (j < len(start_idxs))):
            gold_start_idx = start_idxs[j]
            gold_end_idx = end_idxs[j]

            content_tag = simplified_instance["document_text"].split(" ")[gold_start_idx]
            if (content_tag != '<P>'):
                gold_start_idx = -1
                gold_end_idx = -1
                j+=1
        if (gold_start_idx == -1):
            return None                                                                                                                                                       
        total_words = 1000
        text_ls = simplified_instance["document_text"].split(" ")
        start_distance = max(0, int(gold_start_idx-total_words//2))
        total_words -= (gold_start_idx - start_distance)
        end_distance = min(int(gold_end_idx+total_words), len(text_ls))
        references = [" ".join(text_ls[start_distance:end_distance])]
        return references

    def fix_punctuation_spacing(self, references):
        j=0
        for reference in references:
            list_reference = list(reference)
            i = 0
            while (i < len(list_reference)):
                curr_char = list_reference[i]
                if ((curr_char==',') or (curr_char=='.') 
                                     or (curr_char=='\'') 
                                     or (curr_char==')')
                                     or (curr_char==':')
                                     or (curr_char==';')
                                     or (curr_char=='-')
                                     or (curr_char=='%')):
                    if ((i != 0) and list_reference[i-1]==' '):
                        del list_reference[i-1]
                        i -= 1
                if ((curr_char == '(') or (curr_char == '-')):
                    if ((i != len(list_reference)-1) and list_reference[i+1]==' '):
                        del list_reference[i+1]
                if ((curr_char == '\'')):
                    if ((i != len(list_reference)-1) and list_reference[i+1]=='\''):
                        del list_reference[i+1]
                i += 1
            fixed_reference = "".join(list_reference)
            references[j] = fixed_reference
            j+=1
        return references
    
    def __getitem__(self, i):
        # Long QA
        long_instances = [0, 4, 9, 12, 17, 21]

        # Abstained
        abstained_instances = [3, 6, 8, 16, 20]
        all = abstained_instances+long_instances
        return self.processed_data[i]

    def get_legacy_item(self, i):
        simplified_instance = simplify_nq_example(self.data[i])
        gold_reference = self.get_one_gold_reference(simplified_instance)
        
        if (gold_reference == None):
            return None

        gold_reference = self.fix_punctuation_spacing(gold_reference)

        if (gold_reference == None):
            return None

        full_reference = self.get_full_reference(simplified_instance)
        full_reference = self.fix_punctuation_spacing(full_reference)
        return {'question': simplified_instance['question_text'], 
                'gold_reference': gold_reference, 
                'id':simplified_instance['example_id'],
                'urls': ['wikipedia'],
                'full_reference': full_reference
                }

    def __len__(self):
        return len(self.processed_data)