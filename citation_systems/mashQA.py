from dataset import Dataset 
import json
import pandas as pd
import os.path
import pickle
import random
import numpy as np
import copy

class MashQA(Dataset): 

    def __init__(self, seed):
        print('Loading data...')
        results_fp = 'data/mashqa_dataset/formatted_mash_data.pkl'
        if (os.path.exists(results_fp)):
            with open(results_fp, 'rb') as f:
                self.data = pickle.load(f)
        else:
            file_name = 'data/mashqa_dataset/train_webmd_squad_v2_full.json'
            with open(file_name, 'r') as f:
                self.data_og = json.load(f)
            random.seed(seed)
            self.data = []
            for i in range(len(self.data_og['data'])):
                curr_instance = {}
                curr_instance['urls'] = [self.data_og['data'][i]['title']]
                if ('pets.webmd.com' in curr_instance['urls']):
                    continue
                curr_instance['gold_reference'] = [self.data_og['data'][i]['paragraphs'][0]['context']]
                num_qas = len(self.data_og['data'][i]['paragraphs'][0]['qas'])

                # Get information-seeking questions
                information_seeking_questions = []
                for j in range(num_qas):
                    question = self.data_og['data'][i]['paragraphs'][0]['qas'][j]['question']
                    is_impossible = self.data_og['data'][i]['paragraphs'][0]['qas'][j]['is_impossible']
                    if (('I ' in question) or ('you ' in question)):
                        if (is_impossible == False):
                            information_seeking_questions.append(question)
                if (len(information_seeking_questions) == 0):
                    continue
                # Select a random information-seeking question for each article 
                random_qa_idx = np.random.randint(0, len(information_seeking_questions))
                question = information_seeking_questions[random_qa_idx]
                curr_instance['question'] = question
                self.data.append(curr_instance)

            for i in range(len(self.data)):
                self.data[i]['id'] = i

            with open(results_fp, 'wb') as f:
                pickle.dump(self.data, f)
            
    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)
    
    def print_item(self, i):
        item = self.__getitem__(i)
        print(item['id'])
        print(item['question'])
        print(item['urls'])
        print(item['gold_reference'])
        return
