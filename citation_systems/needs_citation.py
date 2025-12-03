import vertexai
import requests
from mashQA import MashQA
from wikiMultiHopQA import WikiMultiHopQA
from openai import OpenAI
import os
import copy 
from naturalQuestions import NaturalQuestions
from retrieval import GoogleDPRRetrieval, PostHocRetrieval
from utils import *
import random
import argparse
import json
from abstraction_metrics import *
import time
import ast

from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Tool,
    grounding,
)
from google.cloud import discoveryengine_v1alpha as discoveryengine

project_id = '' # TODO Google project ID
client = discoveryengine.GroundedGenerationServiceClient()
grounding_config = client.grounding_config_path(
        project=project_id,
        location="global",
        grounding_config="default_grounding_config",
    )

def get_for_all(baselines=True):
    for data_str in ['mash', 'eli3', 'nq', 'mh']: 
        mturk_results = get_one_mturk_result(data_str, baselines=baselines)
        mturk_results = get_for_df(mturk_results)

        # save new file
        if (baselines):
            save_path = 'mturk_results/'+data_str+'_baseline_mturk_with_needs_citation_labels.csv'
        else:
            save_path = 'mturk_results/'+data_str+'_mturk_with_needs_citation_labels.csv'
        if (os.path.isfile(save_path)):
            print('Did not save file; it would overwrite the file '+save_path+' that potentially contains annotations. Remove or rename that file and rerun this script.')
        else:
            mturk_results.to_csv(save_path)
        print('Saved to '+save_path)


def get_for_df(df):
    num_calls = 0
    new_col = []
    for i in range(len(df)):
        if (df['op'].iloc[i] == 'Snippet'):
            sentences_need_citation = []
        else:
            sentences = ast.literal_eval(df.iloc[i]['Sent'])
            sentences_need_citation = []
            for sentence in sentences:
                try:
                    request = discoveryengine.CheckGroundingRequest(
                        grounding_config=grounding_config,
                        answer_candidate=sentence,
                        facts=[discoveryengine.GroundingFact(
                                        fact_text=("None"),
                                        attributes={"uri": "None"},
                                        )],
                        grounding_spec=discoveryengine.CheckGroundingSpec(citation_threshold=.1),
                    )
                    needs_citation_response = client.check_grounding(request=request)
                    num_calls += 1
                    if (num_calls % 100 == 0):
                        time.sleep(45)

                    sentence_needs_citation = False
                    for j in range(len(needs_citation_response.claims)):
                        needs_citation = needs_citation_response.claims[j].grounding_check_required
                        # If any part of the sentence needs citation, then mark the sentence as needing citation
                        sentence_needs_citation = sentence_needs_citation or needs_citation
                except:
                    print(f"Error at index {i} ID: {df['query_id'].iloc[i]} OP: {df['op'].iloc[i]}. Skipping...")
                    print(f"Index: {i} OP: {df['op'].iloc[i]} ID: {df['query_id'].iloc[i]}")
                    sentence_needs_citation = True # Assume it needs citation if there's an error

                sentences_need_citation.append(sentence_needs_citation)
                print(f"{sentence_needs_citation}:   {sentence}")
                
        new_col.append(sentences_need_citation)
    df['Sentences Need Citation'] = new_col
    return df


def main():
    get_for_all(baselines=True)

if __name__ == "__main__":
    main()



