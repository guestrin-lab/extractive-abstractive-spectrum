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

from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Tool,
    grounding,
)
from google.cloud import discoveryengine_v1alpha as discoveryengine

GOOGLE_API_KEY = '' # TODO

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
    new_col = []
    for i in range(len(df)):
        if (i%100==0):
            time.sleep(60)
        if (df['op'].iloc[i] == 'Snippet'):
            sentences_need_citation = []
        else:
            response = df.iloc[i]['Output']
            request = discoveryengine.CheckGroundingRequest(
                grounding_config=grounding_config,
                answer_candidate=response,
                facts=[discoveryengine.GroundingFact(
                                fact_text=("None"),
                                attributes={"uri": "None"},
                                )],
                grounding_spec=discoveryengine.CheckGroundingSpec(citation_threshold=.1),
            )
            needs_citation_response = client.check_grounding(request=request)

            sentences_need_citation = []
            for j in range(len(needs_citation_response.claims)):
                needs_citation = needs_citation_response.claims[j].grounding_check_required
                sentences_need_citation.append(needs_citation)
        new_col.append(sentences_need_citation)
    df['Sentences Need Citation'] = new_col
    return df

def get_for_entailed_entries_of_json(df):
    new_col = []
    for i in range(len(df)):
        if (i%100==0):
            time.sleep(60)
        response = df.iloc[i]['Entailed Output']
        try:
            request = discoveryengine.CheckGroundingRequest(
                grounding_config=grounding_config,
                answer_candidate=response,
                facts=[discoveryengine.GroundingFact(
                                fact_text=("None"),
                                attributes={"uri": "None"},
                                )],
                grounding_spec=discoveryengine.CheckGroundingSpec(citation_threshold=.1),
            )
        
            needs_citation_response = client.check_grounding(request=request)            

            sentences_need_citation = []
            for j in range(len(needs_citation_response.claims)):
                needs_citation = needs_citation_response.claims[j].grounding_check_required
                sentences_need_citation.append(needs_citation)
        except:
            sentences_need_citation = []
        new_col.append(sentences_need_citation)
    df['Entailed Sentences Need Citation'] = new_col
    return df



def main():
    get_for_all(baselines=True)

if __name__ == "__main__":
    main()



