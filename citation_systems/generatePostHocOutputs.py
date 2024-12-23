# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini
# Search entry point display requirements: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/grounding-search-entry-points
import vertexai
import requests
import ast
import pandas as pd
from mashQA import MashQA
from wikiMultiHopQA import WikiMultiHopQA
from openai import OpenAI
import os
import copy 
from naturalQuestions import NaturalQuestions
from retrieval import PostHocRetrieval
from utils import *
import random
import argparse
import json
from instructions import nq_baseline_instruction_str, mh_baseline_instruction_str, mash_baseline_instruction_str

from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Tool,
    grounding,
)
from google.cloud import discoveryengine_v1alpha as discoveryengine

COLORS = {0:'\033[92m', 1:'\033[96m', 2:'\033[95m', 3:'\033[1;31;60m', 4:'\033[102m', 5:'\033[1;35;40m', 6:'\033[0;30;47m', 7:'\033[0;33;47m', 8:'\033[0;34;47m', 9:'\033[0;31;47m', 10:'\033[0m', 11:'\033[1m'}
BACKBONE_MODEL = OpenAI()

def remove_escape_sequences(text):
    string_literal_text = repr(text)
    while ('\\x' in string_literal_text):
        start_idx = string_literal_text.find('\\x')
        sequence_length = min(4, len(string_literal_text)-start_idx)
        escape_sequence = string_literal_text[start_idx:start_idx+sequence_length]
        string_literal_text = string_literal_text.replace(escape_sequence, ' ')
    return ast.literal_eval(string_literal_text)

def generate_post_hoc_cited_answer(instance, retriever, use_gold, citation_threshold, args):
    project_id = "" # TODO Google project ID
    client = discoveryengine.GroundedGenerationServiceClient()
    grounding_config = client.grounding_config_path(
            project=project_id,
            location="global",
            grounding_config="default_grounding_config",
        )
        
    if ((args.data == 'nq') or (args.data == 'eli5_nq')):
        baseline_instruction_str = nq_baseline_instruction_str
    elif (args.data == 'multihop'):
        baseline_instruction_str = mh_baseline_instruction_str
    elif (args.data == 'mash'):
        baseline_instruction_str = mash_baseline_instruction_str
    else:
        print('Need to implement baseline_instruction_str for dataset!')
        return
    abstention_type = 'No Failure'
    if (not retriever.is_cached(instance['question'])):
        prompt = baseline_instruction_str+instance['question']
        answer_candidate, details_dict = generate_from_model(BACKBONE_MODEL, prompt)
        request = discoveryengine.CheckGroundingRequest(
            grounding_config=grounding_config,
            answer_candidate=answer_candidate,
            facts=[discoveryengine.GroundingFact(
                            fact_text=("None"),
                            attributes={"uri": "None"},
                            )],
            grounding_spec=discoveryengine.CheckGroundingSpec(citation_threshold=citation_threshold),
        )
        response = client.check_grounding(request=request)

        # get the sentences of the response to the query that require citation
        answer_candidate_sentences = []
        for i in range(len(response.claims)):
            needs_citation = response.claims[i].grounding_check_required
            if (needs_citation):
                answer_candidate_sentences.append(response.claims[i].claim_text)
    else:
        answer_candidate_sentences = [] # placeholder; retrieving from the cache doesn't require this
        answer_candidate = '' # placeholder; retrieving from the cache doesn't require this and will provide this

    # Retrieve sources for the sentences that require citation, and also cache the full response (if not done already)
    try:
        answer_candidate, sources, urls = retriever.retrieve(instance, answer_candidate_sentences, answer_candidate, use_gold) 
    except:
        abstention_type = 'Retrieval Failure'
        print('Retrieval Failure!!!')
        # save the results if not debugging
        if (not args.debug):
            results_to_save = {"ID": int(instance['id']), 
                            "All Sources": None,
                            "All URLs": None,
                            "All Sources (cited)": None, 
                            "Used Sources (cited)": None, 
                            "Question": instance['question'], 
                            "Post Hoc Cited Output (cited)": None, 
                            "Post Hoc Cited Output": answer_candidate, 
                            "Post Hoc Cited Sent (cited)": None, 
                            "Post Hoc Cited Sent": None, 
                            "Post Hoc Cited Citation Dict": None,
                            "Sentences Need Citation": None, 
                            "Abstention Type": abstention_type,
                            "Max Tokens": 512, 
                            "Temperature": .2,
                            }
            save_new_line_of_results(results_to_save, args)
        return None, None, None, None, None, None, None
    
    for i in range(len(sources)):
        sources[i] = sources[i].replace('\n', ' ')
        # replace incorrectly decoded hexadecimal escape sequences with ' '
        sources[i] = remove_escape_sequences(sources[i])
        
    # split the chunks up into "facts"
    fact_list = []
    for i in range(len(sources)):
        source = sources[i]
        url = urls[i]
        source_sentences = get_sentences_tokenizer(source)
        for sentence in source_sentences:
            if (len(sentence) > 30):
                curr_fact = discoveryengine.GroundingFact(
                            fact_text=(sentence),
                            attributes={"uri": url},
                            )
                fact_list.append(curr_fact)

    print('Number of facts:', len(fact_list))
    if (len(fact_list) > 200):
        # If there are too many retrieved facts for the Google API, keep only 200. 
        # Take a random sample; there is no ordering of importance across sources because some are retrieved for the query and others are retrieved for sentences in the response.
        fact_list = random.sample(fact_list, 200)
    
    request = discoveryengine.CheckGroundingRequest(
        grounding_config=grounding_config,
        answer_candidate=answer_candidate,
        facts=fact_list,
        grounding_spec=discoveryengine.CheckGroundingSpec(citation_threshold=citation_threshold),
    )
    # for debugging:
    try: 
        response = client.check_grounding(request=request)
    except:
        # this should not occur
        print('API Failure...')
        abstention_type = 'API Failure'
        # save the results if not debugging
        if (not args.debug):
            results_to_save = {"ID": int(instance['id']), 
                            "All Sources": sources,
                            "All URLs": urls,
                            "All Sources (cited)": None, 
                            "Used Sources (cited)": None, 
                            "Question": instance['question'], 
                            "Post Hoc Cited Output (cited)": None, 
                            "Post Hoc Cited Output": answer_candidate, 
                            "Post Hoc Cited Sent (cited)": None, 
                            "Post Hoc Cited Sent": None, 
                            "Post Hoc Cited Citation Dict": None,
                            "Sentences Need Citation": None, 
                            "Abstention Type": abstention_type,
                            "Max Tokens": 512, 
                            "Temperature": .2,
                            }
            save_new_line_of_results(results_to_save, args)
        return None, None, None, None, None, None, None
    answer_is_abstained = is_abstained_gpt4(instance['question'], answer_candidate)
    if (answer_is_abstained == 'True'):
        abstention_type = 'Generation Failure'
    # Create the highlighted and cited response
    # Also, get the highlighted and cited response sentences
    citations_dict = {}
    cited_response = answer_candidate
    unmarked_response_sentences = []
    cited_response_sentences = []
    for i in range(len(response.claims)):
        citation_numbers = response.claims[i].citation_indices # map to the chunks that should be cited for this claim
        citation_number_ls = [x for x in citation_numbers]
        citations_dict[i] = {'citation_numbers': citation_number_ls}
        claim_text = response.claims[i].claim_text
        citation_str = ' '
        for cn in citation_numbers:
            citation_str += COLORS[0]+"["+str(cn)+"]"+COLORS[10]
        if (len(citation_str) == 1):
            citation_str = ''
        cited_claim = claim_text[:-1]+citation_str+claim_text[-1]
        cited_response = cited_response.replace(claim_text, cited_claim)
        unmarked_response_sentences.append(claim_text)
        cited_response_sentences.append(cited_claim)

    print('QUERY')
    print(instance['question'])
    print()
    
    print('RESPONSE')
    print(cited_response)
    print()

    # Create the highlighted and cited sources
    all_cited_sources = copy.deepcopy(sources)
    for i in range(len(response.cited_chunks)):
        chunk = response.cited_chunks[i].chunk_text
        cited_chunk = COLORS[0]+'['+str(i)+'] '+chunk+COLORS[10]
        source_idx = int(response.cited_chunks[i].source)
        url = fact_list[source_idx].attributes['uri']
        candidate_source_idxs = np.where(np.array(urls) == url)[0]
        for candidate_idx in candidate_source_idxs:
            if (chunk in all_cited_sources[candidate_idx]):
                all_cited_sources[candidate_idx] = all_cited_sources[candidate_idx].replace(chunk, cited_chunk)
                break
    
    for i in range(len(all_cited_sources)):
        all_cited_sources[i] = urls[i]+'\n'+all_cited_sources[i]
    
    used_cited_sources = []  
    for i in range(len(all_cited_sources)): 
        curr_cited_source =  all_cited_sources[i]
        if (COLORS[0] in curr_cited_source):
            used_cited_sources.append(curr_cited_source)


    print('USED SOURCES')
    for cs in used_cited_sources:
        print(cs)
    print()

    # save labels of whether each sentence requires citation or not
    sentences_need_citation = []
    for j in range(len(response.claims)):
        sentences_need_citation.append(response.claims[j].grounding_check_required)

    # save the results if not debugging
    if (not args.debug):
        results_to_save = {"ID": int(instance['id']), 
                        "All Sources": sources,
                        "All URLs": urls,
                        "All Sources (cited)": all_cited_sources, 
                        "Used Sources (cited)": used_cited_sources, 
                        "Question": instance['question'], 
                        "Post Hoc Cited Output (cited)": cited_response, 
                        "Post Hoc Cited Output": answer_candidate, 
                        "Post Hoc Cited Sent (cited)": cited_response_sentences, 
                        "Post Hoc Cited Sent": unmarked_response_sentences, 
                        "Post Hoc Cited Citation Dict": citations_dict,
                        "Sentences Need Citation": sentences_need_citation, 
                        "Abstention Type": abstention_type,
                        "Max Tokens": 512, 
                        "Temperature": .2,
                        }
        save_new_line_of_results(results_to_save, args)
    
    return answer_candidate, unmarked_response_sentences, cited_response, cited_response_sentences, all_cited_sources, used_cited_sources, citations_dict

def save_new_line_of_results(results_to_save, args):
    results_fp = 'generation_results/'+args.project_name+'.jsonl'
    if (not os.path.exists(results_fp)):
        with open(results_fp, "w") as f:
            json_string = json.dumps(results_to_save)  # Convert item to JSON string
            f.write(json_string + "\n")  # Write with newline character
    else:
        with open(results_fp, "a") as f:
            json_string = json.dumps(results_to_save)  # Convert item to JSON string
            f.write(json_string + "\n")  # Write with newline character
    print('Saved to '+results_fp)


def save_results_for_sl(args):
    # Load results file as pd df and save as a csv
    results_fp = 'generation_results/'+args.project_name+'.jsonl'
    with open(results_fp, "r") as f:
        results = [json.loads(line) for line in f]
    df = pd.DataFrame(results)
    print()
    # Report failures of the Vertex API citing system (already excluded from the dataset
    print('Post hoc citation API failure rate:', 1-(len(df)/args.n))
    print()
    # Make the columns compatible
    df = df[['ID', 
             'Used Sources (cited)', 
             'Question', 
             'Post Hoc Cited Output (cited)', 
             'Post Hoc Cited Output',
             'Post Hoc Cited Sent (cited)',
             'Post Hoc Cited Sent',
             'Post Hoc Cited Citation Dict',
             'Abstention Type'
             ]]
    df = df.rename(columns={'ID':'ID', 
             'Used Sources (cited)':'Used Sources (cited)', 
             'Question':'Question', 
             'Post Hoc Cited Output (cited)':'Output (cited)', 
             'Post Hoc Cited Output':'Output',
             'Post Hoc Cited Sent (cited)':'Sent (cited)',
             'Post Hoc Cited Sent':'Sent',
             'Post Hoc Cited Citation Dict':'Citation Dict',
             'Abstention Type':'Abstention Type'
             })
    df['op'] = ['Post Hoc']*len(df)

    # save as a csv
    save_path = 'generation_results/'+args.project_name+'_0_'+str(len(df))+'_byQueryPostHoc.csv'
    df.to_csv(save_path)
    print('Saved csv for annotation to '+save_path)

def main(args):
    use_gold = False
    if (args.data == 'multihop'):
        data = WikiMultiHopQA(seed=0) 
        use_gold = True
        citation_threshold = 0.05
    elif (args.data == 'mash'):
        data = MashQA(seed=0) 
        use_gold = True
        citation_threshold = 0.25
    elif (args.data == 'nq'):
        data = NaturalQuestions(seed=0)
        citation_threshold = 0.5
    elif (args.data == 'eli5_nq'):
        data = NaturalQuestions(seed=0)
        citation_threshold = 0.25
    else:
        print('Dataset not yet implemented')
        return

    retriever = PostHocRetrieval(50, 5)
    idx_ls = np.arange(args.start_n, args.start_n+args.n)
    for i in idx_ls:
        instance = data[i]
        instance['id'] = i # overriding all query IDs
        if (args.data == 'eli5_nq'):
            instance['question'] = 'Explain to a third-grader: '+instance['question']
        print('Instance '+str(i))
        generate_post_hoc_cited_answer(instance, retriever, use_gold, citation_threshold, args)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print()
    if (not args.debug):
        save_results_for_sl(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--start_n', type=int, default=0)
    parser.add_argument('--n', type=int, default=60)
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--data', type=str, required=True) # 'nq' or 'multihop' or 'eli5_nq' or 'mash'
    args = parser.parse_args()
    main(args)