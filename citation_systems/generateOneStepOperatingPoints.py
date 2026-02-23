import os
import ast
from openai import OpenAI
from regex import I
# from anthropic import Anthropic
from naturalQuestions import NaturalQuestions
from wikiMultiHopQA import WikiMultiHopQA
from medicalQA import MedicalQA
from emrQA import EmrQA
from mashQA import MashQA
from quoteEval import eval_precision, eval_quote_coverage
import numpy as np
import argparse
from local_retrieval import CachedRetrieval
from retrieval import GoogleDPRRetrieval
from one_step_operating_points import best_of_k_quoted_answer, generate_paraphrased_answer, generate_entailed_answer, generate_backbone_model_answer, generate_vanilla_answer, generate_citeable_abstractive_answer, get_sub_questions
from eval import AutoEvaluator, evaluate_quote_precision, evaluate_quote_coverage, eval_n_gram_precision
from utils import *
import json
from copy import deepcopy
from global_vars import MODEL_STR
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY'] 
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']
DEEPSEEK_API_KEY = os.environ['DEEPSEEK_API_KEY']
WANDB_API_KEY = os.environ['WANDB_API_KEY']

if 'gpt' in MODEL_STR:
    backbone_model = OpenAI(api_key=OPENAI_API_KEY)
# elif 'claude' in MODEL_STR:
#     backbone_model = Anthropic(api_key=ANTHROPIC_API_KEY)
elif 'deepseek' in MODEL_STR:
    backbone_model = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
else:
    raise ValueError("Model string not recognized.")
    
results_keys = ["ID", 
                "All Sources",
                "All URLs",
                "All Sources (cited)", 
                "Used Sources (cited)", 
                "Question", 
                "Snippet Output (cited)", 
                "Quoted Output (cited)", 
                "Paraphrased Output (cited)", 
                "Entailed Output (cited)", 
                "Abstractive Output (cited)", 
                "Snippet Output", 
                "Quoted Output", 
                "Paraphrased Output", 
                "Entailed Output", 
                "Abstractive Output", 
                "Quoted Sent (cited)", 
                "Paraphrased Sent (cited)", 
                "Entailed Sent (cited)", 
                "Abstractive Sent (cited)", 
                "Quoted Sent", 
                "Paraphrased Sent", 
                "Entailed Sent", 
                "Abstractive Sent", 
                "Quoted Citation Dict", 
                "Paraphrased Citation Dict", 
                "Entailed Citation Dict", 
                "Abstractive Citation Dict", 
                "Quoted Citation Count", 
                "Paraphrased Citation Count", 
                "Entailed Citation Count", 
                "Abstractive Citation Count", 
                "Snippet Fluency Rating", 
                "Snippet Perceived Utility Rating", 
                "Snippet n-gram precision",
                "Quoted Fluency Rating", 
                "Quoted Perceived Utility Rating", 
                "Quoted n-gram precision",
                "Paraphrased Fluency Rating", 
                "Paraphrased Perceived Utility Rating", 
                "Paraphrased n-gram precision",
                "Entailed Fluency Rating", 
                "Entailed Perceived Utility Rating", 
                "Entailed n-gram precision",
                "Abstractive Fluency Rating", 
                "Abstractive Perceived Utility Rating", 
                "Abstractive n-gram precision",
                "Chunks", 
                "Max Tokens", 
                "Temperature",
                "Quoted inference_run_count",
                "Quoted inference_token_input_count",
                "Quoted inference_token_output_count",
                "Paraphrased inference_run_count",
                "Paraphrased inference_token_input_count",
                "Paraphrased inference_token_output_count",
                "Entailed inference_run_count",
                "Entailed inference_token_input_count",
                "Entailed inference_token_output_count",
                "Abstractive inference_run_count",
                "Abstractive inference_token_input_count",
                "Abstractive inference_token_output_count",
                "Abstention Type",
                ]
                
def get_citation_number(quote, text):
    # Escape special regex characters in the quote
    escaped_quote = re.escape(quote)

    # Regex pattern: quote followed by whitespace and [number]
    pattern = rf'{escaped_quote}"\s*\[(\d+)\]'

    match = re.search(pattern, text)
    if match:
        citation_number = int(match.group(1))
        return citation_number
    return -1

def get_numbered_source_str(source_ls):
    source_str = "\n\nSources:\n\n"
    i = 0
    for source in source_ls:
        source_str += "\""
        source = source.strip()
        source_sentence_ls = source.split(".")
        for sentence in source_sentence_ls:
            source_str += f" [{i+1}] {sentence}."
            i += 1
        source_str += "\"\n\n"
    return source_str  

def get_quote_from_source_with_sentence_number(source_with_sentence_numbers, citation_number):
    if citation_number == -1:
        return ""
    citation_marker = f" [{citation_number}]"
    half_source = source_with_sentence_numbers.split(citation_marker)[1]
    quote = half_source.split(f"[{citation_number+1}]")[0].strip()
    return quote

def highlight_direct_quotes(response, source_with_sent_numbers, sources, client, data_str):
    
    quotes_ls = get_quoted_sentences(response, client, data_str=data_str)
    unmarked_response = format_remove_quotation_marks(response)
    if (data_str != 'mash'):
        unmarked_sentences = get_sentences_gpt4(unmarked_response, client)
    else:
        unmarked_sentences = get_sentences_tokenizer(unmarked_response)

    citation_number_ls = [get_citation_number(quote, response) for quote in quotes_ls]

    citation_number_mapping = {}
    y = 1
    for num in citation_number_ls:
        citation_number_mapping[num] = y
        y += 1

    num_citations = 0
    # sources_idxs is a dictionary that map quote # to source #, the corresponding index slices for highlighting, and the citation number!
    sources_idxs = {}
    highlighted_sources = copy.deepcopy(sources)
    highlighted_uncited_sources = copy.deepcopy(sources)
    used_sources = {}
    sentences = copy.deepcopy(unmarked_sentences)
    sentences_to_citations = {} # always contains all sentences idxs (even if a sentence doesn't have a citation)
    for i in range(len(sentences)):
        sentences_to_citations[i] = {'citation_numbers': []}
    for i in range (len(quotes_ls)):
        quote = quotes_ls[i]
        highlighted_quote = COLORS[i]+'\"'+quote+'\"'+COLORS[10]
        citation_number = '['+str(citation_number_mapping[citation_number_ls[i]])+']'
        source_quote = get_quote_from_source_with_sentence_number(source_with_sent_numbers, citation_number_ls[i])
        if not source_quote:
            continue
        highlighted_source_quote = COLORS[i]+citation_number+' \"'+source_quote+'\"'+COLORS[10]

        for j, source in enumerate(sources):
            if (source_quote in source):
                highlighted_sources[j] = highlighted_sources[j].replace(source_quote, highlighted_source_quote) 
                highlighted_uncited_sources[j] = highlighted_uncited_sources[j].replace(source_quote, COLORS[i]+' \"'+source_quote+'\"'+COLORS[10])
                for ii in range(len(sentences)):
                    if (quote in sentences[ii]):
                        sentences[ii] = sentences[ii].replace(quote, highlighted_quote)
                        sentences[ii] = sentences[ii].replace('['+str(citation_number_ls[i])+']', citation_number)
                        sentences_to_citations[ii]['citation_numbers'].append(citation_number_mapping[citation_number_ls[i]])
                used_sources[j] = None

                num_citations += 1
                start_idx = source.lower().find(source_quote.lower())
                end_idx = start_idx+len(source_quote)
                sources_idxs[i] = {'source_idx':j, 'citation_numbers':[citation_number], 'start':start_idx, 'end':end_idx}
             
        if (i not in sources_idxs.keys()):
            sources_idxs[i] = None
    
    used_unmarked_sources = []
    used_highlighted_sources = []
    used_highlighted_uncited_sources = []
    for j in used_sources.keys():
        used_unmarked_sources.append(sources[j])
        used_highlighted_sources.append(highlighted_sources[j])
        used_highlighted_uncited_sources.append(highlighted_uncited_sources[j])
    cited_response = " ".join(sentences)
    for i in range(len(sentences)):
        sentences[i] = sentences[i].replace("'", "\'")
        unmarked_sentences[i] = unmarked_sentences[i].replace("'", "\'")

    return cited_response, highlighted_sources, sources_idxs, used_sources, used_unmarked_sources, used_highlighted_sources, sentences, unmarked_sentences, num_citations, sentences_to_citations, used_highlighted_uncited_sources



def cite_paraphrased_quotes(paraphrased_response, client):
    if not paraphrased_response:
        return None, None, None, None, None
    # identify all paraphrased sentences
    cited_paraphrased_sentences = get_sentences_gpt4(paraphrased_response, client) 

    num_citations = 0
    # citations_dict is a mapping from sentence to the citation numbers for that sentence
    citations_dict = {}   

    for i in range(len(cited_paraphrased_sentences)):
        citations_dict[i] = {'citation_numbers': []}
        paraphrased_sentence = cited_paraphrased_sentences[i]
        if (len(paraphrased_sentence) == 0):
            continue
        
        proposed_citation_numbers = re.findall(r'\[(\d+)\]', paraphrased_sentence)
        citation_numbers_ls = []
        for proposed_citation_number in proposed_citation_numbers:
            citation_numbers_ls.append(int(proposed_citation_number))
            num_citations += 1

        citations_dict[i]['citation_numbers'].extend(citation_numbers_ls)

    for i in range(len(cited_paraphrased_sentences)):
        cited_paraphrased_sentences[i] = cited_paraphrased_sentences[i].replace("'", "\'")
    paraphrased_response = paraphrased_response.replace("'", "\'")
        
    cited_paraphrased_response = paraphrased_response
    paraphrased_sentences = []
    for sent in cited_paraphrased_sentences:
        # Remove the citation numbers
        sent = re.sub(r'\[(\d+)\]', '', sent)
        paraphrased_sentences.append(sent)

    return cited_paraphrased_response, cited_paraphrased_sentences, paraphrased_sentences, num_citations, citations_dict

def run_operating_points_and_eval(sample, args, source_type, evaluator, wandb_table=None):
    if (source_type == 'gold'):
        sources = sample['gold_reference']
        urls = sample['urls'] # TODO need to implement for all datasets
    elif (source_type == 'full'):
        sources = sample['full_reference']
    elif (source_type == 'retrieved'):
        sources = sample['retrieved_sources']
        urls = sample['retrieved_urls']
    else:
        print('Source type not supported')
        exit()   
         
    for i in range(len(sources)):
        sources[i] = sources[i].replace('\xa0', ' ')
        sources[i] = sources[i].replace('\n', ' ')
    sources_with_urls = [u+'\n'+s for u, s in zip(urls, sources)]
    query = sample['question']
    abstention_type = 'No Failure'
    if (len(sources) == 0):
        # assign 'Insufficient information to ...'
        quoted_response = 'Insufficient information to generate a grounded response.'

        # placeholder details_dict
        quoted_details_dict ={}
        quoted_details_dict['max_tokens'] = -1
        quoted_details_dict['temperature'] = -1
    else:
        sources_with_sentence_numbers = get_numbered_source_str(sources)
        quoted_response, quoted_details_dict = best_of_k_quoted_answer(query, sources_with_sentence_numbers, backbone_model, k=args.best_of_k, dataset=args.data)

    if ('Insufficient information to generate a grounded response.' in quoted_response):
        # prepare to log a generation failure (although these can really be retrieval failures if sources does not contain an answer)
        if (len(sources) == 0):
            # prepare to log a retrieval failure
            abstention_type = 'Retrieval Failure'
        else:
            # prepare to log a generation failure
            abstention_type = 'Generation Failure'
            
        snippet_response, paraphrased_response, entailed_response, abstractive_response = '', '', '', ''
        highlighted_quoted_response, cited_paraphrased_response, cited_entailed_response, cited_abstractive_response = quoted_response, '', '', ''
        m_sentences_quoted_response, m_sentences_paraphrased_response, m_sentences_entailed_response, m_sentences_abstractive_response = [], [], [], []
        sentences_quoted_response, sentences_paraphrased_response, sentences_entailed_response, sentences_abstractive_response = [], [], [], []
        quoted_num_cited, paraphrased_num_cited, entailed_num_cited, abstractive_num_cited = 0,0,0,0
        quoted_citations_dict, paraphrased_citations_dict, entailed_citations_dict, abstractive_citations_dict = {}, {}, {}, {}
        m_sentences_quoted_response, m_sentences_paraphrased_response, m_sentences_entailed_response, m_sentences_abstractive_response = [], [], [], []
        sentences_quoted_response, sentences_paraphrased_response, sentences_entailed_response, sentences_abstractive_response = [], [], [], []
        abstained = True
        quoted_token_counts = [-1,-1,-1]
        paraphrased_token_counts = [-1,-1,-1]
        entailed_token_counts = [-1,-1,-1]
        abstractive_token_counts = [-1,-1,-1]
    else:
        highlighted_quoted_response, highlighted_sources, sources_idxs, used_sources, used_unmarked_sources, used_highlighted_cited_sources, m_sentences_quoted_response, sentences_quoted_response, quoted_num_cited, quoted_citations_dict, used_highlighted_uncited_sources = highlight_direct_quotes(quoted_response, sources_with_sentence_numbers, sources, backbone_model, args.data)
        print('\n'+COLORS[11]+'Grounded & Quoted Response:'+COLORS[10]+'\n'+highlighted_quoted_response)
        quoted_token_counts = deepcopy([global_vars.inference_run_count, global_vars.inference_token_input_count, global_vars.inference_token_output_count])
        global_vars.inference_run_count = 0
        global_vars.inference_token_input_count = 0
        global_vars.inference_token_output_count = 0
        
        cited_paraphrased_response=None
        while not cited_paraphrased_response:
            paraphrased_response, paraphrased_details_dict = generate_paraphrased_answer(query, highlighted_quoted_response, backbone_model, args.data)

            cited_paraphrased_response, m_sentences_paraphrased_response, sentences_paraphrased_response, paraphrased_num_cited, paraphrased_citations_dict = cite_paraphrased_quotes(paraphrased_response, backbone_model)

        print('\n'+COLORS[11]+'Grounded & Paraphrased Response:'+COLORS[10]+'\n'+cited_paraphrased_response)
        
        paraphrased_token_counts = deepcopy([global_vars.inference_run_count, global_vars.inference_token_input_count, global_vars.inference_token_output_count])
        global_vars.inference_run_count = 0
        global_vars.inference_token_input_count = 0
        global_vars.inference_token_output_count = 0

        cited_entailed_response=None
        while not cited_entailed_response:
            entailed_response, entailed_details_dict = generate_entailed_answer(query, highlighted_quoted_response, backbone_model, args.data)
            cited_entailed_response, m_sentences_entailed_response, sentences_entailed_response, entailed_num_cited, entailed_citations_dict = cite_paraphrased_quotes(entailed_response, backbone_model)
        print('\n'+COLORS[11]+'Grounded & Entailed Response:'+COLORS[10]+'\n'+cited_entailed_response)

        entailed_token_counts = deepcopy([global_vars.inference_run_count, global_vars.inference_token_input_count, global_vars.inference_token_output_count])
        global_vars.inference_run_count = 0
        global_vars.inference_token_input_count = 0
        global_vars.inference_token_output_count = 0

        abstained = False
        cited_abstractive_response=None
        while not cited_abstractive_response:
            abstractive_response, abstractive_details_dict = generate_citeable_abstractive_answer(query, highlighted_quoted_response, backbone_model, args.data)
            cited_abstractive_response, m_sentences_abstractive_response, sentences_abstractive_response, abstractive_num_cited, abstractive_citations_dict = cite_paraphrased_quotes(abstractive_response, backbone_model)
        print('\n'+COLORS[11]+'Partially Grounded Abstractive Response:'+COLORS[10]+'\n'+cited_abstractive_response)
        
        abstractive_token_counts = deepcopy([global_vars.inference_run_count, global_vars.inference_token_input_count, global_vars.inference_token_output_count])
        global_vars.inference_run_count = 0
        global_vars.inference_token_input_count = 0
        global_vars.inference_token_output_count = 0

    if (not abstained):
        # Print sources
        print('\nSources:')
        for i in range(len(used_highlighted_cited_sources)):
            print("\n"+used_highlighted_cited_sources[i].replace("\n", "")+"\n")
        cited_sources_str = highlighted_sources # format_source_list(highlighted_sources) 

        snippet_response_without_urls = []
        for s in used_unmarked_sources:
            pieces = s.split('\n')
            pieces_to_keep = pieces[1:]
            snippet_response_without_urls.append("\n".join(pieces_to_keep))

        used_highlighted_uncited_sources_without_urls = []
        for s in used_highlighted_uncited_sources:
            pieces = s.split('\n')
            pieces_to_keep = pieces[1:]
            used_highlighted_uncited_sources_without_urls.append("\n".join(pieces_to_keep))
    else:
        cited_sources_str = []
        used_highlighted_cited_sources = []
        snippet_response_without_urls = []
        used_highlighted_uncited_sources_without_urls = []
        used_highlighted_uncited_sources = []
        # highlighted_snippet_without_urls = []

    operating_points_unmarked = [snippet_response_without_urls, quoted_response, paraphrased_response, entailed_response, abstractive_response]
    # operating_points = [quoted_response, paraphrased_response, entailed_response, abstracted_response]
    # TODO uncomment the section below when ready to evaluate
    # operating_points = [quoted_response, paraphrased_response, entailed_response]
    eval_values = []
    for op in operating_points_unmarked: # TODO uncomment!
        # results = run_all_evals(query, sources, op, evaluator, abstained)
        results = [None, None, None]
        eval_values.extend(results)

    if (not args.debug):
        quoted_details_dict['max_tokens'] = -1
        quoted_details_dict['temperature'] = -1
        results_values = [sample['id'],
                    sources,
                    urls,
                    cited_sources_str,
                    used_highlighted_cited_sources,
                    sample['question']
                    ]
        cited_operating_points = [used_highlighted_uncited_sources, highlighted_quoted_response.replace("'", "\'"), cited_paraphrased_response.replace("'", "\'"), cited_entailed_response.replace("'", "\'"), cited_abstractive_response.replace("'", "\'")]
        operating_points_unmarked = [used_highlighted_uncited_sources_without_urls, quoted_response.replace("'", "\'"), paraphrased_response.replace("'", "\'"), entailed_response.replace("'", "\'"), abstractive_response.replace("'", "\'")] # redefine to include highlighted snippet responses
        num_cited_by_op = [quoted_num_cited, paraphrased_num_cited, entailed_num_cited, abstractive_num_cited]
        op_marked_sentences = [m_sentences_quoted_response, m_sentences_paraphrased_response, m_sentences_entailed_response, m_sentences_abstractive_response]
        op_sentences = [sentences_quoted_response, sentences_paraphrased_response, sentences_entailed_response, sentences_abstractive_response]
        op_citation_dicts = [quoted_citations_dict, paraphrased_citations_dict, entailed_citations_dict, abstractive_citations_dict]
        results_values.extend(cited_operating_points)
        results_values.extend(operating_points_unmarked)
        results_values.extend(op_marked_sentences) 
        results_values.extend(op_sentences) 
        results_values.extend(op_citation_dicts) 
        results_values.extend(num_cited_by_op)
        results_values.extend(eval_values)
        results_values.extend([args.num_IC_sources, quoted_details_dict['max_tokens'], quoted_details_dict['temperature']])
        results_values.extend(quoted_token_counts)
        results_values.extend(paraphrased_token_counts)
        results_values.extend(entailed_token_counts)
        results_values.extend(abstractive_token_counts)
        results_values.append(abstention_type)
       
        # open file and write another line
        results_fp = "one_step_results/"+args.project_name+".jsonl"
        data_dict = dict(zip(results_keys, results_values))

        if (not os.path.exists(results_fp)):
            with open(results_fp, "w") as f:
                json_string = json.dumps(data_dict, default=to_serializable)  # Convert item to JSON string
                f.write(json_string + "\n")  # Write with newline character
        else:
            with open(results_fp, "a") as f:
                json_string = json.dumps(data_dict, default=to_serializable)  # Convert item to JSON string
                f.write(json_string + "\n")  # Write with newline character
        print('Saved to '+results_fp)
        # wandb_table.add_data(*results_values)
    return

def to_serializable(val):
    if isinstance(val, np.generic):
        return val.item()
    raise TypeError(f"Type {type(val)} not serializable")

def main(args):
    evaluator = AutoEvaluator()
    if (args.gold):
        print('Answering each question with GOLD-STANDARD source.')
        source_type = 'gold'
    elif (args.full):
        print('Answering each question with FULL source.')
        source_type = 'full'
    elif (args.retriever):
        print('Answering each question with RETRIEVED sources.')
        source_type = 'retrieved'
    else:
        # this is outdated/not used
        print('Please specify the source type.')
        exit

    if args.rerun_question_source_pairs:
        # Load in csv of question-source pairs to rerun
        df = pd.read_csv(args.rerun_question_source_pairs)

        for i in range(args.start_n, min(len(df), args.end_n)):
            all_sources = df['All Sources'].iloc[i]
            all_urls = df['All URLs'].iloc[i]
            question = df['Question'].iloc[i]
            id = df['ID'].iloc[i]

            instance = {}
            print(f"Answering question {str(i)}/{len(df)}")
            print('\n'+COLORS[11]+'Question: '+COLORS[10]+question)

            instance['question'] = question
            instance['id'] = id
            if args.gold:
                instance['gold_reference'] = ast.literal_eval(all_sources)
                instance['urls'] = ast.literal_eval(all_urls)
            else:
                print('Warning: Choose gold to use the sources previously used for this question in the file `rerun_question_source_pairs`.')

            run_operating_points_and_eval(instance, args, source_type, evaluator) 
            print('__________________________________________________________________')

    else:
        print('Please specify the file of question-source pairs to rerun.')
        exit()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', action='store_true')
    parser.add_argument('--retriever', default=None, type=str) # 'google'
    parser.add_argument('--full', action='store_true')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--start_n', type=int, default=0)
    parser.add_argument('--end_n', type=int, default=200)
    parser.add_argument('--n', type=int, default=60)
    parser.add_argument('--num_IC_sources', type=int, default=10) # only used for GoogleDPRRetrieval
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--data', type=str, required=True) # 'nq' or 'medical' or 'multihop' or 'emr' or 'eli5_nq'or 'mash'
    parser.add_argument('--best_of_k', type=int, default=10)
    parser.add_argument('--rerun_question_source_pairs', type=str, default=None) # mturk_results/nq_mturk3_0_180_byQueryOP.csv
    parser.add_argument('--use_sub_questions', action='store_true')
    parser.add_argument('--use_optimized_prompt', action='store_true')

    args = parser.parse_args()
    main(args)

# python generateInlineQuotes.py --project_name test --gold --data multihop --n 10 --best_of_k 1 --debug --start_n 0
# python generateInlineQuotes.py --project_name test --retriever google --data nq --n 10 --best_of_k 1 --debug --start_n 0


# python generateInlineQuotes.py --start_n 60 --n 20 --project_name nq_googleDPR_mturk --data nq --retriever google --num_IC_sources 10 --best_of_k 5
# python generateInlineQuotes.py --start_n 60 --n 20 --project_name eli5_nq_googleDPR_may13 --data nq --retriever google --num_IC_sources 10 --best_of_k 5

# python generateInlineQuotes.py --start_n 30 --n 1 --project_name lithium --data nq --retriever google --best_of_k 10

# python generateInlineQuotes.py --project_name mh_teddi_eval2 --gold --data multihop --n 20 --best_of_k 1 --start_n 20

# python generateInlineQuotes.py --project_name eli3_test --retriever google --data nq --n 20 --best_of_k 1 --start_n 0


# local # python generateInlineQuotes.py --project_name mturk --retriever google --data nq --n 40 --best_of_k 10 --start_n 40
# local # python generateInlineQuotes.py --project_name mturk --retriever google --data nq --n 40 --best_of_k 10 --start_n 80 # crashed on instance 102; see last row
# tmux0 # python generateInlineQuotes.py --project_name mturk2 --retriever google --data nq --n 40 --best_of_k 10 --start_n 120 # crashed on 145 # completed below!!
# tmux0 # python generateInlineQuotes.py --project_name mturk2 --retriever google --data nq --n 14 --best_of_k 10 --start_n 146 # completed!!


# tmux1 # python generateInlineQuotes.py --project_name mturk3 --retriever google --data nq --n 40 --best_of_k 10 --start_n 160 # completed!!!
# tmux2 # python generateInlineQuotes.py --project_name mturk4 --retriever google --data nq --n 40 --best_of_k 10 --start_n 201 # crashed on 200 # completed!!!
# tmux3 # python generateInlineQuotes.py --project_name mturk1 --retriever google --data nq --n 10 --best_of_k 10 --start_n 110 # crashed on 102, 109 # completed!!


# python generateInlineQuotes.py --project_name mh_mturk --gold --data multihop --n 40 --best_of_k 10 --start_n 60
# python generateInlineQuotes.py --project_name mh_mturk --gold --data multihop --n 40 --best_of_k 10 --start_n 100
# python generateInlineQuotes.py --project_name mh_mturk --gold --data multihop --n 40 --best_of_k 10 --start_n 140
# python generateInlineQuotes.py --project_name mh_mturk --gold --data multihop --n 40 --best_of_k 10 --start_n 180

# python generateInlineQuotes.py --project_name nq_comparison1 --retriever google --data nq --n 10 --best_of_k 10 --start_n 0
# python generateInlineQuotes.py --project_name eli5_nq_comparison1 --retriever google --data eli5_nq --n 10 --best_of_k 10 --start_n 0

# run annotation_processing_for_sl and note the abstention rate

# python generateInlineQuotes.py --project_name eli5_teddi_eval2 --retriever google --data eli5_nq --n 25 --best_of_k 10 --num_IC_sources 10 --start_n 10 

# python generateInlineQuotes.py --project_name debug --retriever google --data mash --n 5 --best_of_k 10 --num_IC_sources 10 --start_n 10 --debug 

###############################################################################################################################################

# python generateOperatingPoints.py --project_name gpt5_nq --gold --data nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/nq.csv
# python generateOperatingPoints.py --project_name gpt5_eli5_nq --gold --data eli5_nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/eli5_nq.csv
# python generateOperatingPoints.py --project_name gpt5_mash --gold --data mash --best_of_k 10 --rerun_question_source_pairs evaluated_instances/mash.csv
# python generateOperatingPoints.py --project_name gpt5_multihop --gold --data multihop --best_of_k 10 --rerun_question_source_pairs evaluated_instances/multihop.csv 

###############################################################################################################################################

# python generateOperatingPoints.py --project_name sonnet4.5_nq --gold --data nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/nq.csv --end_n 50 --start_n 0 
# python generateOperatingPoints.py --project_name sonnet4.5_nq --gold --data nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/nq.csv --end_n 100 --start_n 50 
# python generateOperatingPoints.py --project_name sonnet4.5_nq --gold --data nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/nq.csv --end_n 200 --start_n 100 

# python generateOperatingPoints.py --project_name sonnet4.5_eli5_nq --gold --data eli5_nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/eli5_nq.csv --end_n 50 --start_n 0 
# python generateOperatingPoints.py --project_name sonnet4.5_eli5_nq --gold --data eli5_nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/eli5_nq.csv --end_n 100 --start_n 50 
# python generateOperatingPoints.py --project_name sonnet4.5_eli5_nq --gold --data eli5_nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/eli5_nq.csv --end_n 200 --start_n 100 

# python generateOperatingPoints.py --project_name sonnet4.5_mash --gold --data mash --best_of_k 10 --rerun_question_source_pairs evaluated_instances/mash.csv --end_n 50 --start_n 0 
# python generateOperatingPoints.py --project_name sonnet4.5_mash --gold --data mash --best_of_k 10 --rerun_question_source_pairs evaluated_instances/mash.csv --end_n 100 --start_n 50 
# python generateOperatingPoints.py --project_name sonnet4.5_mash --gold --data mash --best_of_k 10 --rerun_question_source_pairs evaluated_instances/mash.csv --end_n 200 --start_n 100 

# python generateOperatingPoints.py --project_name sonnet4.5_multihop --gold --data multihop --best_of_k 10 --rerun_question_source_pairs evaluated_instances/multihop.csv --end_n 50 --start_n 0 
# python generateOperatingPoints.py --project_name sonnet4.5_multihop --gold --data multihop --best_of_k 10 --rerun_question_source_pairs evaluated_instances/multihop.csv --end_n 100 --start_n 50 
# python generateOperatingPoints.py --project_name sonnet4.5_multihop --gold --data multihop --best_of_k 10 --rerun_question_source_pairs evaluated_instances/multihop.csv --end_n 200 --start_n 100 

### debugging
# python generateOperatingPoints.py --project_name gpt5_nq_debug --gold --data nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/nq.csv

########### Generate without sub-questions ###########
# python generateOperatingPoints.py --project_name gpt5_nq --gold --data nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/nq.csv --end_n 25 --start_n 0 
# python generateOperatingPoints.py --project_name gpt5_nq --gold --data nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/nq.csv --end_n 50 --start_n 25 
# python generateOperatingPoints.py --project_name gpt5_nq --gold --data nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/nq.csv --end_n 75 --start_n 50 
# python generateOperatingPoints.py --project_name gpt5_nq --gold --data nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/nq.csv --end_n 100 --start_n 75 
# python generateOperatingPoints.py --project_name gpt5_nq --gold --data nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/nq.csv --end_n 125 --start_n 100 
# python generateOperatingPoints.py --project_name gpt5_nq --gold --data nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/nq.csv --end_n 150 --start_n 125 
# python generateOperatingPoints.py --project_name gpt5_nq --gold --data nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/nq.csv --end_n 175 --start_n 150 
# python generateOperatingPoints.py --project_name gpt5_nq --gold --data nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/nq.csv --end_n 200 --start_n 175 




# python generateOperatingPoints.py --project_name DEBUG_gpt5_nq --gold --data nq --best_of_k 10 --rerun_question_source_pairs evaluated_instances/DEBUG_nq.csv --end_n 100 --start_n 0 






