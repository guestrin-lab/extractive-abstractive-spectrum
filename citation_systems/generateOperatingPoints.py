import os
from openai import OpenAI
from naturalQuestions import NaturalQuestions
from wikiMultiHopQA import WikiMultiHopQA
from mashQA import MashQA
from quoteEval import eval_precision, eval_quote_coverage
import numpy as np
import argparse
from operating_points import best_of_k_quoted_answer, generate_paraphrased_answer, generate_entailed_answer, generate_backbone_model_answer, generate_vanilla_answer, generate_citeable_abstractive_answer, get_sub_questions
from eval import AutoEvaluator, evaluate_quote_precision, evaluate_quote_coverage, eval_n_gram_precision
from utils import *
import json
from copy import deepcopy

backbone_model = OpenAI()
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

def run_all_evals(query, sources, response, evaluator, abstained):
    if (abstained):
        return [1, 0, 0, 0, 0, 1]
    unmarked_response = format_remove_quotation_marks(response)
    fluency_rating = evaluator.evaluate_fluency(query, unmarked_response)
    perceived_utility_rating = evaluator.evaluate_perceived_utility(query, unmarked_response)

    p_ls = [eval_n_gram_precision(unmarked_response, sources, i) for i in range(1,5)] # list of (num, denom) tuples
    return [fluency_rating, perceived_utility_rating, p_ls]

def run_operating_points_and_eval(sample, args, source_type, evaluator):
    if (source_type == 'gold'):
        sources = sample['gold_reference']
        urls = sample['urls'] 
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
        quoted_response = 'Insufficient information to generate a grounded response.'

        # placeholder details_dict
        quoted_details_dict ={}
        quoted_details_dict['max_tokens'] = -1
        quoted_details_dict['temperature'] = -1
    else:
        quoted_response, quoted_details_dict = best_of_k_quoted_answer(query, sources, backbone_model, k=args.best_of_k, dataset=args.data, using_gold=(source_type=='gold'))

    if ('Insufficient information to generate a grounded response.' in quoted_response):
        # prepare to log a failure
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
        highlighted_quoted_response, highlighted_sources, sources_idxs, used_sources, used_unmarked_sources, used_highlighted_cited_sources, m_sentences_quoted_response, sentences_quoted_response, quoted_num_cited, quoted_citations_dict, used_highlighted_uncited_sources = highlight_direct_quotes(quoted_response, sources_with_urls, backbone_model, args.data)
        print('\n'+COLORS[11]+'Grounded & Quoted Response:'+COLORS[10]+'\n'+highlighted_quoted_response)
        quoted_token_counts = deepcopy([global_vars.inference_run_count, global_vars.inference_token_input_count, global_vars.inference_token_output_count])
        global_vars.inference_run_count = 0
        global_vars.inference_token_input_count = 0
        global_vars.inference_token_output_count = 0
        #
        paraphrased_response, paraphrased_details_dict = generate_paraphrased_answer(query, quoted_response, backbone_model, dataset=args.data)
        cited_paraphrased_response, m_sentences_paraphrased_response, sentences_paraphrased_response, paraphrased_num_cited, paraphrased_citations_dict = cite_paraphrased_quotes(quoted_response, paraphrased_response, sources, backbone_model, 'paraphrased', sources_idxs, args.data)
        print('\n'+COLORS[11]+'Grounded & Paraphrased Response:'+COLORS[10]+'\n'+cited_paraphrased_response)
        
        paraphrased_token_counts = deepcopy([global_vars.inference_run_count, global_vars.inference_token_input_count, global_vars.inference_token_output_count])
        global_vars.inference_run_count = 0
        global_vars.inference_token_input_count = 0
        global_vars.inference_token_output_count = 0

        entailed_response, entailed_details_dict = generate_entailed_answer(query, quoted_response, backbone_model, dataset=args.data)
        cited_entailed_response, m_sentences_entailed_response, sentences_entailed_response, entailed_num_cited, entailed_citations_dict = cite_paraphrased_quotes(quoted_response, entailed_response, sources, backbone_model, 'entailed', sources_idxs, args.data)
        print('\n'+COLORS[11]+'Grounded & Entailed Response:'+COLORS[10]+'\n'+cited_entailed_response)

        entailed_token_counts = deepcopy([global_vars.inference_run_count, global_vars.inference_token_input_count, global_vars.inference_token_output_count])
        global_vars.inference_run_count = 0
        global_vars.inference_token_input_count = 0
        global_vars.inference_token_output_count = 0

        abstained = False
        abstractive_response, abstractive_details_dict = generate_citeable_abstractive_answer(query, quoted_response, backbone_model, dataset=args.data)
        cited_abstractive_response, m_sentences_abstractive_response, sentences_abstractive_response, abstractive_num_cited, abstractive_citations_dict = cite_paraphrased_quotes(quoted_response, abstractive_response, sources, backbone_model, 'abstracted', sources_idxs, args.data)
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
        cited_sources_str = highlighted_sources 

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

    operating_points_unmarked = [snippet_response_without_urls, quoted_response, paraphrased_response, entailed_response, abstractive_response]
    eval_values = []
    for op in operating_points_unmarked:
        results = [None, None, None]
        eval_values.extend(results)

    if (not args.debug):
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
        results_fp = "generation_results/"+args.project_name+".jsonl"
        data_dict = dict(zip(results_keys, results_values))
        if (not os.path.exists(results_fp)):
            with open(results_fp, "w") as f:
                json_string = json.dumps(data_dict)  # Convert item to JSON string
                f.write(json_string + "\n")  # Write with newline character
        else:
            with open(results_fp, "a") as f:
                json_string = json.dumps(data_dict)  # Convert item to JSON string
                f.write(json_string + "\n")  # Write with newline character
        print('Saved to '+results_fp)
    return

def run_only_quoted_op(sample, args, source_type, evaluator):
    backbone_model = OpenAI()

    if (source_type == 'gold'):
        sources = sample['gold_reference']
        urls = sample['urls'] 
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
    print('\n'+COLORS[11]+'Question: '+COLORS[10]+sample['question'])
    
    quoted_response, quoted_details_dict = best_of_k_quoted_answer(query, sources, backbone_model, k=args.best_of_k, dataset=args.data, using_gold=(source_type=='gold'))
    highlighted_quoted_response, highlighted_sources, sources_idxs, used_sources, used_unmarked_sources, used_highlighted_cited_sources, m_sentences_quoted_response, sentences_quoted_response, quoted_num_cited, quoted_citations_dict, used_highlighted_uncited_sources = highlight_direct_quotes(quoted_response, sources_with_urls, backbone_model)
    print('\n'+COLORS[11]+'Grounded & Quoted Response:'+COLORS[10]+'\n'+highlighted_quoted_response)

def getPreference(client, sample, quoted_response_text, vanilla_response_text):
    instruction = "\nWhich of the two responses below is a more fluent answer to the question? Answer with either \"Response 1\", \"Response 2\", or \"Both are equally fluent\".\n"
    question = "Question: "+sample['question']+"\n"
    quoted_response_text = "Response 1: "+format_remove_quotation_marks(quoted_response_text)+"\n"
    vanilla_response_text = "Response 2: "+vanilla_response_text+"\n"
    gpt_user_prompt =  instruction + question + quoted_response_text + vanilla_response_text 
    response_text, details_dict = generate_from_model(client, gpt_user_prompt)

    print('PREFERENCE response_text:', response_text)
    if (response_text == "Response 1"):
        return response_text, 1
    if (response_text == "Response 2"):
        return response_text, 0
    if ((response_text == "Both are equally fluent") or (response_text == "Both are equally fluent.")):
        return response_text, 2
    return response_text, -1

def main(args):
    client = OpenAI()
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
        if (args.retriever == 'google'):
            retriever = GoogleDPRRetrieval(50, args.num_IC_sources)
        else:
            print('Retriever not supported')
    else:
        # this is outdated/not used
        print('Please specify the source type.')
        exit
    
    if ((args.data == 'nq') or (args.data == 'eli5_nq')):
        data = NaturalQuestions(seed=0)
    elif (args.data == 'multihop'):
        data = WikiMultiHopQA(seed=0)
    elif (args.data == 'mash'):
        data = MashQA(seed=0)
    
    query_id_range = range(args.start_n, args.start_n+args.n)

    for i in query_id_range: 
        global_vars.inference_run_count = 0
        global_vars.inference_token_input_count = 0
        global_vars.inference_token_output_count = 0

        print('Answering question', str(i))
        
        instance = data[i]
        if (instance == None):
            continue

        print('\n'+COLORS[11]+'Question: '+COLORS[10]+instance['question'])
        
        if (args.retriever):
            try:
                sources, top_urls, top_titles, best_chunk, best_url, best_title = retriever.retrieve(instance['question'])
            except:
                sources = []
            if ((sources == None) or (len(sources) == 0)):
                sources = []
                top_urls = []
            instance['retrieved_sources'] = sources
            instance['retrieved_urls'] = top_urls
        instance['id'] = i # overriding all query IDs
        if (args.data == 'eli5_nq'):
            instance['question'] = 'Explain to a third-grader: '+instance['question']
        run_operating_points_and_eval(instance, args, source_type, evaluator) 
        print('__________________________________________________________________')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', action='store_true')
    parser.add_argument('--retriever', default=None, type=str) # 'google'
    parser.add_argument('--full', action='store_true')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--start_n', type=int, default=0)
    parser.add_argument('--n', type=int, default=60)
    parser.add_argument('--num_IC_sources', type=int, default=10) # only used for GoogleDPRRetrieval
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--data', type=str, required=True) # 'nq' or 'multihop' or 'eli5_nq'or 'mash'
    parser.add_argument('--best_of_k', type=int, default=10)
    args = parser.parse_args()
    main(args)