import os
import re 
import ast
import numpy as np
import pandas as pd 
from tqdm import tqdm

import asyncio
from openai import AsyncOpenAI
import time
import argparse

class AutoEvaluator:
    def __init__(self):
        self.model = AsyncOpenAI()
        self.frequency_penalty=0.0

    async def evaluate_likert(self, query, response, answer_choices):
        response_for_prompt = "\nResponse: "+response.replace('\"', '')+"\n"
        prompt = query+response_for_prompt+answer_choices+"\nAnswer: "
        rating_output_text = None
        rating_output_text = await self.prompt_model(prompt, ['1', '2', '3'])
        if (rating_output_text in ['1', '2', '3']):
            rating_output_text = int(rating_output_text)
        else: 
            print('Took short-cut to parse rating output text:      ', rating_output_text)
            if ('1' in rating_output_text):
                rating_output_text = 1
            elif ('2' in rating_output_text):
                rating_output_text = 2
            elif ('3' in rating_output_text):
                rating_output_text = 3
            else:
                rating_output_text = -1
        return rating_output_text
    
    async def prompt_model(self, prompt, answer_choices):
        message=[{"role": "assistant", "content": "a helpful expert"}, {"role": "user", "content": prompt}]
        rating_output_text = -1
        num_attempts = 0
        while ((rating_output_text not in answer_choices) and (num_attempts < 5)):
            rating_output = await self.model.chat.completions.create(
                    model="gpt-5-2025-08-07", 
                    messages = message,
                    frequency_penalty=self.frequency_penalty
            )
            rating_output_text = rating_output.choices[0].message.content
            num_attempts += 1
        return rating_output_text
    
    async def evaluate_fluency(self, query, response):
        query = "You are an expert in fluent writing who keenly observes misprints, grammar, sentence construction, and flow. To what extent is the response to the query fluent and coherent? Answer with a single number from the multiple choice options below."
        answer_choices = "Multiple choice options:\n1: The response has noticeable misprints, poor grammar, or abrupt transitions between sentences\n2: The response has no misprints, proper grammar, and mostly smooth transitions between sentences\n3: The response has no misprints, proper grammar, and all of the sentences flow nicely together"
        return await self.evaluate_likert(query, response, answer_choices)
    
    async def evaluate_perceived_utility(self, query, response):
        query = "You are an expert in effective question-answering who keenly observes the verbosity, relevance, clarity, and style of responses. To what extent is the response to the query useful? Answer with a single number from the multiple choice options below."
        answer_choices = "Multiple choice options:\n1: The response does not address the query or requires major changes to trim down unnecessary information and/or fix the style\n2: The response addresses the query and requires minor changes to trim down unnecessary information and/or fix the style\n3: The response is satisfying in terms of its length, style, and information addressing the query"
        return await self.evaluate_likert(query, response, answer_choices)

    async def evaluate_citation_precision(self, cited_sentence, response, cited_quote, relevant_cited_source):
        user_prompt = "You are an intelligent and fair attribution evaluator. Your task is to verify whether a given quote supports at least one claim made in the given sentence. The full response containing the sentence and the full source containing the quote are provided as additional context.\n\n"
        user_prompt += f"Sentence: {cited_sentence}\n"
        user_prompt += f"Quote: {cited_quote}\n\n"
        user_prompt += f"Full Response (containing the sentence): {response}\n\n"
        user_prompt += f"Full Source (containing the quote): {relevant_cited_source}\n\n"
        user_prompt += "Does the quote support at least one claim in the sentence? Answer '1' if the quote supports at least one claim in the sentence, or '0' if it does not support any claim in the sentence.\n"
        user_prompt += "Do not explain your answer, just return '1' or '0'.\n"
        user_prompt += "Answer:"
        raw_answer = await self.prompt_model(user_prompt, ['0', '1'])
        answer_cleaned = re.sub(r"[^\w]", "", raw_answer)
        answer = int(answer_cleaned) if answer_cleaned in ["0", "1"] else 0
        return answer
    
    async def evaluate_citation_coverage(self, cited_sentence, response, cited_quotes, relevant_cited_sources):
        user_prompt = "You are an intelligent and fair attribution evaluator. Your task is to verify whether the quote(s), given their context in the full source, support all of the claims made in the given sentence (i.e. coverage). The full response containing the sentence and the full source(s) containing the quote(s) are provided as context. Be sure to consider each quote given its surrounding text in the original source when determining coverage.\n\n"
        user_prompt += f"Sentence: {cited_sentence}\n"
        user_prompt += f"Quotes: {'\n'.join(cited_quotes)}\n\n"
        user_prompt += f"Full Response (containing the sentence): {response}\n\n"
        user_prompt += f"Full Source (containing the quote(s)): {'\n'.join(relevant_cited_sources)}\n\n"
        user_prompt += "Do the quote(s) support every single claim in the sentence? Answer '1' if the quote(s) support all claims in the sentence, or '0' if the quote(s) fail to support part of the sentence.\n"
        user_prompt += "Do not explain your answer, just return '1' or '0'.\n"
        user_prompt += "Answer:"
        raw_answer = await self.prompt_model(user_prompt, ['0', '1'])
        answer_cleaned = re.sub(r"[^\w]", "", raw_answer)
        answer = int(answer_cleaned) if answer_cleaned in ["0", "1"] else 0
        return answer
    
    async def evaluate_response_citation_precision(self, citation_dict, sent_ls, cited_source_ls, source_ls):
        citation_number_to_cited_quote_mapping = self.get_citation_number_to_cited_quote_mapping(cited_source_ls)

        full_response = ' '.join(sent_ls)
        precise_citations = []
        t2v_precision = []
        # Save precision in the same format as the mturk evals: [{"annotations":[1,1,0,1,1,1],"sentence_id":0},{"annotations":[1,1],"sentence_id":1}]
        # For each sentence, evaluate the precision of its citations
        citation_dict_key_ls = list(citation_dict.keys())
        for j in range(len(citation_dict_key_ls)):
            sentence_key = citation_dict_key_ls[j]
            citation_numbers = citation_dict[sentence_key]['citation_numbers']
            precise_citations.append({'annotations': [], 'sentence_id': sentence_key})
            for citation_number in citation_numbers:
                if str(citation_number) not in citation_number_to_cited_quote_mapping:
                    cited_quote = "" 
                else:
                    cited_quote = citation_number_to_cited_quote_mapping[str(citation_number)]
                # identify the source snippet that contains this quote
                relevant_cited_source = ""
                for source in source_ls:
                    if cited_quote in source:
                        relevant_cited_source = source
                        break
                
                cited_sentence = sent_ls[j]
                precision_start_time = time.time()
                precision = await self.evaluate_citation_precision(cited_sentence, full_response, cited_quote, relevant_cited_source)
                t2v_precision.append(time.time() - precision_start_time)
                precise_citations[j]['annotations'].append(precision)

        return precise_citations, t2v_precision
    
    def get_citation_number_to_cited_quote_mapping(self, cited_source_ls):
        # Extract the cited quote from the cited_source_ls
        source_quotes = []
        for cited_source in cited_source_ls:

            # first, replace all highlights with one highlight color
            colors = ['\x1b[92m', '\x1b[96m', '\x1b[95m', '\x1b[1;31;60m', '\x1b[102m', '\x1b[1;35;40m', '\x1b[0;30;47m', '\x1b[0;33;47m', '\x1b[0;34;47m', '\x1b[0;31;47m']
            for color in colors:
                cited_source = cited_source.replace(color, '\x1b[92m')

            pattern = r'\x1b\[\d{1,2}m\[\d+\].*?\x1b\[0m'
            source_quotes.extend(re.findall(pattern, cited_source, flags=re.DOTALL))

        citation_number_to_cited_quote_mapping = {}

        for i in range(len(source_quotes)):
            source_quotes[i] = source_quotes[i].replace('\x1b[0m', '')
            source_quotes[i] = re.sub(r'\x1b\[\d{1,2}m', '', source_quotes[i])
            citation_number_match = re.match(r'\[(\d+)\]', source_quotes[i])
            curr_citation_marker = citation_number_match.group(0) # string with bracket
            curr_citation_number = citation_number_match.group(1) # integer as string

            citation_number_to_cited_quote_mapping[curr_citation_number] = source_quotes[i][len(curr_citation_marker):].strip()
            
            # Handle nested duplicate quotes: If the start of the quote begins with another citation marker, then remove it and also add the quote under that other number (if that number isn't already present in the dict)
            if re.match(r'^\[\d+\]', citation_number_to_cited_quote_mapping[curr_citation_number][:5]): # if there is another citation marker at the start
                other_citation_number = re.match(r'^\[(\d+)\]', citation_number_to_cited_quote_mapping[curr_citation_number][:5]).group(1) # integer as string
                citation_number_to_cited_quote_mapping[curr_citation_number] = citation_number_to_cited_quote_mapping[curr_citation_number][2+len(other_citation_number):].strip()
                if other_citation_number not in citation_number_to_cited_quote_mapping:
                    citation_number_to_cited_quote_mapping[other_citation_number] = citation_number_to_cited_quote_mapping[curr_citation_number]
        return citation_number_to_cited_quote_mapping

    async def evaluate_response_citation_coverage(self, citation_dict, sent_ls, cited_source_ls, source_ls, verbose=False):
        # Extract the cited quotes from the cited_source_ls
        citation_number_to_cited_quote_mapping = self.get_citation_number_to_cited_quote_mapping(cited_source_ls)

        full_response = ' '.join(sent_ls)
        sentence_coverage = []
        t2v_coverage = []
        # Save coverage in the same format as the mturk evals (-1 if no citations): [{"coverage":-1,"sentence_id":0},{"coverage":1,"sentence_id":1},{"coverage":-1,"sentence_id":2}]
        # For each sentence, evaluate the coverage given the citations
        citation_dict_key_ls = list(citation_dict.keys())
        for j in range(len(citation_dict_key_ls)):
            start_time = time.time()
            sentence_key = citation_dict_key_ls[j]
            citation_numbers = citation_dict[sentence_key]['citation_numbers']

            if (len(citation_numbers) == 0):
                sentence_coverage.append({'coverage': -1, 'sentence_id': sentence_key})
                continue
            
            cited_quotes = []
            for citation_number in citation_numbers:
                if str(citation_number) not in citation_number_to_cited_quote_mapping:
                    cited_quote = "" # Nested citations
                else:
                    cited_quote = citation_number_to_cited_quote_mapping[str(citation_number)]
                # identify the source snippet that contains this quote
                cited_quotes.append(cited_quote)
            
            relevant_cited_sources = []
            for source in source_ls:
                for cited_quote in cited_quotes:
                    if cited_quote in source:
                        relevant_cited_sources.append(source)
                        break

            cited_sentence = sent_ls[int(sentence_key)]
            coverage = await self.evaluate_citation_coverage(cited_sentence, full_response, cited_quotes, relevant_cited_sources)

            sentence_coverage.append({'coverage': coverage, 'sentence_id': sentence_key})
            time_elapsed = time.time() - start_time
            t2v_coverage.append(time_elapsed)

        return sentence_coverage, t2v_coverage

def evaluate_quote_coverage(response, sources):
    "Returns both the number of quoted words, the number of total words, and the total number of quotes"
    fragments = response.split('\"')
    quotes = fragments[1:-1:2]
    num_quotes = len(quotes)
    num_quoted_words = 0
    num_words = len(response.split(' '))
    for quote in quotes:
        precise_quote = False
        if not quote:
            continue
        if ((quote[-1]=='.') or (quote[-1]==',')):
            quote = quote[:-1]
        for source in sources:
            smushed_source = re.sub(r'\s', '', source)
            smushed_quote = re.sub(r'\s', '', quote)
            if (smushed_quote.lower() in smushed_source.lower()):
                precise_quote = True
                break
        if (precise_quote):
            num_quoted_words += len(quote.split(' '))

    return num_quoted_words, num_words, num_quotes

def evaluate_quote_precision(response, sources):
    fragments = response.split('\"')
    quotes = fragments[1:-1:2]
    num_quotes = len(quotes)
    num_precise_quotes = 0
    for quote in quotes:
        precise_quote = False
        if not quote:
            continue
        if ((quote[-1]=='.') or (quote[-1]==',')):
            quote = quote[:-1]
        for source in sources:
            smushed_source = re.sub(r'\s', '', source)
            smushed_quote = re.sub(r'\s', '', quote)
            if (smushed_quote.lower() in smushed_source.lower()):
                precise_quote = True
                break
        if (precise_quote):
            num_precise_quotes += 1

    return num_precise_quotes, num_quotes

def get_n_grams(text, n):
    tokens = text.lower().split(' ') # make lowercase
    ngrams = []
    for i in range(0, len(tokens)-n+1):
        curr_ngram = ' '.join(tokens[i:i+n])
        if (len(curr_ngram) == 0):
            continue
        if (curr_ngram[-1] in [',', '.', '?', '!']): # remove end punctuation
            curr_ngram = curr_ngram[:-1]
        ngrams.append(curr_ngram)
    return ngrams

def num_common_ngrams(text_ngrams, source_ngrams):
    count = 0    
    for i in range(len(text_ngrams)):
        tng = text_ngrams[i]
        if tng in source_ngrams:
            count += 1
    return count

def eval_n_gram_precision(response, sources_ls, n):
    # w.r.t. multiple sources!
    response_ngrams = get_n_grams(response, n)
    all_source_ngrams = []
    for source in sources_ls:
        if (source):
            all_source_ngrams.extend(get_n_grams(source, n))
    count = num_common_ngrams(response_ngrams, all_source_ngrams)
    return (count, len(response_ngrams))

async def old_op_autoUF(results_dir, results_file):

    start_time = time.time()
    evaluator = AutoEvaluator()

    # Open up results file and read in jsonl results
    results_df = pd.read_json(results_dir+'/'+results_file, lines=True)

    op_strs = ['Snippet', 'Quoted', 'Paraphrased', 'Entailed', 'Abstractive']
    op_to_fluency_scores_mapping = {'Snippet': [], 
                                    'Quoted': [], 
                                    'Paraphrased': [], 
                                    'Entailed': [], 
                                    'Abstractive': []}
    
    op_to_perceived_utility_scores_mapping = {'Snippet': [], 
                                    'Quoted': [], 
                                    'Paraphrased': [], 
                                    'Entailed': [], 
                                    'Abstractive': []}
    
    semaphore = asyncio.Semaphore(16)  # Limit to 16 concurrent tasks
    async def sem_task(query, response, op_str, i):
        async with semaphore:
            fluency_score = await evaluator.evaluate_fluency(query, response)
            perceived_utility_score = await evaluator.evaluate_perceived_utility(query, response)
            return fluency_score, perceived_utility_score, op_str, i
        
    queries_to_send = []
    responses_to_send = []
    ops_to_send = []
    i_to_send = []

    for i in tqdm(range(len(results_df))):
        query = results_df['Question'].iloc[i]

        for op_str in op_strs:
            response = results_df[f"{op_str} Output"].iloc[i]
            if op_str == 'Quoted':
                # Strip out quotation marks for fluency and perceived utility evals
                response = response.replace('"', '').replace("'", "")
            elif op_str == 'Snippet':
                # Join the snippet response list into a single string
                response = '\n'.join(response)

            queries_to_send.append(query)
            responses_to_send.append(response)
            ops_to_send.append(op_str)
            i_to_send.append(i)

    results = await asyncio.gather(*[sem_task(query, response, op_str, i) for (query, response, op_str, i) in zip(queries_to_send, responses_to_send, ops_to_send, i_to_send)], return_exceptions=True)

    # Sort the results to ensure we store the correct score for each row
    results = sorted(results, key=lambda x: x[3])

    print("total time taken for evaluation: %s seconds" % (time.time() - start_time))

    for result in results:
        fluency_score, perceived_utility_score, op_str, _ = result
        op_to_fluency_scores_mapping[op_str].append(fluency_score)
        op_to_perceived_utility_scores_mapping[op_str].append(perceived_utility_score)
  
    for op_str in op_strs:
        results_df[f"{op_str} Fluency Score"] = op_to_fluency_scores_mapping[op_str]
        results_df[f"{op_str} Perceived Utility Score"] = op_to_perceived_utility_scores_mapping[op_str]
        print(f"Average fluency score for {op_str}: {np.mean(results_df[f"{op_str} Fluency Score"])}")
        print(f"Average perceived utility score for {op_str}: {np.mean(results_df[f"{op_str} Perceived Utility Score"])}")

    results_df.to_json(results_dir+'/'+results_file, orient='records', lines=True)


def format_remove_highlights(output):
    colors = ['\x1b[92m', '\x1b[96m', '\x1b[95m', '\x1b[1;31;60m', '\x1b[102m', '\x1b[1;35;40m', '\x1b[0;30;47m', '\x1b[0;33;47m', '\x1b[0;34;47m', '\x1b[0;31;47m', '\x1b[0m']
    for color in colors:
        output = output.replace(color, '')
    return output

async def auto_eval(results_dir, results_filename):
    overall_start_time = time.time()
    evaluator = AutoEvaluator()

    # Open up results file and read in csv results
    results_df = pd.read_csv(f"{results_dir}/{results_filename}.csv")

    queries_to_send = []
    responses_to_send = []
    citation_dict_to_send = []
    sent_ls_to_send = []
    cited_source_ls_to_send = []
    source_ls_to_send = []
    ops_to_send = []
    i_to_send = []

    for i in range(len(results_df)):
        queries_to_send.append(results_df['Question'].iloc[i])
        op_str = results_df['op'].iloc[i]
        ops_to_send.append(op_str)
        i_to_send.append(results_df['ID'].iloc[i])

        if op_str != "Snippet":
            curr_response = results_df['Output'].iloc[i]
            responses_to_send.append(curr_response)
            citation_dict_to_send.append(ast.literal_eval(results_df[f"Citation Dict"].iloc[i]))
            sent_ls_to_send.append(ast.literal_eval(results_df[f"Sent"].iloc[i])) 
            cited_source_ls_to_send.append(ast.literal_eval(results_df[f"Used Sources (cited)"].iloc[i]))
            if op_str in ["Post Hoc", "Gemini"]:
                cited_used_source_ls = ast.literal_eval(results_df[f"Used Sources (cited)"].iloc[i])
                all_source_ls = []
                for cited_s in cited_used_source_ls:
                    s = '\n'.join(cited_s.split('\n')[1:])
                    all_source_ls.append(s)
                source_ls_to_send.append(all_source_ls)
            else:
                source_ls_to_send.append(ast.literal_eval(results_df['All Sources'].iloc[i]))

        else:
            # Join the snippet response list into a single string
            raw_response = results_df['Output'].iloc[i]
            if isinstance(raw_response, str):
                raw_response = ast.literal_eval(raw_response)
                
            response = '\n'.join(raw_response)
            response = format_remove_highlights(response)
            responses_to_send.append(response)
            citation_dict_to_send.append(None)
            sent_ls_to_send.append(None)
            cited_source_ls_to_send.append(None)
            source_ls_to_send.append(None)
        
    semaphore = asyncio.Semaphore(5)  
    async def sem_task(query, response, citation_dict, sent_ls, cited_source_ls, source_ls, op_str, i):
        async with semaphore:
            fluency_scores = await evaluator.evaluate_fluency(query, response)
            perceived_utility_scores = await evaluator.evaluate_perceived_utility(query, response)
            if op_str != "Snippet":
                precision_scores, t2v_precision = await evaluator.evaluate_response_citation_precision(citation_dict, sent_ls, cited_source_ls, source_ls)
                coverage_scores, t2v_coverage = await evaluator.evaluate_response_citation_coverage(citation_dict, sent_ls, cited_source_ls, source_ls)

            else:
                precision_scores = None
                coverage_scores = None
                t2v_coverage = None
                t2v_precision = None
            return fluency_scores, perceived_utility_scores, precision_scores, coverage_scores, t2v_coverage, t2v_precision, op_str, i
    results = await asyncio.gather(*[sem_task(query, response, citation_dict, sent_ls, cited_source_ls, source_ls, op_str, i) for (query, response, citation_dict, sent_ls, cited_source_ls, source_ls, op_str, i) in zip(queries_to_send, responses_to_send, citation_dict_to_send, sent_ls_to_send, cited_source_ls_to_send, source_ls_to_send, ops_to_send, i_to_send)], return_exceptions=True)


    # Make a dataframe to store the auto-eval results and merge into the original results_df
    fluency_score_ls = []
    perceived_utility_score_ls = []
    precision_scores_ls = []
    coverage_scores_ls = []
    t2v_coverage_ls = []
    t2v_precision_ls = []
    op_str_ls = []
    id_ls = []

    for result in results:
        if isinstance(result, Exception):
            print("Error during evaluation:", repr(result))
            fluency_score_ls.append(None)
            perceived_utility_score_ls.append(None)
            precision_scores_ls.append(None)
            coverage_scores_ls.append(None)
            t2v_coverage_ls.append(None)
            t2v_precision_ls.append(None)
            op_str_ls.append(None)
            id_ls.append(None)
            continue

        fluency_score, perceived_utility_score, precision_scores, coverage_scores, t2v_coverage, t2v_precision, op_str, id = result
        fluency_score_ls.append(fluency_score)
        perceived_utility_score_ls.append(perceived_utility_score)
        precision_scores_ls.append(precision_scores)
        coverage_scores_ls.append(coverage_scores)
        t2v_coverage_ls.append(t2v_coverage)
        t2v_precision_ls.append(t2v_precision)
        op_str_ls.append(op_str)
        id_ls.append(id)

    auto_eval_dict = {
        'auto_fluency_rating': fluency_score_ls,
        'auto_utility_rating': perceived_utility_score_ls,
        'auto_precise_citations': precision_scores_ls,
        'auto_is_covered': coverage_scores_ls,
        'auto_t2v_coverage': t2v_coverage_ls,
        'auto_t2v_precision': t2v_precision_ls,
        'op': op_str_ls,
        'ID': id_ls
    }

    auto_eval_df = pd.DataFrame(auto_eval_dict)

    merged_results_df = results_df.merge(auto_eval_df, on=['op', 'ID'])

    print("Total time taken for evaluation: %s seconds" % (time.time() - overall_start_time))
    save_path = f"{results_dir}/autoEval_{results_filename}.csv" 
    print("Results saved to: ", save_path)
    merged_results_df.to_csv(save_path, index=False) 
    
    op_strs = merged_results_df['op'].unique()
    print()
    print("Fluency")
    for op_str in op_strs:
        print(f"{op_str}: {np.mean(merged_results_df[merged_results_df['op'] == op_str]['auto_fluency_rating'])}")

    print()

    print("Perceived Utility")
    for op_str in op_strs:
        print(f"{op_str}: {np.mean(merged_results_df[merged_results_df['op'] == op_str]['auto_utility_rating'])}")

    print()

    print("Citation Coverage")
    for op_str in op_strs:
        if op_str == "Snippet":
            continue
        flattened_coverage = []
        coverage_results = merged_results_df[merged_results_df['op'] == op_str]['auto_is_covered']
        for i in range(len(coverage_results)):
            coverage_results_i = coverage_results.iloc[i]
            for sent_cov in coverage_results_i:
                if sent_cov['coverage'] == -1:
                    flattened_coverage.append(0)
                else:
                    flattened_coverage.append(sent_cov['coverage'])
        print(f"{op_str}: {np.mean(flattened_coverage)}")

    print()

    print("Citation Precision")
    for op_str in op_strs:
        if op_str == "Snippet":
            continue
        flattened_precision = []
        precision_results = merged_results_df[merged_results_df['op'] == op_str]['auto_precise_citations']
        for i in range(len(precision_results)):
            precision_results_i = precision_results.iloc[i]
            for sent_prec in precision_results_i:
                flattened_precision.extend(sent_prec['annotations'])
        print(f"{op_str}: {np.mean(flattened_precision)}")

    print()

    print("T2V Coverage")
    for op_str in op_strs:
        if op_str == "Snippet":
            continue
        t2v_coverage_results = merged_results_df[merged_results_df['op'] == op_str]['auto_t2v_coverage']
        all_times = []
        for i in range(len(t2v_coverage_results)):
            t2v_coverage_results_i = t2v_coverage_results.iloc[i]
            all_times.extend(t2v_coverage_results_i)
        print(f"{op_str}: {np.mean(all_times)} seconds")

    print()

    print("T2V Precision")
    for op_str in op_strs:
        if op_str == "Snippet":
            continue
        t2v_precision_results = merged_results_df[merged_results_df['op'] == op_str]['auto_t2v_precision']
        all_times = []
        for i in range(len(t2v_precision_results)):
            t2v_precision_results_i = t2v_precision_results.iloc[i]
            all_times.extend(t2v_precision_results_i)
        print(f"{op_str}: {np.mean(all_times)} seconds")

    print()
    print('-------------------------------------------------------')
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_filename', default='nq_mturk3_0_180_byQueryOP', type=str) 
    parser.add_argument('--results_dir', default='autoEval_results', type=str) 
    args = parser.parse_args()
    
    evaluator = AutoEvaluator()
    print(args.results_filename)
    print()
    asyncio.run(auto_eval(args.results_dir, args.results_filename))

    print()
    print()

# python eval.py --results_dir autoEval_results --results_file gpt5_nq_byQueryOP
    
