import re
import copy 
import os
from few_shot_examples import *
from instructions import *
import nltk
import numpy as np
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from instructions import is_abstained_instruction_str
from few_shot_examples import is_abstained_few_shot_examples_dict
from openai import OpenAI
import global_vars

COLORS = {0:'\033[92m', 1:'\033[96m', 2:'\033[95m', 3:'\033[1;31;60m', 4:'\033[102m', 5:'\033[1;35;40m', 6:'\033[0;30;47m', 7:'\033[0;33;47m', 8:'\033[0;34;47m', 9:'\033[0;31;47m', 10:'\033[0m', 11:'\033[1m'}

punkt_param = PunktParameters()
abbreviation = ['u.s.a', 'fig', 'u.s', 'i.e', 'e.g', 'etc', 'c', 'ca', 'ms', 'mrs', 'mr', 'vs', 'lt', 'sgt', 'col', 'maj', 'capt', 'cpt', 'a.x', 'no', 'op', 'dr']
punkt_param.abbrev_types = set(abbreviation)
SENT_TOKENIZER = PunktSentenceTokenizer(punkt_param)
BACKBONE_MODEL = OpenAI()

def is_abstained_gpt4(query, response):
    i = 0
    answer = None
    while ((answer not in ['True', 'False']) and (i<3)):
        prompt = construct_prompt(is_abstained_instruction_str, is_abstained_few_shot_examples_dict, ['Query: '+query, 'Response: '+response], 'Answer: ')
        answer, _ = generate_from_model(BACKBONE_MODEL, prompt)
        i+= 1
    return answer

def standardize_quotation_marks(text):
    text = text.replace('“', '\'')
    text = text.replace('”', '\'')
    text = text.replace('\"', '\'')
    return text

def format_remove_quotation_marks(output):
    return output.replace('\"', '')

def format_remove_highlights(output):
    colors = ['\x1b[92m', '\x1b[96m', '\x1b[95m', '\x1b[1;31;60m', '\x1b[102m', '\x1b[1;35;40m', '\x1b[0;30;47m', '\x1b[0;33;47m', '\x1b[0;34;47m', '\x1b[0;31;47m', '\x1b[0m']
    for color in colors:
        output = output.replace(color, '')
    return output

def generate_from_model(backbone_model, prompt, model_str="gpt-4-0125-preview"):
    message=[{"role": "system", "content": "You are a helpful expert who follows instructions closely."}, {"role": "user", "content": prompt}]
    temperature=0.2
    max_tokens=512

    response = backbone_model.chat.completions.create(
            model=model_str, 
            messages = message,
            temperature=temperature,
            max_tokens=max_tokens,
    )
    details_dict = {'instruction': prompt, 
                    'tokens_used': response.usage.total_tokens, 
                    'max_tokens':max_tokens, 
                    'temperature':temperature}

    return response.choices[0].message.content, details_dict

def format_source_list(source_ls):
    source_str = ""
    if (isinstance(source_ls, list)):
        for s in source_ls:
            source_str = source_str+'\"'+s+'\"\n'
    else:
        source_str = '\"'+source_ls+'\"\n' # source_ls is just a string for the medQA gold_reference
    return source_str

def find_all_indexes(source, quote_substring):
    return [m.start() for m in re.finditer(re.escape(quote_substring), source)]

def find_smushed_quote_idxs_in_source(quote, source, window):
    rolling_search_idx = 0
    source_start_idx = -1
    while ((rolling_search_idx+window < len(quote.lower())) and (source_start_idx == -1)):
        quote_substring = quote.lower()[rolling_search_idx:rolling_search_idx+window]
        num_occurrences = source.lower().count(quote_substring)
        if (num_occurrences == 1):
            source_start_idx = source.lower().find(quote_substring)
        elif (num_occurrences > 1):
            all_idxs = find_all_indexes(source.lower(), quote_substring)
            for idx in all_idxs:
                smushed_source_sample = re.sub(r'\s', '', source.lower()[idx:idx+window+20])
                smushed_quote = re.sub(r'\s', '', quote.lower())
                if (smushed_source_sample in smushed_quote):
                    source_start_idx = idx
        else: 
            source_start_idx = -1 # keep searching
        rolling_search_idx += 5
    return source_start_idx

def identify_and_cite_quote(quotes_ls, i, response_sentence, response_sentence_i, citation_number, source, highlighted_source, highlighted_uncited_source, sentences_to_citations):
    quote = quotes_ls[response_sentence_i][i]
    response_sentence = response_sentence[response_sentence_i]
    smushed_source = re.sub(r'\s', '', source)
    smushed_quote = re.sub(r'\s', '', quote)
    citation_number_str = '['+str(citation_number)+']'
    sentences_to_citations[response_sentence_i] = {'citation_numbers': []}
    if (quote.lower() in source.lower()):
        response_sentence = response_sentence[:-1]+' '+COLORS[0]+citation_number_str+COLORS[10]+response_sentence[-1]
        sentences_to_citations[response_sentence_i]['citation_numbers'].append(citation_number)
        start_idx = source.lower().find(quote.lower())
        end_idx = start_idx+len(quote)
        source_quote = source[start_idx:end_idx]
        highlighted_source_quote = source_quote+COLORS[10]
        highlighted_source = highlighted_source.replace(source_quote, COLORS[0]+citation_number_str+' '+highlighted_source_quote) 
        highlighted_uncited_source = highlighted_uncited_source.replace(source_quote, COLORS[0]+' '+highlighted_source_quote) 
    elif (smushed_quote.lower() in smushed_source.lower()):
        approx_source_start_idx = -1
        window = 30
        while ((approx_source_start_idx == -1) and (window > 10)):
            approx_source_start_idx = find_smushed_quote_idxs_in_source(quote, source, window)
            window -= 5
        if (approx_source_start_idx == -1):
            return None, None, None, None
        response_sentence = response_sentence[:-1]+' '+COLORS[0]+citation_number_str+COLORS[10]+response_sentence[-1]
        sentences_to_citations[response_sentence_i]['citation_numbers'].append(citation_number)
                
        approx_source_end_idx = approx_source_start_idx + len(quote)
        source_quote = source[approx_source_start_idx:approx_source_end_idx]
        highlighted_source_quote = source_quote+COLORS[10]
        highlighted_source = highlighted_source.replace(source_quote, COLORS[0]+citation_number_str+' '+highlighted_source_quote)
        highlighted_uncited_source = highlighted_uncited_source.replace(source_quote, COLORS[0]+' '+highlighted_source_quote)

    return response_sentence, highlighted_source, highlighted_uncited_source, sentences_to_citations

def highlight_direct_quotes(response, sources, backbone_model, data_str):
    
    quotes_ls = get_quoted_sentences(response, backbone_model, data_str=data_str)
    unmarked_response = format_remove_quotation_marks(response)
    if (data_str != 'mash'):
        unmarked_sentences = get_sentences_gpt4(unmarked_response, backbone_model)
    else:
        unmarked_sentences = get_sentences_tokenizer(unmarked_response)

    num_citations = 0
    # sources_idxs is a dictionary that map quote # to source #, the corresponding index slices for highlighting, and the citation number
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
        j = 0
        
        for source in sources:
            smushed_source = re.sub(r'\s', '', source)
            smushed_quote = re.sub(r'\s', '', quote)
            citation_number = '['+str(i+1)+']'
            highlighted_quote = COLORS[j]+'\"'+quote+'\"'+' '+citation_number+COLORS[10]
            if (quote.lower() in source.lower()):
                for ii in range(len(sentences)):
                    if (quote in sentences[ii]):
                        sentences[ii] = sentences[ii].replace(quote, highlighted_quote)
                        sentences_to_citations[ii]['citation_numbers'].append(i+1)
                num_citations += 1
                start_idx = source.lower().find(quote.lower())
                end_idx = start_idx+len(quote)
                source_quote = source[start_idx:end_idx]
                sources_idxs[i] = {'source_idx':j, 'citation_numbers':[citation_number], 'start':start_idx, 'end':end_idx}
                highlighted_source_quote = source_quote+COLORS[10]
                highlighted_sources[j] = highlighted_sources[j].replace(source_quote, COLORS[j]+citation_number+' '+highlighted_source_quote) 
                highlighted_uncited_sources[j] = highlighted_uncited_sources[j].replace(source_quote, COLORS[j]+' '+highlighted_source_quote) 
                used_sources[j] = None
                break
            elif (smushed_quote.lower() in smushed_source.lower()):
                num_citations += 1
                approx_source_start_idx = -1
                window = 30
                while ((approx_source_start_idx == -1) and (window > 10)):
                    approx_source_start_idx = find_smushed_quote_idxs_in_source(quote, source, window)
                    window -= 5
                if (approx_source_start_idx == -1):
                    break
                for ii in range(len(sentences)):
                    if (quote in sentences[ii]):
                        sentences[ii] = sentences[ii].replace(quote, highlighted_quote)
                        sentences_to_citations[ii]['citation_numbers'].append(i+1)
                        
                approx_source_end_idx = approx_source_start_idx + len(quote)
                source_quote = source[approx_source_start_idx:approx_source_end_idx]
                sources_idxs[i] = {'source_idx':j, 'citation_numbers':[citation_number], 'start':approx_source_start_idx, 'end':approx_source_end_idx}
                highlighted_source_quote = source_quote+COLORS[10]
                highlighted_sources[j] = highlighted_sources[j].replace(source_quote, COLORS[j]+citation_number+' '+highlighted_source_quote)
                highlighted_uncited_sources[j] = highlighted_uncited_sources[j].replace(source_quote, COLORS[j]+' '+highlighted_source_quote)
                used_sources[j] = None
                break
            j += 1

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

def get_sentences_gpt4(text, backbone_model):
    prompt = construct_prompt(parse_sent_instruction_str, parse_sent_few_shot_examples_dict, ['Text: '+text], 'Answer:\n')
    response, _ = generate_from_model(backbone_model, prompt)
    sent_ls = response.split('\n')
    return sent_ls

def get_sentences_tokenizer(text):
    return SENT_TOKENIZER.tokenize(text)

def handle_ellipses(quote):
    if ('...' in quote):
        quote_parts = quote.split('...')
        quote_part_lengths = [len(x) for x in quote_parts]
        longest_quote_part_idx = np.argmax(quote_part_lengths)
        quote = quote_parts[longest_quote_part_idx]
    return quote

def remove_end_punctuation(quote):
    if ((quote[-1]=='.') or (quote[-1]==',')):
        quote = quote[:-1]
    return quote

def get_sentences_tokenizer(text):
    sentences = SENT_TOKENIZER.tokenize(text)
    return sentences

def get_quoted_sentences(quoted_response, backbone_model, data_str='not multihop'): # avoids splitting on common abbreviations (see list at top of page)
    fragments = quoted_response.split('\"')
    quote_ls = fragments[1:-1:2]
    
    if (data_str != 'multihop'):
        quoted_sentences_ls = []
        for quote in quote_ls:
            curr_quote_sentences = get_sentences_tokenizer(quote)
            for i in range(len(curr_quote_sentences)):
                curr_quote_sentences[i] = handle_ellipses(curr_quote_sentences[i])
                curr_quote_sentences[i] = standardize_quotation_marks(curr_quote_sentences[i])
                curr_quote_sentences[i] = remove_end_punctuation(curr_quote_sentences[i])
            quoted_sentences_ls.extend(curr_quote_sentences) 
        quote_ls = quoted_sentences_ls
    else: 
        new_quote_ls = []
        for i in range(len(quote_ls)):
            if ('.' in quote_ls[i]):
                quote_ls_i = get_sentences_gpt4(quote_ls[i], backbone_model)
                for j in range(len(quote_ls_i)):
                    quote_ls_i[j] = handle_ellipses(quote_ls_i[j])
                    quote_ls_i[j] = standardize_quotation_marks(quote_ls_i[j])
                    quote_ls_i[j] = remove_end_punctuation(quote_ls_i[j])
                new_quote_ls.extend(quote_ls_i)
            else:
                quote_ls[i] = handle_ellipses(quote_ls[i])
                quote_ls[i] = standardize_quotation_marks(quote_ls[i])
                quote_ls[i] = remove_end_punctuation(quote_ls[i])
                new_quote_ls.append(quote_ls[i])
        quote_ls = new_quote_ls

    quote_ls = remove_contained_quotes(quote_ls)
    return quote_ls

def remove_contained_quotes(quoted_sentences_ls):
    sanitized_quoted_sentences_ls = []
    for i in range(len(quoted_sentences_ls)):
        curr_quote = quoted_sentences_ls[i]
        contained = False
        for j in range(len(quoted_sentences_ls)):
            if (i == j):
                continue
            j_quote = quoted_sentences_ls[j]
            if ((curr_quote != j_quote) and (curr_quote.lower() in j_quote.lower())):
                contained = True
        if (not contained):
            sanitized_quoted_sentences_ls.append(curr_quote)
    return sanitized_quoted_sentences_ls

def get_quotes(quoted_response):
    fragments = quoted_response.split('\"')
    quote_ls = fragments[1:-1:2]
    return quote_ls

    return full_sentences_of_quotes

def cite_paraphrased_quotes(quoted_response, paraphrased_response, sources, backbone_model, op, sources_idxs, data_str):

    # identify all quoted sentences
    quoted_sentences_ls = get_quoted_sentences(quoted_response, backbone_model, data_str=data_str)

    # identify all paraphrased sentences
    paraphrased_sentences = get_sentences_gpt4(paraphrased_response, backbone_model) 
    # for each sentence of the paraphrased_response, identify the quoted sentences that match the information content
    quotes_str = 'Quotes:\n'+format_quoted_sentences_ls(quoted_sentences_ls)
    cited_paraphrased_sentences = []
    num_citations = 0
    # citations_dict is a mapping from sentence to the citation numbers for that sentence
    citations_dict = {}   

    for i in range(len(paraphrased_sentences)):
        citations_dict[i] = {'citation_numbers': []}
        prompt = None
        paraphrased_sentence = paraphrased_sentences[i]
        if (len(paraphrased_sentence) == 0):
            continue
        # identify the items from quoted_sentences_ls who's information is contained  in paraphrased_sentence
        text_str = 'Text: '+paraphrased_sentence
        if (op == 'paraphrased'):
            if (data_str == 'multihop'):
                few_shot_examples_dict = id_paraphrased_mh_citations_few_shot_examples_dict
            elif ((data_str == 'nq') | (data_str == 'eli5_nq')):
                few_shot_examples_dict = id_paraphrased_citations_few_shot_examples_dict
            elif (data_str == 'mash'):
                few_shot_examples_dict = id_paraphrased_mash_citations_few_shot_examples_dict
            else:
                print('Citation not implemented for this dataset.')
                exit()
        elif (op == 'entailed'):
            if (data_str == 'multihop'):
                few_shot_examples_dict = id_entailed_mh_citations_few_shot_examples_dict
            elif ((data_str == 'nq') | (data_str == 'eli5_nq')):
                few_shot_examples_dict = id_entailed_citations_few_shot_examples_dict
            elif (data_str == 'mash'):
                few_shot_examples_dict = id_entailed_mash_citations_few_shot_examples_dict
            else:
                print('Citation not implemented for this dataset.')
                exit()
        elif (op == 'abstracted'):
            if (data_str == 'multihop'):
                few_shot_examples_dict = id_abstractive_mh_citations_few_shot_examples_dict
            elif ((data_str == 'nq') | (data_str == 'eli5_nq')):
                few_shot_examples_dict = id_abstractive_citations_few_shot_examples_dict
            elif (data_str == 'mash'):
                few_shot_examples_dict = id_abstractive_mash_citations_few_shot_examples_dict
            else:
                print('Citation not implemented for this dataset.')
                exit()
        else:
            print('Citation not implemented for this operating point.')
            exit()
        prompt = construct_prompt(id_pp_ent_abs_citations_instruction_str, few_shot_examples_dict, [text_str, quotes_str], citation_response_str)
        prompt_box = construct_pp_ent_abs_citation_prompt_box(id_pp_ent_abs_citations_instruction_str, few_shot_examples_dict, [text_str, quotes_str], citation_response_str)

        response, _ = generate_from_model(backbone_model, prompt)

        split_ls = response.split('[')
        citation_numbers = ' '
        citation_numbers_ls = []
        for s in split_ls:
            number_ls = re.findall(r'[0-9]+', s)
            if (len(number_ls)>0):
                citation_number = int(number_ls[0])
                if (sources_idxs[citation_number-1] != None): # only keep citation if the quote is precise
                    source_idx = sources_idxs[citation_number-1]['source_idx']
                    citation_numbers += COLORS[source_idx]+"["+str(citation_number)+"]"+COLORS[10]
                    citation_numbers_ls.append(citation_number)
                    num_citations += 1
        citations_dict[i]['citation_numbers'].extend(citation_numbers_ls)
        if (citation_numbers == ' '):
            citation_numbers = ''
            
        # remove the end punctuation to keep the reference inside the sentence it refers to
        end_punctuation = ''
        if ((paraphrased_sentence[-1]=='.') or (paraphrased_sentence[-1]==',') or (paraphrased_sentence[-1]=='!') or (paraphrased_sentence[-1]=='?')):
            end_punctuation = paraphrased_sentence[-1]
            paraphrased_sentence = paraphrased_sentence[:-1]
        # paraphrased_response = paraphrased_response.replace(paraphrased_sentence, paraphrased_sentence+citation_numbers)
        paraphrased_sentence = paraphrased_sentence+citation_numbers+end_punctuation
        cited_paraphrased_sentences.append(paraphrased_sentence)

    paraphrased_response = " ".join(cited_paraphrased_sentences)
    for i in range(len(cited_paraphrased_sentences)):
        cited_paraphrased_sentences[i] = cited_paraphrased_sentences[i].replace("'", "\'")
        paraphrased_sentences[i] = paraphrased_sentences[i].replace("'", "\'")
        
    return paraphrased_response, cited_paraphrased_sentences, paraphrased_sentences, num_citations, citations_dict

def format_quoted_sentences_ls(quoted_sentences_ls):
    formatted_str = ''
    for i in range(len(quoted_sentences_ls)):
        formatted_str += '['+str(i+1)+'] \"'+quoted_sentences_ls[i]+'\"\n'
    return formatted_str
