import os
import json
import argparse
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, ParserRejectedMarkup
from openai import OpenAI
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait

from pathlib import Path
import urllib.request
import time

from instructions import *
from utils import *
from naturalQuestions import NaturalQuestions
from wikiMultiHopQA import WikiMultiHopQA
from medicalQA import MedicalQA
from mashQA import MashQA

BACKBONE_MODEL = OpenAI()
COLORS = {0:'\033[92m', 1:'\033[96m', 2:'\033[95m', 3:'\033[1;31;60m', 4:'\033[102m', 5:'\033[1;35;40m', 6:'\033[0;30;47m', 7:'\033[0;33;47m', 8:'\033[0;34;47m', 9:'\033[0;31;47m', 10:'\033[0m', 11:'\033[1m'}

def html_to_text(html):
    try:
        soup = BeautifulSoup(html, features="html.parser")
    except ParserRejectedMarkup:
        print('Caught bs4.builder.ParserRejectedMarkup')
        return None
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    text = standardize_quotation_marks(text)
    return text

def hashCode(query):
    hash = 0
    if (len(query) == 0):
        return hash
    for i in range(len(query)):
        char_i = query[i]
        unicode_i = ord(char_i)
        hash = ((hash << 5) - hash) + unicode_i
        hash = hash % (2**32)
    return hash

def scrape_response(soup): # if span class contains citation
    response_text_ls = [x.get_text() for x in soup.find_all(['p', 'li'])][1:-2]
    return ' '.join(response_text_ls).strip()

def scrape_gemini_html_for_citation(soup, i, cited_response_spans, cited_source_spans, cited_source_urls):
    cited_response_text = ''.join([x.get_text() for x in soup.find_all('span', class_='citation-'+str(i))])
    cited_response_spans[i+1] = cited_response_text
    cited_source_text = soup.find_all('a', class_='link-container')[0].get_text().strip()
    similar_content_str = 'Google Search found similar content, like this: '
    if (similar_content_str in cited_source_text):
        cited_source_text = cited_source_text.replace(similar_content_str, '')
    if (('High Confidence Response:' in cited_source_text) and ('Context:' in cited_source_text)):
        cited_source_text = cited_source_text.split('Context: ')[1]
    cited_source_text = cited_source_text.strip()
    cited_source_spans[i+1] = cited_source_text
    cited_source_link = soup.find_all('a', class_='link-container')[0]['href']
    cited_source_urls[i+1] = cited_source_link
    return cited_response_spans, cited_source_spans, cited_source_urls

def standardize_quotations_apostrophes(text):
    text = text.replace('“', '\'')
    text = text.replace('”', '\'')
    text = text.replace('\"', '\'')
    text = text.replace('’', '\'')
    text = text.replace('‚Äú', '\'')
    text = text.replace('‚Äù', '\'')
    text = text.replace('‚Äì', '–')
    return text

def clean_wikipedia_citations(text):
    number_ls = re.findall(r'[0-9]+', text)
    strings_to_remove = ['[edit]']
    for number in number_ls:
        strings_to_remove.append('['+str(number)+']')
    for x in strings_to_remove:
        text = text.replace(x, '')
    return text

def get_larger_source_snippet(url, source_quote):
    if ((len(source_quote)>=3) and (source_quote[-3:]=='...')):
        source_quote = source_quote[:-4]

    if ((len(source_quote)>=3) and (source_quote[:3]=='...')):
        source_quote = source_quote[4:]

    source_quote = standardize_quotations_apostrophes(source_quote)
    
    source_quote = source_quote.replace('\n-', '') # remove bullet points
    source_quote = source_quote.replace('\n', ' ') 
    print()
    print('>>>>>>>>>>>>>>> source quote: '+source_quote)
    print()
    try:
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', user_agent)]
        urllib.request.install_opener(opener)
        req = urllib.request.Request(
                                    url, 
                                    data=None, 
                                    headers={
                                        'User-Agent': user_agent
                                    }
                                )
        with urllib.request.urlopen(req) as url_response:
            html = url_response.read().decode('utf-8')
    except:
        html = None

    if (html == None):
        try:
            driver = webdriver.Chrome()
            driver.set_page_load_timeout(20)
            time.sleep(10)
            driver.get(url)
            html = driver.page_source
            driver.quit()
        except:
            print()
            print(url)
            print('!!!!!! Url would not open')
            return source_quote, source_quote, 0
    
    text = html_to_text(html)
    text = standardize_quotations_apostrophes(text)
    text = text.replace('\n', ' ').replace('\xa0', ' ')
    if ('https://en.wikipedia.org' in url):
        text = clean_wikipedia_citations(text)
    # get first sentence of source_quote
    text_idx = -1
    sentence_idx = 0
    start_idx = 0
    source_quote_sent_ls = source_quote.split('.')
    while ((sentence_idx<len(source_quote_sent_ls)) and (text_idx == -1)):
        curr_sent = source_quote_sent_ls[sentence_idx]
        if ((len(curr_sent)==0) or (len(curr_sent.split(' '))<4)):
            sentence_idx += 1
            continue
        text_idx = text.find(curr_sent)
        sentence_idx += 1
        len_of_prev_sentences = len('.'.join(source_quote_sent_ls[:sentence_idx]))
        
        
    if (text_idx >= 0):
        bottom_idx = max(text_idx-500, 0)
        top_idx = min(text_idx+500, len(text))
        source_snippet = text[bottom_idx:top_idx].strip()
        return source_snippet, source_quote, -1
    else: # this can happen when the quote is split across heading and body text 
        print()
        print('Cannot locate Gemini source quote. Trying another way...')
        return get_larger_source_snippet_by_ngrams(text, source_quote)
    
def get_larger_source_snippet_by_ngrams(text, source_quote, n=4):
    ngrams = []
    for i in range(len(source_quote)-n+1):
        ngrams.append(source_quote[i:i+n])
    text_idxs_of_ngrams = np.zeros(len(text))
    for ngram in ngrams:
        idx = text.find(ngram)
        if (idx != -1):
            text_idxs_of_ngrams[idx] = 1
    window = len(source_quote)

    # find the window with the highest number of ngram occurrences
    highest_count = 0
    window_start_idx = -1
    for i in range(len(text)-window+1):
        curr_window_start_idx = i
        curr_window_end_idx = i+window
        # get number of occurrences in this window
        curr_num_occurrences = np.sum(text_idxs_of_ngrams[curr_window_start_idx:curr_window_end_idx])
        if (curr_num_occurrences > highest_count):
            highest_count = curr_num_occurrences
            window_start_idx = curr_window_start_idx
    if ((window_start_idx == -1) or (highest_count < 15)): # if nothing is found, default to the source_quotes
        return source_quote, source_quote, 0

    bottom_idx = max(window_start_idx-500, 0)
    top_idx = min(window_start_idx+500, len(text))
    source_snippet = text[bottom_idx:top_idx].strip()
    return source_snippet, source_quote, 500

def remove_escape_sequences(text):
    return text.encode().decode('unicode_escape').encode('ascii', 'ignore').decode()

def scrape_gemini_html_for_query(query, baseline_instruction_str, html_directory_path):
    # For one query
    full_response = ''
    cited_response_spans = {} # citation number to span in responose
    cited_source_spans = {} # citation number to span in source
    cited_source_urls = {} # citation number to source url

    query = query.replace(' \u200b', '')
    # Handle hashing anomalies
    if (query == 'what is the process of amending the united states constitution & north carolina constitution'):
        hash = '3988964556'
    elif (query == 'fast & furious 8 release date in india'):
        hash = '1232989536'
    elif (query == 'Do both films Womb (film) and Fast, Cheap & Out of Control have the directors that share the same nationality?'):
        hash = '1655188203'
    elif (query == 'How can I keep my child from spitting up?'):
        hash = '4272221864'
    elif (query == 'How do you treat a dog\'s sprain or  strain?'):
        hash = '439469393'
    elif ('being treated for cancer, how often should I wash my hands?' in query):
        hash = '3087121426'
    elif ('How do you diagnose Alzheimer' in query):
        hash = '2535232522' 
    else:
        hash = hashCode(baseline_instruction_str+query)

    for i in range(-1, 20):
        fp = html_directory_path+'/'+str(hash)+'_'+str(i)+'.txt'
        if (os.path.exists(fp)):
            with open(fp, 'rb') as f:
                html = f.read()
            soup = BeautifulSoup(html, features="html.parser")
            if (i == -1):
                full_response = scrape_response(soup)
            else:
                cited_response_spans, cited_source_spans, cited_source_urls = scrape_gemini_html_for_citation(soup, i, cited_response_spans, cited_source_spans, cited_source_urls)
        else:
            continue
    return full_response, cited_response_spans, cited_source_spans, cited_source_urls

def save_gemini_results_for_queries(data, start_n, n, save_fp, baseline_instruction_str, html_directory_path, data_str, debug=False):
    instance_ls = np.arange(start_n, start_n+n)
    for i in instance_ls:
        query = data[i]['question'].strip()
        abstention_type = 'No Failure'
        if (data_str == 'eli5_nq'):
            query = 'Explain to a third-grader: '+query

        full_response, cited_response_spans, cited_source_spans, cited_source_urls = scrape_gemini_html_for_query(query, baseline_instruction_str, html_directory_path)
        if (full_response == ''):
            breakpoint()
        response, cited_response, response_sentences, cited_response_sentences, cited_sources, citations_dict = format_gemini_results(full_response, cited_response_spans, cited_source_spans, cited_source_urls)
        
        answer_is_abstained = is_abstained_gpt4(query, full_response)
        if (answer_is_abstained == 'True'):
            abstention_type = 'Generation Failure'
        results_keys = ["ID", 
                "Used Sources (cited)", 
                "Question", 
                "Output (cited)", 
                "Output", 
                "Sent (cited)", 
                "Sent", 
                "Citation Dict", 
                "op",
                "Abstention Type",
                ]
        results_values = [int(i),
                          cited_sources,
                          query,
                          cited_response,
                          response,
                          cited_response_sentences,
                          response_sentences,
                          citations_dict,
                          "Gemini",
                          abstention_type
                          ]
        # open file and write another line
        data_dict = dict(zip(results_keys, results_values))
        print(data_dict['ID'])
        print()
        print(data_dict['Question'])
        print()
        print(data_dict['Output (cited)'])
        print()

        for s in data_dict['Used Sources (cited)']:
            print(s)
            print()
        print()
        print(data_dict["Citation Dict"])
        print()
        print(data_dict["op"])
        print()
        print(abstention_type)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`')
        if (not debug):
            if (not os.path.exists(save_fp)):
                with open(save_fp, "w") as f:
                    json_string = json.dumps(data_dict)  # Convert item to JSON string
                    f.write(json_string + "\n")  # Write with newline character
            else:
                with open(save_fp, "a") as f:
                    json_string = json.dumps(data_dict)  # Convert item to JSON string
                    f.write(json_string + "\n")  # Write with newline character
            print('Saved to '+save_fp)
    return    
            

def format_gemini_results(full_response, cited_response_spans, cited_source_spans, cited_source_urls):
    response_sentences = get_sentences_gpt4(full_response, BACKBONE_MODEL)
    cited_response_sentences = []
    citations_dict = {} # maps sentence number to citation
    # unmatched_cited_response_spans = copy.deepcopy(cited_response_spans) 
    for i in range(len(response_sentences)):
        curr_response_sentence = response_sentences[i]
        # find the citation
        citation_str = ' '
        citation_num_ls = []
        for k in cited_response_spans.keys():
            candidate_response_span = cited_response_spans[k]
            # split candidate_response_span into sentences
            candidate_response_span_sentences = get_sentences_gpt4(candidate_response_span, BACKBONE_MODEL)
            for candidate_response_span_sentence in candidate_response_span_sentences:
                if (candidate_response_span_sentence in curr_response_sentence):
                    citation_str += COLORS[0]+"["+str(k)+"]"+COLORS[10]
                    citation_num_ls.append(k)
                    # unmatched_cited_response_spans.pop(k)
        if (len(citation_str) == 1):
            citation_str = ''
        
        cited_response_sentence = curr_response_sentence[:-1]+citation_str+curr_response_sentence[-1]
        cited_response_sentences.append(cited_response_sentence)
        citations_dict[i] = {'citation_numbers': citation_num_ls}
    
    cited_response = ' '.join(cited_response_sentences)
    response = ' '.join(response_sentences)
    cited_sources = []
    for k in cited_source_spans.keys():
        url = cited_source_urls[k]
        source_quote = cited_source_spans[k]

        # get larger span
        source, cleaned_source_quote, start_idx = get_larger_source_snippet(url, source_quote)
        cited_source_sent_idx = -1
        sentence_idx = 0
        source_quote_sent_ls = cleaned_source_quote.split('.')

        if (start_idx == -1): # if the span was found by sentence
            while ((sentence_idx<len(source_quote_sent_ls)) and (cited_source_sent_idx == -1)):
                curr_sent = source_quote_sent_ls[sentence_idx]
                if ((len(curr_sent)==0) or (len(curr_sent.split(' '))<4)):
                    sentence_idx += 1
                    continue
                cited_source_sent_idx = source.find(curr_sent)
                cited_source_span_idx = cited_source_sent_idx-len('.'.join(source_quote_sent_ls[:sentence_idx]))
                cited_source_span_end_idx = cited_source_sent_idx+len('.'.join(source_quote_sent_ls[sentence_idx:]))
                sentence_idx += 1
        else: # if the span was found by ngram
            cited_source_span_idx = start_idx
            cited_source_span_end_idx = start_idx + len(cleaned_source_quote)

        cited_source = source[0:cited_source_span_idx]+COLORS[0]+"["+str(k)+"]"+" "+source[cited_source_span_idx:cited_source_span_end_idx]+COLORS[10]+source[cited_source_span_end_idx:]
        cited_source = url+"\n"+cited_source
        cited_sources.append(cited_source)

    return response, cited_response, response_sentences, cited_response_sentences, cited_sources, citations_dict

def save_results_for_sl(args):
    # Load results file as pd df and save as a csv
    results_fp = 'generation_results/'+args.project_name+'.jsonl'
    with open(results_fp, "r") as f:
        results = [json.loads(line) for line in f]
    df = pd.DataFrame(results)
    print()
    # Report failures of the Vertex API citing system (already excluded from the dataset)
    print('Gemini/scraping failure rate:', 1-(len(df)/args.n))
    print()
    # Save as a csv
    save_path = 'generation_results/'+args.project_name+'_0_'+str(len(df))+'_byQueryGemini.csv'
    df.to_csv(save_path)
    print('Saved csv for annotation to '+save_path)

def main(args):
    if ((args.data == 'nq') or (args.data == 'eli5_nq')):
        data = NaturalQuestions(seed=0)
        baseline_instruction_str = nq_baseline_instruction_str
    elif (args.data == 'medical'):
        data = MedicalQA(seed=0)
        print('Dataset baseline_instruction_str not implemented')
        return
    elif (args.data == 'multihop'):
        data = WikiMultiHopQA(seed=0)
        baseline_instruction_str = mh_baseline_instruction_str
    elif (args.data == 'mash'):
        data = MashQA(seed=0)
        baseline_instruction_str = mash_baseline_instruction_str
    save_fp = 'generation_results/'+args.project_name+'.jsonl'
    save_gemini_results_for_queries(data, args.start_n, args.n, save_fp, baseline_instruction_str, args.html_directory_path, args.data, args.debug)
    save_results_for_sl(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--start_n', type=int, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--html_directory_path', type=str, required=True)
    parser.add_argument('--data', type=str, required=True) # 'nq' or 'medical' or 'multihop' or 'emr' or 'eli5_nq'or 'mash'
    args = parser.parse_args()
    main(args)