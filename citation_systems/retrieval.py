from txtai import Embeddings
from txtai.pipeline import LLM
import torch
from transformers import RagTokenizer, RagRetriever, DPRReader, DPRReaderTokenizer, RagSequenceForGeneration, DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import datasets
from pathlib import Path
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup, ParserRejectedMarkup
from rank_bm25 import BM25Okapi
from utils import standardize_quotation_marks
import numpy as np
import time 
import socket
import re
import time
import pickle
import os
from naturalQuestions import NaturalQuestions
from operating_points import get_sub_questions
from openai import OpenAI
from sentence_transformers import SentenceTransformer

GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
GOOGLE_SEARCH_ENGINE_ID = os.environ['GOOGLE_SEARCH_ENGINE_ID']
BING_API_KEY = os.environ['BING_API_KEY']
timeout = 10
socket.setdefaulttimeout(timeout)

class TxtaiRetrieval:
    def __init__(self):
        self.embeddings = Embeddings()
        self.embeddings.load(provider='huggingface-hub', container='neuml/txtai-wikipedia')

    def retrieve(self, question):
        retrieved_sources = [x["text"] for x in self.embeddings.search(question)]
        return retrieved_sources

class BM25Retrieval:
    def __init__(self):
        return 

    def retrieve(self, question, text):
        text_chunks = []
        j = 0
        while (j < len(text)):
            text_chunks.append(text[j:min(len(text), j+1000)])
            j += 1000

        tokenized_text_chunks = [chunk.split(" ") for chunk in text_chunks]
        bm25 = BM25Okapi(tokenized_text_chunks)
        tokenized_query = question.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        top_idxs = (-doc_scores).argsort()[:3] 
        top_chunks = np.array(text_chunks)[top_idxs].tolist()
        return top_chunks

class FBDPRRetrieval:
    def __init__(self):
        datasets.config.DOWNLOADED_DATASETS_PATH = Path("/u/scr/nlp/data/wiki_dpr/huggingface/datasets/downloads")
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        self.retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=False)

    def retrieve(self, question):
        input_dict = self.tokenizer.prepare_seq2seq_batch(question, return_tensors="pt")
        input_ids = input_dict["input_ids"]
        question_hidden_states = self.model.question_encoder(input_ids)[0]
        docs_dict = self.retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
        doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)
        source_ls = self.tokenizer.batch_decode(docs_dict['context_input_ids'], skip_special_tokens=True) # returns a list of length 5
        for i in range(len(source_ls)):
            s = source_ls[i]
            source_ls[i] = s.replace(question, "")
            source_ls[i] = s.replace(question.lower(), "")
        return source_ls

class BingBM25Retrieval:
    def __init__(self):
        self.search_url = "https://api.bing.microsoft.com/v7.0/search"

    def retrieve(self, question):
        texts, urls, titles = bing_retrieve(question, 5)
        all_text_chunks, chunk_url_idxs = chunk_results(texts)
        tokenized_text_chunks = [chunk.split(" ") for chunk in all_text_chunks]
        bm25 = BM25Okapi(tokenized_text_chunks)
        tokenized_query = question.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        top_idxs = (-doc_scores).argsort()[:3] 
        top_chunks = np.array(all_text_chunks)[top_idxs].tolist()
        top_chunk_urls = np.array(urls)[np.array(chunk_url_idxs)[top_idxs]]
        return top_chunks, top_chunk_urls
    
def google_retrieve(question, num_webpages, avoid_webmd, avoid_wikipedia):
    texts = []
    urls = []
    titles = []
    search_url = 'https://www.googleapis.com/customsearch/v1'

    num_searches = 0
    while (len(urls) < num_webpages):
        params = {"key": GOOGLE_API_KEY,
                  "cx": GOOGLE_SEARCH_ENGINE_ID,
                  "q": question, 
                  "start": num_searches*10,
                  "num": 10} 
        if (avoid_webmd):
            params['siteSearchFilter'] = "e"
            params['siteSearch'] = "www.webmd.com"
        if (avoid_wikipedia):
            params['siteSearchFilter'] = "e"
            params['siteSearch'] = "en.wikipedia.org"

        response = requests.get(search_url, params=params)

        if (response.status_code != 200):
            print('Response status code is:', str(response.status_code))
            raise Exception('Google search failed :(')
        
        response_json = response.json()
        
        num_searches += 1
        i = 0

        if ('items' not in response_json.keys()):
            print('Google fail. Using as many webpages as possible.')
            break

        while ((i < len(response_json['items'])) and (len(urls) < num_webpages)):
            curr_url = response_json['items'][i]['link']
            curr_title = response_json['items'][i]['title']
            try:
                with urlopen(curr_url) as url_response:
                    html = url_response.read()
            except:
                i += 1
                continue
            if (curr_url in urls):
                i+=1
                continue
            text = html_to_text(html, curr_url)
            if (text):
                urls.append(curr_url)
                titles.append(curr_title)
                texts.append(text)
            i += 1
    return texts, urls, titles 

def clean_wikipedia_citations(text):
    number_ls = re.findall(r'[0-9]+', text)
    strings_to_remove = ['[edit]']
    for number in number_ls:
        strings_to_remove.append('['+str(number)+']')
    for x in strings_to_remove:
        text = text.replace(x, '')
    return text

def html_to_text_new(html, url):
    soup2 = BeautifulSoup(html, features="html.parser")
    for script2 in soup2(["script", "style"]):
        script2.extract()
    paragraphs = soup2.find_all('p')
    clean_paragraphs = []                
    for paragraph in paragraphs:
        if paragraph.find_parents(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']) == []:
            if not paragraph.find(['strong', 'b']):
                clean_paragraphs.append(paragraph.get_text().strip())
    text2 = ' \n'.join(clean_paragraphs)
    text2 = standardize_quotation_marks(text2)
    if ('https://en.wikipedia.org' in url):
        text2 = clean_wikipedia_citations(text2)
    return text2

def html_to_text(html, url):
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
    if ('https://en.wikipedia.org' in url):
        text = clean_wikipedia_citations(text)
    return text

def bing_retrieve(question, num_webpages):
    search_url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    
    texts = []
    urls = []
    titles = []
    num_searches = 0
    while (len(urls) < num_webpages):
        params = {"q": question, "textDecorations": True, "textFormat": "HTML", "count":23, "offset":23*num_searches}
        response = requests.get(search_url, headers=headers, params=params)
        response_json = response.json()
        num_searches += 1
        i = 0
        while ((not response_json) or ('webPages' not in response_json.keys())):
            print('Bing timeout. Trying again...')
            time.sleep(3)
            response = requests.get(search_url, headers=headers, params=params)
            response_json = response.json()
            
        while ((i < len(response_json['webPages']['value'])) and (len(urls) < num_webpages)):

            curr_url = response_json['webPages']['value'][i]['url']
            try:
                with urlopen(curr_url) as url_response:
                    html = url_response.read()
            except:
                i += 1
                continue
            urls.append(curr_url)
            titles.append(response_json['webPages']['value'][i]['name'])
            texts.append(html_to_text(html, curr_url))
            i += 1
        
    
    return texts, urls, titles 

def old_chunk_results(texts): # chunks into full sentences
    all_text_chunks = []
    chunk_url_idxs = []
    for k in range(len(texts)):
        text_chunks = []
        text = texts[k]
        sentences = text.split('.')

        # remove sentences with over 1000 characters
        for i in range(len(sentences)):
            if (len(sentences[i]) >= 1000):
                sentences[i] = ''

        i = 0
        while (i<len(sentences)):
            curr_chunk = ''
            j = 0
            while ((i<len(sentences)) and (j + len(sentences[i]) + 1 <= 1000)): # if adding next sentence fits under 1000 characters
                curr_chunk += sentences[i] + '.'
                j += len(sentences[i]) + 1
                assert j == len(curr_chunk)
                i += 1
            if (not discard_chunk(curr_chunk)):
                text_chunks.append(curr_chunk)

        all_text_chunks.extend(text_chunks)
        chunk_url_idxs.extend([k]*len(text_chunks))
    return all_text_chunks, chunk_url_idxs

def chunk_results(texts): # chunks into full sentences
    all_text_chunks = []
    chunk_url_idxs = []
    for k in range(len(texts)):
        text_chunks = []
        text = texts[k]
        paragraphs = text.split('\n')

        i = 0
        while (i<len(paragraphs)):
            curr_chunk = ''
            j = 0
            while ((i<len(paragraphs)) and ((j + len(paragraphs[i]) <= 1000) or (j==0))): # if adding next paragraph fits under 1000 characters or is just one paragraph
                curr_chunk += paragraphs[i] + '\n'
                j += len(paragraphs[i]) + 1
                assert j == len(curr_chunk)
                i += 1
            if (not discard_chunk(curr_chunk)):
                text_chunks.append(curr_chunk)

        all_text_chunks.extend(text_chunks)
        chunk_url_idxs.extend([k]*len(text_chunks))
    return all_text_chunks, chunk_url_idxs

def discard_chunk(chunk):
    discard_factor1 = (chunk.count('\n') >= 15)
    discard_factor2 = False 
    discard_factor3 = (chunk.count('.') <= 3)
    discard_factor4 = (len(re.findall(r"[^a-zA-Z0-9 \t\n\r\f]", chunk)) >= 70) # rules out wikipedia citations
    return discard_factor1 or discard_factor2 or discard_factor3 or discard_factor4

def consolidate_consecutive_chunks(top_idxs, top_chunks, top_urls, top_titles):
    tupled_info = list(zip(top_idxs, top_urls, top_titles, top_chunks))
    sorted_tupled_info = sorted(tupled_info, key=lambda item: (item[1], item[0]))
    top_idxs = [i for i,j,k,l in sorted_tupled_info]
    top_urls = [j for i,j,k,l in sorted_tupled_info]
    top_titles = [k for i,j,k,l in sorted_tupled_info]
    top_chunks = [l for i,j,k,l in sorted_tupled_info]
    i=len(top_idxs)-1
    while (i > 0):
        if ((top_idxs[i]-1 == top_idxs[i-1]) and (top_urls[i] == top_urls[i-1])):
            top_chunks[i-1] = top_chunks[i-1]+top_chunks[i]
            top_chunks.pop(i)
            top_urls.pop(i)
            top_titles.pop(i)
        i -= 1
    return top_chunks, top_urls, top_titles

class PostHocRetrieval:
    def __init__(self, num_webpages_to_retrieve, num_chunks_to_retrieve_per_sentence, data=None):
        self.search_url = "https://api.bing.microsoft.com/v7.0/search"
        self.num_chunks_per_sentence = num_chunks_to_retrieve_per_sentence
        self.num_webpages = num_webpages_to_retrieve
        self.semantic_similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.google_dpr_retriever = GoogleDPRRetrieval(50, 10)
        
        # Cache for the retrieved sources
        self.source_cache_fp = 'cache/post_hoc_cache_dict.pkl'
        if (not os.path.exists(self.source_cache_fp)):
            with open(self.source_cache_fp, 'wb') as f: 
                pickle.dump({}, f)
        with open(self.source_cache_fp, 'rb') as f:
            self.source_cache_dict = pickle.load(f)

    def get_from_post_hoc_cache(self, query):
        cached_entry = self.source_cache_dict[query]
        if (len(cached_entry['top_urls']) == 0):
            return None, None, None
        print('Using '+str(len(cached_entry['top_chunks']))+' chunks!')

        return cached_entry['response'], cached_entry['top_chunks'], cached_entry['top_urls']
    
    def is_cached(self, query):
        return query in self.source_cache_dict

    def retrieve_by_query(self, query, all_text_chunks, chunk_url_idxs, urls):
        # get the top ten chunks, as done for other OPs with GoogleDPRRetrieval 
        titles = ['placeholder']*len(urls)
        return self.google_dpr_retriever.dpr_retrieval(query, urls, titles, all_text_chunks, chunk_url_idxs)

    def retrieve(self, instance, response_sentences, response, use_gold):
        query = instance['question']
        if (self.is_cached(query)):
            print('Using cached entry!')
            return self.get_from_post_hoc_cache(query)

        print('Searching the internet...')
        texts, urls, titles = google_retrieve(query, self.num_webpages, False, False)
        
        print('Found '+str(len(urls))+' webpages!')
        all_text_chunks, chunk_url_idxs = chunk_results(texts)
        if (len(all_text_chunks)==0):
            print('NOTHING SCRAPED :(')
            return response, [], [], []

        all_idxs_by_response, all_chunks_by_response, all_urls_by_response = self.retrieve_by_response(all_text_chunks, chunk_url_idxs, urls, response_sentences)
        if (not use_gold):
            all_idxs_by_query, all_chunks_by_query, all_urls_by_query, _ = self.retrieve_by_query(query, all_text_chunks, chunk_url_idxs, urls)
        else:
            all_chunks_by_query = instance['gold_reference'] 
            all_idxs_by_query = [-1]*len(all_chunks_by_query) 
            all_urls_by_query = instance['urls']
        all_urls = []
        all_chunks = []
        all_idxs = []
        for j in range(len(all_chunks_by_response)):
            chunk = all_chunks_by_response[j]
            if (chunk not in all_chunks):
                    all_idxs.append(all_idxs_by_response[j])
                    all_chunks.append(chunk)
                    all_urls.append(all_urls_by_response[j])
        for j in range(len(all_chunks_by_query)):
            chunk = all_chunks_by_query[j]
            if (chunk not in all_chunks):
                    all_idxs.append(all_idxs_by_query[j])
                    all_chunks.append(chunk)
                    all_urls.append(all_urls_by_query[j])

        self.source_cache_dict[query] = {'top_urls':all_urls, 
                                    'top_chunks':all_chunks, 
                                    'top_idxs': all_idxs,
                                    'response': response}
        with open(self.source_cache_fp, 'wb') as f:
            pickle.dump(self.source_cache_dict, f)
                                    
        return self.get_from_post_hoc_cache(query)
    
    def retrieve_by_response(self, all_text_chunks, chunk_url_idxs, urls, response_sentences):

        # encode each chunk
        all_idxs, all_chunks, all_urls = [], [], []

        sentence_embeddings = self.semantic_similarity_model.encode(response_sentences)
        chunk_embeddings = self.semantic_similarity_model.encode(all_text_chunks)
        scores = sentence_embeddings @ chunk_embeddings.T
        for i in range(len(response_sentences)):
            top_idxs = (-scores[i,:]).argsort()[:self.num_chunks_per_sentence]
            top_chunks = np.array(all_text_chunks)[top_idxs]
            top_urls = np.array(urls)[np.array(chunk_url_idxs)[top_idxs]]
            for j in range(len(top_chunks)):
                chunk = top_chunks[j]
                if (chunk not in all_chunks):
                    all_idxs.append(top_idxs[j])
                    all_chunks.append(chunk)
                    all_urls.append(top_urls[j])

        return all_idxs, all_chunks, all_urls
        
class GoogleDPRRetrieval:
    def __init__(self, num_webpages_to_retrieve, num_chunks_to_retrieve, data=None):
        self.search_url = "https://api.bing.microsoft.com/v7.0/search"
        self.source_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", model_max_length=512)
        self.source_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(torch.device('cuda'))
        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base", model_max_length=512)
        self.query_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(torch.device('cuda'))
        self.num_chunks = num_chunks_to_retrieve
        self.num_webpages = num_webpages_to_retrieve
        self.tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
        self.avoid_webmd = (data=='mash')
        self.avoid_wikipedia = (data=='multihop')
        self.retrieve = self.retrieve_for_question
        if (data=='multihop'):
            self.backbone_model = OpenAI()
            self.retrieve = self.retrieve_for_super_question
        
        if (not os.path.exists('cache/google_cache_dict.pkl')):
            with open('cache/google_cache_dict.pkl', 'wb') as f:
                pickle.dump({}, f)
        with open('cache/google_cache_dict.pkl', 'rb') as f:
            self.cache_dict = pickle.load(f)

    def get_from_cache(self, question):
        cached_entry = self.cache_dict[question]
        if (len(cached_entry['top_idxs']) == 0):
            return None, None, None, None, None, None
        num_chunks = min(len(cached_entry['top_urls']), self.num_chunks)
        best_chunk = cached_entry['top_chunks'][0]
        best_url = cached_entry['top_urls'][0]
        best_title = cached_entry['top_titles'][0]
        top_chunks, top_urls, top_titles = consolidate_consecutive_chunks(cached_entry['top_idxs'][:num_chunks], 
                                                                            cached_entry['top_chunks'][:num_chunks], 
                                                                            cached_entry['top_urls'][:num_chunks], 
                                                                            cached_entry['top_titles'][:num_chunks])
        print('Using '+str(len(top_chunks))+' chunks...')
        return top_chunks, top_urls, top_titles, best_chunk, best_url, best_title

    def retrieve_for_super_question(self, question):
        if (question in self.cache_dict.keys()):
            print('Using cached search results...')
            return self.get_from_cache(question)
        subquestions = get_sub_questions(question, self.backbone_model)
        subquestions = subquestions.split('?')
        subquestions = [sq.strip()+'?' for sq in subquestions][:-1]
        top_idxs, top_chunks, top_urls, top_titles = [], [], [], []
        all_idxs, all_chunks, all_urls, all_titles = [], [], [], []
        self.num_chunks = 10
        self.num_webpages = 25
        subquestion_idxs_with_results = []
        for i in range(len(subquestions)):
            q = subquestions[i]
            curr_top_idxs, curr_top_chunks, curr_top_urls, curr_top_titles = self.retrieve_for_subquestion(q)
            all_chunks.append(curr_top_chunks)
            all_urls.append(curr_top_urls)
            all_titles.append(curr_top_titles)
            all_idxs.append(curr_top_idxs)
            if (len(curr_top_urls) > 0):
                subquestion_idxs_with_results.append(i)
        
        superquestion_num_chunks = int(10/len(subquestion_idxs_with_results))
        for i in subquestion_idxs_with_results:
            num_chunks = min(len(all_chunks[i]), superquestion_num_chunks)
            top_chunks.extend(all_chunks[i][:num_chunks])
            top_urls.extend(all_urls[i][:num_chunks])
            top_titles.extend(all_titles[i][:num_chunks])
            top_idxs.extend(all_idxs[i][:num_chunks])

        if (question not in self.cache_dict.keys()):
            self.cache_dict[question] = {'top_urls':top_urls, 
                                         'top_chunks':top_chunks, 
                                         'top_titles':top_titles,
                                         'top_idxs': top_idxs} 
        return self.get_from_cache(question)

    def dpr_retrieval(self, question, urls, titles, all_text_chunks, chunk_url_idxs):
        query_ids = self.query_tokenizer(question, return_tensors="pt", truncation=True)["input_ids"].to(torch.device('cuda'))
        query_embeddings = self.query_encoder(query_ids).pooler_output.detach().cpu()
        scores = []
        for i in range(len(all_text_chunks)):
            text_ids = self.source_tokenizer(all_text_chunks[i], return_tensors="pt", truncation=True)["input_ids"].to(torch.device('cuda'))
            text_embeddings = self.source_encoder(text_ids).pooler_output.detach().cpu()
            scores.append(np.dot(query_embeddings.detach().numpy(), text_embeddings.detach().numpy().T).item())
            if (i%100 == 0):
                print('Thinking... '+str(i)+'/'+str(len(all_text_chunks)))
        scores = np.array(scores)
        top_idxs = np.sort((-scores).argsort()[:min(20, len(scores))])
        top_chunks = np.array(all_text_chunks)[top_idxs].tolist()
        top_urls = np.array(urls)[np.array(chunk_url_idxs)[top_idxs]].tolist()
        top_titles = np.array(titles)[np.array(chunk_url_idxs)[top_idxs]].tolist()
        return top_idxs, top_chunks, top_urls, top_titles

    def retrieve_for_subquestion(self, question):
        print('Searching the internet...')
        texts, urls, titles = google_retrieve(question, self.num_webpages, self.avoid_webmd, self.avoid_wikipedia) # bing_retrieve
        
        print('Found '+str(len(urls))+' webpages!')
        all_text_chunks, chunk_url_idxs = chunk_results(texts)
        if (len(all_text_chunks)==0):
            print('Nothing scraped...')
            return [], [], [], []
        return self.dpr_retrieval(question, urls, titles, all_text_chunks, chunk_url_idxs)

    def retrieve_for_question(self, question):
        if (question in self.cache_dict.keys()):
            print('Using cached search results...')
            return self.get_from_cache(question)

        top_idxs, top_chunks, top_urls, top_titles = self.retrieve_for_subquestion(question)
        self.cache_dict[question] = {'top_urls':top_urls, 
                                        'top_chunks':top_chunks, 
                                        'top_titles':top_titles,
                                        'top_idxs': top_idxs} 
        
        with open('cache/google_cache_dict.pkl', 'wb') as f:
            pickle.dump(self.cache_dict, f)

        return self.get_from_cache(question)