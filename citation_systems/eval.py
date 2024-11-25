import os
from openai import OpenAI
import re 
from subprocess import check_output
import json 
import numpy as np

class AutoEvaluator:
    def __init__(self):
        self.model = OpenAI()
        self.temperature=0.2
        self.max_tokens=512
        self.frequency_penalty=0.0

    def evaluate_likert(self, query, response, answer_choices):
        response = "\nResponse: "+response+"\n"
        prompt = query+response+answer_choices+"\nAnswer: "
        message=[{"role": "assistant", "content": "a helpful expert"}, {"role": "user", "content": prompt}]
        rating_output_text = None
        num_attempts = 0
        while ((rating_output_text not in ['1', '2', '3']) and (num_attempts < 5)):
            rating_output = self.model.chat.completions.create(
                    model="gpt-4-0125-preview", 
                    messages = message,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    frequency_penalty=self.frequency_penalty
            )
            rating_output_text = rating_output.choices[0].message.content
            num_attempts += 1
        if (rating_output_text in ['1', '2', '3']):
            rating_output_text = int(rating_output_text)
        else: 
            rating_output_text = -1
        return rating_output_text
    
    def evaluate_fluency(self, query, response):
        query = "To what extent is the response fluent and coherent? Answer with a single number from the multiple choice options below."
        answer_choices = "Multiple choice options:\n1: The response has noticeable misprints or abrupt transitions between sentences\n2: The response has no misprints and mostly smooth transitions between sentences\n3: The response has no misprints and all of the sentences flow nicely together"
        return self.evaluate_likert(query, response, answer_choices)
    
    def evaluate_perceived_utility(self, query, response):
        query = "To what extent is the response a useful answer to the query? Answer with a single number from the multiple choice options below."
        answer_choices = "Multiple choice options:\n1: The response is a frustrating length or the query is not addressed\n2: The response is only a partially satisfying answer to the query\n3: The response is a reasonable length and is a satisfying answer to the query"
        return self.evaluate_likert(query, response, answer_choices)
    
def evaluate_quote_coverage(response, sources):
    "Returns both the number of quoted words, the number of total words, and the total number of quotes"
    fragments = response.split('\"')
    quotes = fragments[1:-1:2]
    num_quotes = len(quotes)
    num_quoted_words = 0
    num_words = len(response.split(' '))
    for quote in quotes:
        precise_quote = False
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
    tokens = text.lower().split(' ') 
    ngrams = []
    for i in range(0, len(tokens)-n+1):
        curr_ngram = ' '.join(tokens[i:i+n])
        if (len(curr_ngram) == 0):
            continue
        if (curr_ngram[-1] in [',', '.', '?', '!']):
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
    # w.r.t. multiple sources
    response_ngrams = get_n_grams(response, n)
    all_source_ngrams = []
    for source in sources_ls:
        if (source == ''):
            breakpoint()
        all_source_ngrams.extend(get_n_grams(source, n))
    count = num_common_ngrams(response_ngrams, all_source_ngrams)
    return (count, len(response_ngrams))