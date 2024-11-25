import difflib
import numpy as np
import re

def punctuation_exceptions(quote, source):
    if (quote in source):
        return True 
    elif ((quote[-1] == '.') and (quote[:-1] in source)):
        return True 
    elif ((quote[-1] == ',') and (quote[:-1] in source)):
        return True 
    return False

def cleaned(source):
    list_source = source.split(" ")
    i=0
    while i < len(list_source):
        if (list_source[i] == "``"):
            del list_source[i]
            i -= 1
            if (i+1 < len(list_source)):
                list_source[i+1] = "\'"+list_source[i+1]
        i += 1
    cleaned_source = " ".join(list_source)
    return cleaned_source

def eval_precision(output, sources):
    fragments = output.split('\"')
    quotes = fragments[1:-1:2]
    num_quotes = len(quotes)
    quote_precision = np.zeros(num_quotes)
    i=0
    for quote in quotes:
        found_quote = False
        for source in sources:
            source = cleaned(source)
            uppercase_quote = quote[0].upper() + quote[1:]
            found_quote = punctuation_exceptions(uppercase_quote, source)
            if (found_quote == True):
                break
            lowercase_quote = quote[0].lower() + quote[1:]
            found_quote = punctuation_exceptions(lowercase_quote, source)
            if (found_quote == True):
                break
        quote_precision[i] = found_quote
        i += 1
    return quote_precision

def eval_quote_coverage(output):
    fragments = output.split('\"')
    quotes = fragments[1:-1:2]
    number_quoted_words = 0
    for quote in quotes:
        number_quoted_words += len(quote.split(" "))
    number_output_words = len(output.split(" "))
    return number_quoted_words*1.0/number_output_words
        

    

