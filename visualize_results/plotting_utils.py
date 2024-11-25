import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###################################################################################################
# Extract data from database CSVs
###################################################################################################
def get_dict_for_op_ls(op_ls):
    d = {}
    for op in op_ls:
        d[op] = []
    return d

def get_precision_annotations_by_source_by_op(results, op_ls):
    # get list of binary citation precision annotations over all responses for each OP, deduplicated by source and aggregated with "or"
    # returns a dictionary of precision results by OP
    precision_annotations_by_source_by_op = get_dict_for_op_ls(op_ls) # {'Quoted':[], 'Paraphrased':[], 'Entailed':[], 'Abstractive':[], 'Post Hoc':[], 'Gemini':[]}
    for op in op_ls: # ['Quoted', 'Paraphrased', 'Entailed', 'Abstractive', 'Post Hoc', 'Gemini']:
        if (op == 'Snippet'):
            continue
        for dataset in np.unique(results['dataset']):
            op_results = results[(results['op']==op)&(results['dataset']==dataset)]
            for i in range(len(op_results)):
                curr_response = op_results.iloc[i]
                response_precision_annotations_by_source = get_precision_annotations_by_source_for_response(curr_response)
                if (response_precision_annotations_by_source == None): 
                    continue
                precision_annotations_by_source_by_op[op].extend(response_precision_annotations_by_source)
    return precision_annotations_by_source_by_op

def get_precision_annotations_by_source_for_response(response_df, suffix=''):
    # get list of binary citation precision annotations across all sentences for one response, deduplicated by source and aggregated with "or"
    # returns a list of the deduplicated precision annotations for one response
    # if there is no coverage annotation for a sentence, there are no precision annotations recorded for that sentence
    citations_dict = eval(response_df['Citation Dict'])
    precision_dict = eval(response_df['precise_citations'+suffix])
    coverage_dict = eval(response_df['is_covered'])
    used_sources = eval(response_df['Used Sources (cited)'])
    deduplicated_precision_annotations = []
    # assert len(citations_dict) == len(precision_dict) # consistent number of sentences
    for i in range(len(precision_dict)):
        # for each sentence:
        sentence_id = precision_dict[i]['sentence_id']
        coverage_id = 1000
        for j in range(len(coverage_dict)):
            if (coverage_dict[j]['sentence_id'] == sentence_id):
                coverage_id = j
                break
        coverage = coverage_dict[coverage_id]['coverage']
        if (coverage == -1):
            continue
        generated_citation_numbers = citations_dict[str(sentence_id)]['citation_numbers']
        precision_annotations = precision_dict[i]['annotations']
        assert len(generated_citation_numbers) == len(precision_annotations) # consistent number of citations for this sentence
        source_number_to_annotation_dict = {}
        for j in range(len(generated_citation_numbers)):
            # for each citation:
            citation_number = generated_citation_numbers[j]
            source_number = get_source_number_from_citation_number(citation_number, used_sources)
            annotation_value = precision_annotations[j]
            if source_number in source_number_to_annotation_dict.keys():
                # if the source has been seen before for this sentence, deduplicate with "or"
                source_number_to_annotation_dict[source_number] = int(annotation_value or source_number_to_annotation_dict[source_number]) 
            else:
                # if the source hasn't been seen before for this sentence, add it!
                source_number_to_annotation_dict[source_number] = int(annotation_value)
        sentence_source_annotations_ls = [source_number_to_annotation_dict[source_number] for source_number in source_number_to_annotation_dict.keys()]
        deduplicated_precision_annotations.extend(sentence_source_annotations_ls)
    return deduplicated_precision_annotations
    
def get_source_number_from_citation_number(citation_number, used_sources):
    # given a citation_number, identify the source to which it refers
    COLORS = {'\x1b[92m':':green[', '\x1b[96m':':orange[', '\x1b[95m':':red[', '\x1b[1;31;60m':':blue[', '\x1b[102m':':violet[', '\x1b[1;35;40m':':grey[', '\x1b[0;30;47m':':rainbow[', '\x1b[0;33;47m':':orange[', '\x1b[0;34;47m':':blue[', '\x1b[0;31;47m':':red[', '\x1b[0m':']'}
    source_num = None
    for k in range(len(used_sources)):
        for ansi_color in COLORS.keys():
            if (ansi_color+'['+str(citation_number)+']' in used_sources[k]):
                source_num = k
                break
        if (source_num):
            break
    return source_num

def get_cov_results_for_response(results, op, query_id, suffix=''):
    # if there is no coverage annotation for a sentence, that sentence is counted as NOT covered
    # Obtain the coverage annotations {0,1} for all citations the specified response
    citation_cov_annotations = results[(results['query_id']==query_id)&(results['op']==op)]['is_covered'+suffix].iloc[0]
    citation_cov_annotations = citation_cov_annotations.replace('true', 'True')
    citation_cov_annotations = citation_cov_annotations.replace('false', 'False')
    citation_cov_annotations = eval(citation_cov_annotations)
    
    curr_citation_cov_df = pd.DataFrame(citation_cov_annotations)
    # do not need citations in order by sentence, eventually for inter-annotator agreement
    if (len(curr_citation_cov_df) > 0):
        curr_citation_cov_df.sort_values(by='sentence_id', ascending=True)
    response_cov_ls = []
    for i in range(len(curr_citation_cov_df)):
        # Obtain the precision annotations {0,1} for all citations the sentence
        sentence_cov_ls = int(curr_citation_cov_df.iloc[i]['coverage'])
        if (sentence_cov_ls == -1):
            sentence_cov_ls = 0
        response_cov_ls.append(sentence_cov_ls)
    return response_cov_ls

def get_coverage_annotations_by_op(results, op_ls):
     # get list of binary citation coverage annotations over all responses for each OP
    # returns a dictionary of coverage results by OP
    coverage_annotations_by_op = get_dict_for_op_ls(op_ls) # {'Quoted':[], 'Paraphrased':[], 'Entailed':[], 'Abstractive':[], 'Post Hoc':[], 'Gemini':[]}
    for op in op_ls: # ['Quoted', 'Paraphrased', 'Entailed', 'Abstractive', 'Post Hoc', 'Gemini']:
        if (op == 'Snippet'):
                continue
        for dataset in np.unique(results['dataset']):
            # Obtain the coverage annotations {0,1} for all citations for all responses for the specified OP
            op_results = results[(results['op']==op)&(results['dataset']==dataset)]
            # return the coverage annotations {0,1} for all citations of that operating point
            all_cov_ls = []
            for query_id in op_results['query_id']:
                all_cov_ls.extend(get_cov_results_for_response(op_results, op, query_id))
            coverage_annotations_by_op[op].extend(all_cov_ls)
    return coverage_annotations_by_op

def get_t2v_for_op(results, op, over_response=False, num_sentences=0):
    # if there is no coverage annotation for a sentence, there is no recorded T2V
    # return the T2V coverage for all sentences of that operating point
    all_ls = []
    for dataset in np.unique(results['dataset']):
        op_df = results[(results['op']==op)&(results['dataset']==dataset)]
        for query_id in op_df['query_id']:
            # Obtain the T2V coverage for all citations the specified response
            cov_t2v_annotations = eval(op_df[op_df['query_id']==query_id]['t2v_coverage'].iloc[0])
            if (over_response):
                if (num_sentences==0):
                    all_ls.append(np.sum(cov_t2v_annotations))
                else:
                    if (len(cov_t2v_annotations) == num_sentences):
                        all_ls.append(np.sum(cov_t2v_annotations))
            else:
                all_ls.extend(cov_t2v_annotations)
    return all_ls

def get_normalized_t2v_for_op(results, op):
    # if there is no coverage annotation for a sentence, there is no recorded T2V
    # return the T2V coverage for all sentences of that operating point
    op_results = results[results['op']==op]
    all_ls = []
    for query_id in op_results['query_id']:
        # Obtain the T2V coverage for all citations the specified response
        curr_results = results[(results['query_id']==query_id)&(results['op']==op)]
        cov_t2v_annotations = eval(curr_results['t2v_coverage'].iloc[0])
        # get word counts for each of these sentences
        sentences_ls = eval(curr_results['Sent'].iloc[0])
        citation_cov_annotations = eval(curr_results['is_covered'].iloc[0])
        sentence_num_words_for_response_ls = []
        for i in range(len(citation_cov_annotations)):
            sentence_id = citation_cov_annotations[i]['sentence_id']
            if (citation_cov_annotations[i]['coverage'] == -1):
                # there is no T2V saved for this sentence if coverage is -1; no coverage evaluation occurred
                continue
            sentence = sentences_ls[sentence_id]
            sentence_num_words = len(sentence.split(' '))
            sentence_num_words_for_response_ls.append(sentence_num_words)    
        # divide
        normalized_cov_t2v_annotations = np.divide(cov_t2v_annotations, sentence_num_words_for_response_ls)
        all_ls.extend(normalized_cov_t2v_annotations)
    return all_ls

def get_t2v_by_op(results, op_ls, normalized=False, remove_outliers=False, coverage_value=-1, over_response=False, num_sentences=0):
    # If the sentence doesn't have any accompanying citations, the system currently does not show it to the user for annotation. No T2V is logged. 
    # And, an entry of -1 is logged in the coverage list (which is by sentence). 
    # And, an entry of [] or [0]*(# of citations) is logged in the precision list (which is by sentence).
    t2v_annotations_by_op = get_dict_for_op_ls(op_ls) 
    for op in op_ls: 
        if (op == 'Snippet'):
            continue
        # Obtain the coverage annotations {0,1} for all citations for all responses for the specified OP
        all_t2v_ls = []
        if (normalized):
            all_t2v_ls.extend(get_normalized_t2v_for_op(results, op))
        elif ((coverage_value != -1) and (over_response==False)):
            # If getting average T2V over sentences for a specific coverage value
            all_t2v_ls.extend(get_t2v_for_op_given_coverage_value(results, op, coverage_value))
        else:
            # If getting average T2V over responses with perfect coverage, results should already by filtered down to just those rows
            all_t2v_ls.extend(get_t2v_for_op(results, op, over_response=over_response, num_sentences=num_sentences))
        if (remove_outliers):
            t2v_annotations_by_op[op] = remove_outliers_by_1_5_IQR(all_t2v_ls)
        else:
            t2v_annotations_by_op[op] = all_t2v_ls

        # remove any T2Vs greater than 5 minutes
        t2v_annotations_by_op[op] = [x for x in t2v_annotations_by_op[op] if x<300]
    return t2v_annotations_by_op
    
def get_t2v_by_op_across_datasets(all_results, op_ls, remove_outliers=False):
    all_t2v_cov0 = get_dict_for_op_ls(op_ls) 
    all_t2v_cov1 = get_dict_for_op_ls(op_ls) 
    all_t2v = get_dict_for_op_ls(op_ls) 
    for k in all_results.keys():
        results = all_results[k]
        t2v_cov0 = get_t2v_by_op(results, op_ls, normalized=False, remove_outliers=remove_outliers, coverage_value=0)
        t2v_cov1 = get_t2v_by_op(results, op_ls, normalized=False, remove_outliers=remove_outliers, coverage_value=1)
        t2v = get_t2v_by_op(results, op_ls, normalized=False, remove_outliers=remove_outliers, coverage_value=-1)
        for op in all_t2v_cov0.keys():
            all_t2v_cov0[op].extend(t2v_cov0[op])
            all_t2v_cov1[op].extend(t2v_cov1[op])
            all_t2v[op].extend(t2v[op])
    return all_t2v_cov0, all_t2v_cov1, all_t2v

def get_median_ci_1(op_t2v):
    op_t2v.sort()
    median = np.median(op_t2v)
    median_idx = np.where(op_t2v==median)[0][0]
    n = len(op_t2v)
    prev_rhs = 0
    rhs = -1
    ci_threshold = .95
    prev_lower_idx = median_idx
    prev_upper_idx = median_idx
    lower_idx = -1
    upper_idx = -1
    while (prev_rhs < .95):
        rhs = prev_rhs
        lower_idx = prev_lower_idx
        upper_idx = prev_upper_idx
        
        prev_lower_idx -= 1
        prev_upper_idx += 1
        prev_rhs = 0
        idx = prev_lower_idx
        while (idx <= prev_upper_idx):
            prev_rhs += math.comb(n, idx)
            idx += 1
        prev_rhs = prev_rhs * 2**(-n)
    
    rhs = prev_rhs
    lower_idx = prev_lower_idx
    upper_idx = prev_upper_idx
    return median, lower_idx, op_t2v[lower_idx], upper_idx, op_t2v[upper_idx], rhs

def get_median_ci_2(op_t2v):
    # http://mchp-appserv.cpe.umanitoba.ca/viewConcept.php?printer=Y&conceptID=1092
    op_t2v.sort()
    median_idx = int(len(op_t2v)//2)
    median = op_t2v[median_idx]
    n = len(op_t2v)
    upper_idx = int(np.floor(median_idx + 1.96*np.sqrt(n)/2))
    lower_idx = int(np.ceil(median_idx - 1.96*np.sqrt(n)/2))
    return median, lower_idx, op_t2v[lower_idx], upper_idx, op_t2v[upper_idx], 0.95

def get_fluency_or_utility_by_op(results, data_key, op_ls):
    annotations_by_op = get_dict_for_op_ls(op_ls) 
    for op in op_ls: 
        annotations_by_op[op] = results[results['op']==op][data_key]
        annotations_by_op[op] = [x for x in annotations_by_op[op] if x != -1]
    return annotations_by_op

def get_t2v_for_op_given_coverage_value(results, op, coverage_value):
    # a function to get the T2V when coverage is 1 and when coverage is 0
    # if there is no coverage annotation for a sentence, there is no recorded T2V
    # return the T2V coverage for all sentences of that operating point with the coverage_value
    op_results = results[results['op']==op]
    all_ls = []
    for query_id in op_results['query_id']:
        # Obtain the T2V coverage for all citations the specified response
        curr_results = results[(results['query_id']==query_id)&(results['op']==op)]
        cov_t2v_annotations = eval(curr_results['t2v_coverage'].iloc[0])
        # get coverage annotations for each of these sentences
        sentences_ls = eval(curr_results['Sent'].iloc[0])
        citation_cov_annotations = eval(curr_results['is_covered'].iloc[0])
        j = 0 
        for i in range(len(citation_cov_annotations)):
            sentence_id = citation_cov_annotations[i]['sentence_id']
            if (citation_cov_annotations[i]['coverage'] == -1):
                # there is no T2V saved for this sentence if coverage is -1; no coverage evaluation occurred
                continue
            elif (citation_cov_annotations[i]['coverage'] == coverage_value):
                all_ls.append(cov_t2v_annotations[j])
                j+=1
            else:
                j+=1
    return all_ls

def get_t2v_df_given_coverage_value(results, coverage_value):
    # a function to get the T2V when coverage is 1 and when coverage is 0
    # if there is no coverage annotation for a sentence, there is no recorded T2V
    # return the df with 'annotator_id', 'op', 't2v_coverage' where 't2v_coverage' is over all sentences with the coverage_value
    results = results.copy(deep=True).reset_index(drop=True) 
    new_t2vs = []
    for k in range(len(results)):
        if (k%100 == 0):
            t2v_and_coverage_consistency(results, tag=str(k)+'!')
        all_ls = []
        op = results['op'].iloc[k]
        if (op == 'Snippet'):
            new_t2vs.append('[]')
            continue
        # Obtain the T2V coverage for all citations the specified response
        cov_t2v_annotations = eval(results['t2v_coverage'].iloc[k])
        # get coverage annotations for each of these sentences
        sentences_ls = eval(results['Sent'].iloc[k])
        citation_cov_annotations = eval(results['is_covered'].iloc[k])

        j = 0 
        for i in range(len(citation_cov_annotations)):
            if (citation_cov_annotations[i]['coverage'] == -1):
                # there is no T2V saved for this sentence if coverage is -1; no coverage evaluation occurred
                continue
            elif (citation_cov_annotations[i]['coverage'] == coverage_value):
                if (j >= len(cov_t2v_annotations)):
                    print('!!!')
                    print(op)
                    print('num T2V annotations', len(cov_t2v_annotations))
                    print('num coverage annotations', len(citation_cov_annotations))
                    print('T2V annotations', cov_t2v_annotations)
                    print('Coverage annotations', citation_cov_annotations)
                    print()
                    break
                all_ls.append(cov_t2v_annotations[j])
                j+=1
            else:
                j+=1
        # results.loc[k, 't2v_coverage'] = str(all_ls)
        new_t2vs.append(str(all_ls))
    results['t2v_coverage'] = new_t2vs
    return results[['annotator_id', 'op', 't2v_coverage']]

def get_response_df_of_fully_cited_outputs(results):
    # a  function to get the responses where each sentence has at least one T2V value
    results = results.reset_index(drop=True) 
    results_copy = results.copy(deep=True).reset_index(drop=True) 
    for k in range(len(results)):
        op = results['op'].iloc[k]
        if (op == 'Snippet'):
            continue
        # Obtain the T2V coverage for all citations the specified response
        cov_t2v_annotations = eval(results['t2v_coverage'].iloc[k])
        # get coverage annotations for each of these sentences
        sentences_ls = eval(results['Sent'].iloc[k])
        if (len(cov_t2v_annotations) != len(sentences_ls)):
            results_copy = results_copy.drop([k])

    return results_copy

def get_response_df_given_coverage_value(results, coverage_value):
    # a  function to get the responses where each sentence has at least one T2V value
    results = results.reset_index(drop=True) 
    results_copy = results.copy(deep=True).reset_index(drop=True) 
    for k in range(len(results)):
        op = results['op'].iloc[k]
        if (op == 'Snippet'):
            continue
        # Obtain the T2V coverage for all citations the specified response
        cov_t2v_annotations = eval(results['t2v_coverage'].iloc[k])
        # get coverage annotations for each of these sentences
        sentences_ls = eval(results['Sent'].iloc[k])
        citation_cov_annotations = eval(results['is_covered'].iloc[k])
        all_sentences_properly_covered = True
        for i in range(len(citation_cov_annotations)):
            if (citation_cov_annotations[i]['coverage'] != coverage_value):
                all_sentences_properly_covered = False
                break
        if (all_sentences_properly_covered == False):
            results_copy = results_copy.drop([k])
    return results_copy

####################################################################################
# Process extracted data
####################################################################################

def remove_outliers_by_1_5_IQR(t2vs):
    sorted_t2vs = np.sort(t2vs)
    median_idx = int(len(sorted_t2vs)//2)
    q1_idx = int(median_idx//2)
    q3_idx = median_idx + int(median_idx//2)
    iqr = sorted_t2vs[q3_idx] - sorted_t2vs[q1_idx]
    upper_outlier_limit = sorted_t2vs[q3_idx] + 1.5*iqr
    lower_outlier_limit = sorted_t2vs[q1_idx] - 1.5*iqr
    t2vs_to_keep = []
    for t2v in t2vs:
        if ((t2v < upper_outlier_limit) and (t2v > lower_outlier_limit)):
            t2vs_to_keep.append(t2v)
    return t2vs_to_keep
    
####################################################################################
# Get average values
####################################################################################

def get_avg_and_ci_by_op(values_by_op_dict, op_ls):
    values = []
    ci_ls = []
    for op in op_ls: 
        if op not in values_by_op_dict.keys():
            continue
        values.append(np.mean(values_by_op_dict[op]))
        ci_ls.append(1.96*np.std(values_by_op_dict[op], ddof=1) / np.sqrt(len(values_by_op_dict[op])))
    return values, ci_ls 

def get_median_and_ci_by_op(values_by_op_dict, op_ls):
    values = []
    ci_ls = [[0,0]]
    for op in op_ls: 
        if op not in values_by_op_dict.keys():
            continue
        median, _, lower_val, _, upper_val, rhs = get_median_ci_2(values_by_op_dict[op])
        values.append(median)
        ci_ls.append([median - lower_val, upper_val - median])
    ci_ls = np.array(ci_ls).T
    return values, ci_ls

def get_avg_fluency_or_utility_by_op(results, data_key, op_ls):
    all_values = get_fluency_or_utility_by_op(results, data_key, op_ls)
    values, ci_ls = get_avg_and_ci_by_op(all_values, op_ls)
    return values, ci_ls

def get_avg_t2v_by_op(results, op_ls, normalized=False, remove_outliers=False, coverage_value=-1, over_response=False, num_sentences=0):
    if (over_response and (coverage_value!=-1)):
        results = get_response_df_given_coverage_value(results, coverage_value)
    all_values = get_t2v_by_op(results, op_ls, normalized=normalized, remove_outliers=remove_outliers, coverage_value=coverage_value, over_response=over_response, num_sentences=num_sentences)
    values, ci_ls = get_avg_and_ci_by_op(all_values, op_ls)
    return [0]+values, [0]+ci_ls # snippet should have T2V of 0 with variance of 0

def get_median_t2v_by_op(results, op_ls, normalized=False, remove_outliers=False, coverage_value=-1):
    all_values = get_t2v_by_op(results, op_ls, normalized=normalized, remove_outliers=remove_outliers, coverage_value=coverage_value)
    values, ci_ls = get_median_and_ci_by_op(all_values, op_ls)
    return [0]+values, ci_ls # snippet should have T2V of 0 with variance of 0

def get_avg_precision_by_op(results, op_ls):
    all_values = get_precision_annotations_by_source_by_op(results, op_ls)
    values, ci_ls = get_avg_and_ci_by_op(all_values, op_ls)
    return [1]+values, [0]+ci_ls # snippet has precision of 1 with variance of 0

def get_avg_coverage_by_op(results, op_ls):
    all_values = get_coverage_annotations_by_op(results, op_ls)
    values, ci_ls = get_avg_and_ci_by_op(all_values, op_ls)
    return [1]+values, [0]+ci_ls # snippet has coverage of 1 with variance of 0

def get_relative_t2v_by_op(results, op_ls, normalizing_op, coverage_value=-1, over_response=False, num_sentences=0):
    # compute each t2v divided by the annotator's average t2v, then average over all sentences for each OP (has CI)
    if ((coverage_value != -1) and (over_response==False)):
        df = get_t2v_df_given_coverage_value(results, coverage_value)
    elif ((coverage_value != -1) and over_response):
        # df = get_response_df_given_coverage_value(results, coverage_value)
        df = get_response_df_of_fully_cited_outputs(results)
    else:
        df = results[['annotator_id', 'op', 't2v_coverage']]
    df = df[df['op']!='Snippet'] 
        
    if (over_response):
        df = df.copy(deep=True) 
        if (num_sentences != 0):
            # only keep rows pertaining to responses with the specified number of sentences
            df = df[df['t2v_coverage'].apply(lambda x: len(eval(x))) == num_sentences]
        df['t2v_coverage'] = df['t2v_coverage'].apply(lambda x: str([np.sum(eval(x))]))
        
    df['t2v_coverage'] = df['t2v_coverage'].apply(ast.literal_eval)
    df = df.groupby(by=['annotator_id', 'op']).agg({'t2v_coverage': 'sum'})
    def remove_over_500s(ls):
        return [x for x in ls if x<300]
    df['t2v_coverage'] = df['t2v_coverage'].apply(remove_over_500s)
    df['avg_t2v_coverage'] = df['t2v_coverage'].apply(lambda x: np.mean(x))
    df = df.reset_index()

    # compute each annotator's t2v relative to their average quoted t2v 
    relative_t2v_coverage = []
    relative_t2v_coverage_by_sent = []
    for i in range(len(df)):
        annotator_id = df['annotator_id'].iloc[i]
        if (len(df[(df['annotator_id']==annotator_id)&(df['op']==normalizing_op)]['t2v_coverage']) == 0): 
            # impute missing quotation t2v as avg quotation t2v over the other users 
            # this is the case where a baseline annotator did not participate in the original OP study
            annotator_avg_quoted_t2v = np.mean(df[df['op']==normalizing_op]['avg_t2v_coverage'])
        else:
            annotator_avg_quoted_t2v = df[(df['annotator_id']==annotator_id)&(df['op']==normalizing_op)]['avg_t2v_coverage'].iloc[0]
        if (np.isnan(annotator_avg_quoted_t2v)): # impute missing quotation t2v as avg quotation t2v over the other users
            annotator_avg_quoted_t2v = np.mean(df[df['op']==normalizing_op]['avg_t2v_coverage'])
        value_by_sent = [x/annotator_avg_quoted_t2v for x in df['t2v_coverage'].iloc[i]] 
        relative_t2v_coverage_by_sent.append(value_by_sent)
    df['relative_t2v_coverage_by_sent'] = relative_t2v_coverage_by_sent
    
    # compute the confidence intervals and average values
    ci_df = df[['op', 'relative_t2v_coverage_by_sent']].groupby(by=['op']).agg({'relative_t2v_coverage_by_sent': 'sum'}).reset_index()
    
    def get_ci(x):
        return 1.96*np.std(x, ddof=1)/np.sqrt(len(x))
    
    ci_df['ci'] = ci_df['relative_t2v_coverage_by_sent'].apply(get_ci)
    ci_df['mean_relative_t2v_coverage_by_sent'] = ci_df['relative_t2v_coverage_by_sent'].apply(np.mean)
    
    # compute the average of these relative t2vs
    values = [1]
    ci_values = [0]
    for op in op_ls:
        if (op == 'Snippet'):
            continue
        if (len(ci_df[ci_df['op']==op]['mean_relative_t2v_coverage_by_sent'])!=0):
            values.append(ci_df[ci_df['op']==op]['mean_relative_t2v_coverage_by_sent'].iloc[0])
            ci_values.append(ci_df[ci_df['op']==op]['ci'].iloc[0])
        else:
            values.append(1)
            ci_values.append(1)

    return values, ci_values

####################################################################################
# Plotting fns
####################################################################################

def plot_all(results, title_tag, t2v_style, op_ls, normalize_t2v=False, remove_outliers=False, coverage_value=-1):
    fluency = get_avg_fluency_or_utility_by_op(results, 'human_fluency_rating', op_ls)
    utility = get_avg_fluency_or_utility_by_op(results, 'human_utility_rating', op_ls)
    precision = get_avg_precision_by_op(results, op_ls[1:])
    coverage = get_avg_coverage_by_op(results, op_ls[1:])
    t2v_axis_label = 'T2V: Time (s)'
    if (normalize_t2v):
        t2v_upper = 2
    else:
        t2v_upper = 36

    if (t2v_style=='relative'):
        t2v_upper = 4.5
        t2v_axis_label = 'Relative T2V w.r.t. Quoted Output'
        
    if (t2v_style=='average'):
        t2v = get_avg_t2v_by_op(results, normalized=normalize_t2v, remove_outliers=remove_outliers, coverage_value=coverage_value)
    elif (t2v_style=='median'):
        t2v = get_median_t2v_by_op(results, op_ls, normalized=normalize_t2v, remove_outliers=remove_outliers, coverage_value=coverage_value)
    elif (t2v_style=='relative'):
        op_t2vs = get_relative_t2v_by_op(results, op_ls[1:], 'Quoted')
        new_avgs = op_t2vs[0][1:5]
        new_errors = op_t2vs[1][1:5]
        baseline_t2vs = get_relative_t2v_by_op(results, op_ls[1:], 'Quoted Reeval')
        new_avgs.extend(baseline_t2vs[0][-2:])
        new_errors.extend(baseline_t2vs[1][-2:])
        t2v = (new_avgs, new_errors)
    else:
        print('T2V visualization not yet implemented')
    
    results_list = [fluency, utility, precision, coverage, t2v]
    results_labels = ['Fluency', 'Utility', 'Precision', 'Coverage', 'T2V']
    ops = op_ls 
    
    # Initialize the plot
    fig, ax1 = plt.subplots()
    
    # Colors and markers
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    # Plot the first set of results
    lines, labels = ax1.get_legend_handles_labels()
    for idx, nested_tuple in enumerate(zip(results_list, results_labels)):
        result_label = nested_tuple[1]
        averages = nested_tuple[0][0]
        conf_intervals = nested_tuple[0][1]
        color = colors[idx % len(colors)]
        marker = 'o'
        # if (max(averages) > 1) & (max(averages) <= 3):  # Assuming the scale > 1
        if ((result_label == 'Fluency') or (result_label == 'Utility')):
            ax1.errorbar(ops, averages, yerr=conf_intervals, fmt=marker + '-', color=color, label=results_labels[idx], capsize=3, ms=4)
            ax1.set_ylim([1,3])
            lines, labels = ax1.get_legend_handles_labels()
        # elif (max(averages) <= 1):  
        elif ((result_label == 'Precision') or (result_label == 'Coverage')):
            
            ax2 = ax1.twinx()
            ax2.errorbar(ops, averages, yerr=conf_intervals, fmt=marker + '-', color=color, label=results_labels[idx], capsize=3, ms=4)
            ax2.set_ylim([0,1])
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        else: # Assuming T2V
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("outward", 60))
            ax3.errorbar(ops[1:], averages, yerr=conf_intervals, fmt=marker + '-', color=color, label=results_labels[idx], capsize=3, ms=4)
            ax3.set_ylim([0,t2v_upper])
            lines3, labels3 = ax3.get_legend_handles_labels()
            lines += lines3
            labels += labels3
    
    # Set labels and title
    ax1.set_xlabel('Operating Points')
    ax1.set_ylabel('Fluency and Utility: 3 point Likert Scale')
    
    ax2.set_ylabel('Coverage and Precision: Proportion')
    
    ax3.set_ylabel(t2v_axis_label)
    
    plt.title(title_tag+': All Results Over OPs')
    
    plt.legend(lines, labels, loc='lower left', bbox_to_anchor=(1.2, 1.2))
    
    # Show the plot
    plt.show()

def plot_t2v_by_coverage(all_results, t2v_style, op_ls, remove_outliers=False, tag='', t2v_upper=60):
    all_t2v_cov0, all_t2v_cov1, all_t2v = get_t2v_by_op_across_datasets(all_results, op_ls, remove_outliers=remove_outliers)
    if (t2v_style == 'average'):
        t2v_cov0_values, t2v_cov0_cis = get_avg_and_ci_by_op(all_t2v_cov0, op_ls)
        t2v_cov1_values, t2v_cov1_cis = get_avg_and_ci_by_op(all_t2v_cov1, op_ls)
        t2v_values, t2v_cis = get_avg_and_ci_by_op(all_t2v, op_ls)
        t2v_cov0_results = ([0]+t2v_cov0_values, [0]+t2v_cov0_cis)
        t2v_cov1_results = ([0]+t2v_cov1_values, [0]+t2v_cov1_cis)
        t2v_results = ([0]+t2v_values, [0]+t2v_cis)
    elif (t2v_style == 'median'):
        t2v_cov0_values, t2v_cov0_cis  = get_median_and_ci_by_op(all_t2v_cov0, op_ls)
        t2v_cov1_values, t2v_cov1_cis = get_median_and_ci_by_op(all_t2v_cov1, op_ls)
        t2v_values, t2v_cis = get_median_and_ci_by_op(all_t2v, op_ls)
        
        t2v_cov0_results = ([0]+t2v_cov0_values, t2v_cov0_cis) # np.vstack([np.array([[0,0]]).T, t2v_cov0_cis])
        t2v_cov1_results = ([0]+t2v_cov1_values, t2v_cov1_cis)
        t2v_results = ([0]+t2v_values, t2v_cis)
    elif (t2v_style == 'not implemented yet'):
        return
    
    results_list = [t2v_cov0_results, t2v_cov1_results, t2v_results]
    results_labels = ['T2V over non-covered sentences', 'T2V over covered sentences', 'T2V over all sentences']
    ops = op_ls 
    
    # Initialize the plot
    fig, ax1 = plt.subplots()
    
    # Colors and markers
    colors = ['b', 'r', 'g']
    
    # Plot the first set of results
    lines, labels = ax1.get_legend_handles_labels()
    for idx, nested_tuple in enumerate(zip(results_list, results_labels)):
        result_label = nested_tuple[1]
        averages = nested_tuple[0][0]
        conf_intervals = nested_tuple[0][1]
        color = colors[idx % len(colors)]
        marker = 'o'
        ax1.errorbar(ops, averages, yerr=conf_intervals, fmt=marker + '-', color=color, label=results_labels[idx], capsize=3, ms=4)
        lines1, labels1 = ax1.get_legend_handles_labels()
        # lines += lines1
        # labels += labels1
    
    # Set labels and title
    ax1.set_xlabel('Operating Points')
    ax1.set_ylabel('T2V: Time (s)')
    ax1.set_ylim([0,t2v_upper])
    
    plt.title('All datasets: T2V by coverage over OPs '+tag)
    plt.legend()
    plt.show()

def plot_t2vs_all_ways(results, op_ls, title_tag):
    average_t2v = get_avg_t2v_by_op(results, normalized=False, remove_outliers=False, coverage_value=-1)
    median_t2v = get_median_t2v_by_op(results, normalized=False, remove_outliers=False, coverage_value=-1)
    normalized_average_t2v = get_avg_t2v_by_op(results, normalized=True, remove_outliers=False, coverage_value=-1)
    normalized_median_t2v = get_median_t2v_by_op(results, normalized=True, remove_outliers=False, coverage_value=-1)
    relative_t2v = get_relative_t2v_by_op(results, op_ls)
    
    results_list = [average_t2v, median_t2v, normalized_average_t2v, normalized_median_t2v, relative_t2v]
    results_labels = ['Average T2V Sentence', 'Median T2V Sentence', 'Average T2V Word', 'Median T2V Word', 'Relative T2V Sentence']
    ops = op_ls 
    
    # Initialize the plot
    fig, ax1 = plt.subplots()
    
    # Colors and markers
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    # Plot the first set of results
    lines, labels = ax1.get_legend_handles_labels()
    for idx, nested_tuple in enumerate(zip(results_list, results_labels)):
        result_label = nested_tuple[1]
        averages = nested_tuple[0][0]
        conf_intervals = nested_tuple[0][1]
        color = colors[idx % len(colors)]
        marker = 'o'
        if ((result_label == 'Average T2V Sentence') or (result_label == 'Median T2V Sentence')):
            ax1.errorbar(ops, averages, yerr=conf_intervals, fmt=marker + '-', color=color, label=results_labels[idx], capsize=3, ms=4)
            ax1.set_ylim([0,36])
            lines, labels = ax1.get_legend_handles_labels()
        elif ((result_label == 'Average T2V Word') or (result_label == 'Median T2V Word')):
            ax2 = ax1.twinx()
            ax2.errorbar(ops, averages, yerr=conf_intervals, fmt=marker + '-', color=color, label=results_labels[idx], capsize=3, ms=4)
            ax2.set_ylim([0,2])
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        else: # Assuming T2V
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("outward", 60))
            ax3.errorbar(ops, averages, yerr=conf_intervals, fmt=marker + '-', color=color, label=results_labels[idx], capsize=3, ms=4)
            ax3.set_ylim([1,3])
            lines3, labels3 = ax3.get_legend_handles_labels()
            lines += lines3
            labels += labels3
    
    ax1.set_xlabel('Operating Points')
    ax1.set_ylabel('T2V Sentence (s)')
    ax2.set_ylabel('T2V Word (s)')
    ax3.set_ylabel('Fraction')
    plt.title(title_tag+': T2V Visualizations')
    plt.legend(lines, labels, loc='lower left', bbox_to_anchor=(1.2, 1.2))
    plt.show()

def t2v_and_coverage_consistency(all_results_df, tag=''):
    for i in range(len(all_results_df)):
        if (all_results_df['op'].iloc[i] == 'Snippet'):
            continue
        coverage = eval(all_results_df['is_covered'].iloc[i])
        t2v = eval(all_results_df['t2v_coverage'].iloc[i])
        num_nonneg_cov = len([x for x in coverage if x['coverage']!=-1])
        if (len(t2v) != num_nonneg_cov):
            print(tag)
            print('num T2V annotations', len(t2v))
            print('num coverage annotations', len(coverage))
            print('T2V annotations', t2v)
            print('Coverage annotations', coverage)
            print()