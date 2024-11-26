import pandas as pd
import os
import numpy as np

# This script does several things, including:
    # 1. Removes precision and T2V annotations for instances where coverage was set to -1 due to display issues
    # 2. Joins in the utility and fluency results from the re-done human evaluation
    # 2. Ensures each dataset is represented by 120 query-generation pairs per OP instantiation
    # 3. Joins in Vertex API results about which sentences require citation (resolves discrepancies in sentence parsing)
    # 4. Creates a copy of results with data only for sentences requiring citation

# The script creates the same files as clean_results.ipynb creates, but in script form.
# There are some data cleaning tasks done by hand in clean_results.ipynb (recorded and automated in this script)

def remove_irrelevant_t2v_and_precision_annotations(df):
    # Occasionally, the annotation interface fails to display a cited sentence (coverage = -1). 
    # In these cases, precision and T2V were still collected. This function identifies and removes these measurements.
    idxs_ops_of_interest = []
    for i in range(len(df)):
        if (df['op'].iloc[i] == 'Snippet'):
            continue
        t2vs = eval(df['t2v_coverage'].iloc[i])
        is_covered = eval(df['is_covered'].iloc[i])
        is_precise = eval(df['precise_citations'].iloc[i])
        actual_is_covered = []
        actual_is_precise = []
        for j in range(len(is_covered)):
            cov_item = is_covered[j]
            if (cov_item['coverage'] != -1):
                actual_is_covered.append(cov_item)
            prec_item = is_precise[j]
            if (len(prec_item['annotations'])!=0):
                actual_is_precise.append(prec_item)
            
        if ((len(actual_is_covered) != len(t2vs)) or \
            (len(actual_is_covered) != len(actual_is_precise))):
            query_id = df['query_id'].iloc[i]
            op = df['op'].iloc[i]
            idxs_ops_of_interest.append((query_id, op))
            is_precise = eval(df['precise_citations'].iloc[i])
            new_is_precise = []
            for j in range(len(is_covered)):
                coverage_item = is_covered[j]
                if (coverage_item['coverage']!=-1):
                    new_is_precise.append(is_precise[j])
                else:
                    new_is_precise.append({"annotations":[],"sentence_id":coverage_item["sentence_id"]})
    
            if (len(is_covered)==len(t2vs)): # all of the sentences have a citation, but some weren't displayed properly
                new_t2vs = []
                for j in range(len(is_covered)):
                    coverage_item = is_covered[j]
                    if (coverage_item['coverage']!=-1):
                        new_t2vs.append(t2vs[j])

            elif (len(is_covered) > len(t2vs)): # some of the sentences had no citations and some weren't displayed properly
                new_t2vs = []
                k = 0 # will be used to index into t2vs
                for j in range(len(is_covered)):
                    coverage_item = is_covered[j]
                    precision_item = is_precise[j]
                    if (not ((coverage_item['coverage']==-1) and (len(precision_item['annotations'])==0))): # if T2V recorded for this sentence
                        if (coverage_item['coverage']!=-1): # if the sentence was displayed correctly
                            new_t2vs.append(t2vs[k]) # keep the corresponding t2v
                        k += 1
            else:
                print('!!!!!! not handled')    
                
            
            df['precise_citations'].iloc[i] = str(new_is_precise)
            df['t2v_coverage'].iloc[i] = str(new_t2vs)

    # print('Corrected:', idxs_ops_of_interest)
    return df

def check_annotations(df):
    # checks whether all precision and T2V annotations are consistent with the coverage dict
    df = df[df['op']!='Snippet']
    
    idxs_ops_of_interest = []
    for i in range(len(df)):
        t2vs = eval(df['t2v_coverage'].iloc[i])
        is_covered = eval(df['is_covered'].iloc[i])
        is_precise = eval(df['precise_citations'].iloc[i])
        actual_is_covered = []
        actual_is_precise = []
        for j in range(len(is_covered)):
            cov_item = is_covered[j]
            if (cov_item['coverage'] != -1):
                actual_is_covered.append(cov_item)
            prec_item = is_precise[j]
            if (len(prec_item['annotations'])!=0):
                actual_is_precise.append(prec_item)
        if (len(actual_is_covered) != len(t2vs)):
            print(df['query_id'].iloc[i])
            print('is_precise', is_precise)
            print('actual_is_covered len', len(actual_is_covered))
            print('actual_is_covered', actual_is_covered)
            print('is_covered len', len(is_covered))
            print('t2vs', t2vs)
            print('t2vs len', len(t2vs))

def add_uf_results(df, baselines, data_str, verbose=False):
    uf_fp_dict = {
        'nq':'../mturk_results/unprocessed_results/mturk_all_nq_uf_annotations_rows.csv',
        'mh':'../mturk_results/unprocessed_results/mturk_all_mh_uf_annotations_rows.csv',
        'eli3':'../mturk_results/unprocessed_results/mturk_all_eli3_uf_annotations_rows.csv',
        'mash':'../mturk_results/unprocessed_results/mturk_all_mash_uf_annotations_rows.csv'
    }
    uf_fp = uf_fp_dict[data_str]
    uf_df = pd.read_csv(uf_fp, index_col=False)
    if (baselines):
        uf_df = uf_df[uf_df['op'].isin(['Gemini', 'Post Hoc', 'Quoted Reeval'])][['human_fluency_rating','human_utility_rating','op','query_id']]
    else:
        uf_df = uf_df[uf_df['op'].isin(['Snippet','Quoted','Paraphrased','Entailed','Abstractive'])][['human_fluency_rating','human_utility_rating','op','query_id']]

    if (verbose): # print out all of the (query_id, op) tuples that are missing an anotation
        missing_items = []
        for i in range(len(df)):
            query_id = df['query_id'].iloc[i]
            op = df['op'].iloc[i]
            if (len(uf_df[(uf_df['query_id']==query_id)&(uf_df['op']==op)])==0):
                missing_items.append((query_id, op))
        print('Missing:', len(missing_items)) # the reeval did not include the quoted reeval examples, nor were all original samples evaluated due to per-annotator sampling constraints 
        print('Found:', len(df)-len(missing_items))
        print()
                
    df = df.rename({'human_fluency_rating': 'first_human_fluency_rating', 'human_utility_rating': 'first_human_utility_rating'}, axis='columns')
    df = pd.merge(df, uf_df, how='inner', on=['query_id', 'op'])
    return df

def check_trimmed_annotations_soft(df, n):
    query_counts_by_method = df.groupby('op')['query_id'].count()
    methods = df.groupby('op')['query_id'].count().index
    for i in range(len(query_counts_by_method)):
        assert query_counts_by_method.iloc[i] >= n
        if (query_counts_by_method.iloc[i] < n):
            print('\tNeed more for '+methods[i]+': '+str(n-query_counts_by_method.iloc[i]))

# The Vertex API sometimes parses sentences differently than our implementation. We resolve these conflicts by hand.
# The cases below are conflict resolutions.
# For cases where one or more sentences don't require citation, add their "Sentences Need Citation" label to a dict below
baseline_corrections = {'mash': {
                10: [True, True, True, True, True, True], # ('Gemini', '105')
                22: [True, True, True, True, True, True], # ('Gemini', '125')
                63: [True]*7, # ('Gemini', '159')
              }, 
               'eli3': {
                   45: [False, True, True, True, True], # ('Gemini', '574')
                   64: [True, True, True, True], # ('Gemini', '529')
                   115: [True, True], # ('Gemini', '546')
                   },
               'nq': {
                   252: [True]*3, # ('Gemini', '293')
               },
               'mh': {}
              }

op_corrections = {'mash': {
                    88: [True]*9, # ('Quoted', '139')
                    118: [True]*12, # ('Quoted', '143')
                    149: [True]*12, # ('Quoted', '123')
                    225: [True]*6, # ('Quoted', '130')
                    320: [True]*5, # ('Paraphrased', '130')
                    361: [True]*4, # ('Abstractive', '130')
                    561: [False]+[True]*9, # ('Paraphrased', '135')
                    565: [True]*5, # ('Quoted', '207')
} , 
               'eli3': {
                   24: [True]*5, # ('Quoted', '493')
                   159: [True]*5, # ('Paraphrased', '493')
                   179: [True]*2, # ('Quoted', '504')
                   428: [True]*5, # ('Entailed', '493')
                   500: [True]*2, # ('Quoted', '546')
                   522: [True], # ('Abstractive', '593')
               },
               'nq': {
                     18: [True], # ('Abstractive', '276')
                     58: [True], # ('Abstractive', '293')
                     122: [True, True, True, True, True, True, True, True], # ('Quoted', '389')
                     151: [True], # ('Quoted', '293')
                     260: [True], # ('Entailed', '293')
                     315: [True], # ('Quoted', '343')
                     336: [True], # ('Paraphrased', '293')
                     426: [True, True, True, True], # ('Quoted', '408')
                     579: [True], # ('Paraphrased', '340')
               },
               'mh': {
                   27: [True, True, True, True], # ('Quoted', '163')
                   55: [True, True, True], # ('Quoted', '201')
                   160: [True, True], # ('Quoted', '76')
                   189: [True, True], # ('Paraphrased', '155')
                   407: [True], # ('Abstractive', '116')
                   481: [True, True, True], # ('Quoted', '116')
                   510: [True, True], # ('Quoted', '155')
                   589: [True, True], # ('Quoted', '197')
                   609: [True, True, True, True], # ('Quoted', '98')
               }
              }
quoted_baseline_corrections = {
            'nq': {
                0:[True, True, True, True], # ('Quoted', '408')
                18:[True]*8, # ('Quoted', '389')
                33:[True], # ('Quoted', '340')
                70: [True], # ('Quoted', '343')
                75: [True], # ('Quoted', '293')
               },
                'eli3': {
                   20: [True, True], # ('Quoted', '504') 
                   113: [True, True], # ('Gemini', '546')
                   135: [True]*5, # ('Quoted', '493')
                   },
            'mash': {
                18: [True]*8, # ('Quoted', '217')
                45: [True]*13, # ('Quoted', '62')
                53: [True]*12, # ('Quoted', '143')
                71: [True]*9, # ('Quoted', '139')
                96: [True]*6, # ('Quoted', '130')
                100: [True, True, True, True, True, True, True, True, False, False, False, True], # ('Quoted', '123')
                132: [True]*7, # ('Quoted', '76')
                133: [True]*27, # ('Quoted', '210')
              }, 
               
               'mh': {
                   8: [True]*4, # ('Quoted', '163')
                   30: [True]*2, # ('Quoted', '197')
                   65: [True]*2, # ('Quoted', '155')
                   118: [True]*2, # ('Quoted', '76')
                   121: [True]*3, # ('Quoted', '116')
                   143: [True]*4, # ('Quoted', '98')
                   147: [True]*3, # ('Quoted', '147')
               }
              }

def fix_mismatches(df, corrections_dict):
    for i in range(len(df)):
        if (df['op'].iloc[i] == 'Snippet'):
            continue
        gpt4_sentence_count = len(eval(df['Sent'].iloc[i]))
        vertex_sentence_count = len(eval(df['Sentences Need Citation'].iloc[i]))
        if (gpt4_sentence_count != vertex_sentence_count):
            if ((all(eval(df['Sentences Need Citation'].iloc[i]))) & (vertex_sentence_count > gpt4_sentence_count)):
                 df.loc[i, 'Sentences Need Citation'] = str([True]*gpt4_sentence_count) 
            else:
                if (i not in corrections_dict):
                    print(i)
                    continue
                df.loc[i, 'Sentences Need Citation'] = str(corrections_dict[i])

    for i in range(len(df)):
        if (df['op'].iloc[i] == 'Snippet'):
            continue
        gpt4_sentence_count = len(eval(df['Sent'].iloc[i]))
        vertex_sentence_count = len(eval(df['Sentences Need Citation'].iloc[i]))
        assert gpt4_sentence_count == vertex_sentence_count
            
    return df

def make_only_needs_citation(df):
    # Remove the precision, coverage, and T2V data for sentences that do not require citation
    # Clean up the precision and coverage annotations, given the "needs citation labels"
    for i in range(len(df)):
        if (df['op'].iloc[i] == 'Snippet'):
            continue
            
        sentences_need_citation = eval(df['Sentences Need Citation'].iloc[i])
        
        # first the coverage
        is_covered = eval(df['is_covered'].iloc[i])
        new_is_covered = []
        for j in range(len(is_covered)):
            sentence_idx = is_covered[j]['sentence_id']
            if (sentences_need_citation[sentence_idx]):
                new_is_covered.append(is_covered[j])
        df.loc[i, 'is_covered'] = str(new_is_covered)
        assert len(eval(df['is_covered'].iloc[i])) == np.sum(sentences_need_citation)
    
        # now the precision
        is_precise = eval(df['precise_citations'].iloc[i])
        new_is_precise = []
        for j in range(len(is_precise)):
            item = is_precise[j]
            sentence_idx = item['sentence_id']
            if (sentences_need_citation[sentence_idx]):
                new_is_precise.append(item)
        df.loc[i, 'precise_citations'] = str(new_is_precise)
        assert len(eval(df['precise_citations'].iloc[i])) == np.sum(sentences_need_citation)
    
        # now T2V
        t2vs = eval(df['t2v_coverage'].iloc[i])
            
        # keep the T2V values that correspond to coverage values that a) exist and b) need citation
        actual_coverage_items = []
        for item in is_covered:
            if (item['coverage'] != -1):
                actual_coverage_items.append(item)

        
        new_t2vs = []
        for j in range(len(actual_coverage_items)):
            sentence_idx = actual_coverage_items[j]['sentence_id']
            if (sentences_need_citation[sentence_idx]):
                new_t2vs.append(t2vs[j])
        df.loc[i, 't2v_coverage'] = str(new_t2vs)

        # Now, handle the citations dict
        actual_citations_dict = {}
        citations_dict = eval(df['Citation Dict'].iloc[i])
        for k in citations_dict.keys():
            if (sentences_need_citation[int(k)]):
                actual_citations_dict[k] = citations_dict[k]
        df.loc[i, 'Citation Dict'] = str(actual_citations_dict)
    return df

def check_needs_citation(df):
    for i in range(len(df)):
        if (df['op'].iloc[i] == 'Snippet'):
            continue
        needs_citation_ls = eval(df['Sentences Need Citation'].iloc[i])
        is_covered_ls = eval(df['is_covered'].iloc[i])
        is_precise_ls = eval(df['precise_citations'].iloc[i])
        assert np.sum(needs_citation_ls) == len(is_covered_ls)
        assert  np.sum(needs_citation_ls) == len(is_precise_ls)
        t2vs = eval(df['t2v_coverage'].iloc[i])
        assert len(is_covered_ls) >= len(t2vs)

def main():
    np.random.seed(0)

    op_fps = {'nq': '../mturk_results/unprocessed_results/nq_mturk_with_needs_citation_labels2',
        'mh': '../mturk_results/unprocessed_results/mh_mturk_with_needs_citation_labels',
        'mash': '../mturk_results/unprocessed_results/mash_mturk_with_needs_citation_labels',
        'eli3': '../mturk_results/unprocessed_results/eli3_mturk_with_needs_citation_labels',
        }
    baseline_fps = {'nq': '../mturk_results/unprocessed_results/nq_baseline_mturk_with_needs_citation_labels',
        'mh': '../mturk_results/unprocessed_results/mh_baseline_mturk_with_needs_citation_labels',
        'mash': '../mturk_results/unprocessed_results/mash_baseline_mturk_with_needs_citation_labels',
        'eli3': '../mturk_results/unprocessed_results/eli3_baseline_mturk_with_needs_citation_labels',
        }

    # load the unprocessed data from MTurk and Vertex
    baseline_dfs = {}
    op_dfs = {}
    for data_str in baseline_fps.keys():
        baseline_dfs[data_str] = pd.read_csv(baseline_fps[data_str]+'.csv', index_col=False)
        op_dfs[data_str] = pd.read_csv(op_fps[data_str]+'.csv', index_col=False)

    # Remove T2V and Precision annotations corresponding to cases where the annotation interface was unable to display a sentence for citation annotation
    for baselines in [False, True]:
        for data_str in baseline_fps.keys():
            if (baselines):
                df_dict = baseline_dfs
            else:
                df_dict = op_dfs
                
            df = df_dict[data_str]
            df_dict[data_str] = remove_irrelevant_t2v_and_precision_annotations(df)
            check_annotations(df_dict[data_str])
    print('Removed T2V and precision annotations corresponding to cases where annotation did not occur.')

    # Drop instances with scraping issues from Gemini Mash
    # Some were accidentally included in the mturk annotation. They only happened for the MASH dataset.
    df = baseline_dfs['mash']
    idx_to_drop = []
    for i in range(len(df)):
        output = df['Output (cited)'].iloc[i]
        if ('...' in output):
            # print(df['query_id'].iloc[i])
            # print(output)
            # print()
            idx_to_drop.append(i)

    for i in idx_to_drop:
        df = df.drop(i)
        
    for i in range(len(df)):
        output = df['Output (cited)'].iloc[i]
        if (df['op'].iloc[i] != 'Gemini'):
            continue
        if ('...' in output):
            print(df['query_id'].iloc[i])
            print(output)
            print()
    baseline_dfs['mash'] = df

    # Add in the utility and fluency results
    # We evaluated utility and fluency over all of the operating points simultaneously to avoid batching effects over subsets of the operating points. Here, we merge those results in.
    for data_str in baseline_fps.keys():
        for baselines in [False, True]:
            if (baselines):
                s = ' baselines'
            else:
                s = ' OPs'
            if (baselines):
                df_dict = baseline_dfs
            else:
                df_dict = op_dfs
            df = df_dict[data_str]
            df = add_uf_results(df, baselines, data_str, verbose=False)
            df_dict[data_str] = df

            if (baselines):
                fp = baseline_fps[data_str]
            else:
                fp = op_fps[data_str]
            fp = fp.split('/')[-1]
            save_path = '../mturk_results/intermediate_results/'+fp+'_cleaned_minus_one_coverage_UF.csv' # Used later for data over all sentences (requiring and not requiring citation)
            df.to_csv(save_path)
                
            print('Added fluency and perceived utility results for '+data_str+s)

    # Ensure there are 120 queries per method
    for data_str in baseline_fps.keys():
        for baselines in [False, True]:
            if (baselines):
                df_dict = baseline_dfs
            else:
                df_dict = op_dfs
            df = df_dict[data_str]
            check_trimmed_annotations_soft(df, 120)

    # Fix the mismatches between vertex and the current sentence count
    for baselines in [False, True]:
        for data_str in baseline_fps.keys():
            if (baselines):
                df_dict = baseline_dfs
            else:
                df_dict = op_dfs
            df = df_dict[data_str]
            
            # assign the "needs citation" labels for the mismatch case from above
            if (baselines):
                corrections_dict = baseline_corrections
            else:
                corrections_dict = op_corrections
                
            df = fix_mismatches(df, corrections_dict[data_str])  
            df_dict[data_str] = df
    print('Fixed mismatches between the Vertex API and the annotated sentence count')

    for baselines in [False, True]:
        for data_str in baseline_fps.keys():
            if (baselines):
                df_dict = baseline_dfs
            else:
                df_dict = op_dfs
            df = df_dict[data_str]
            # clean up the coverage, precision, and T2V annotations, given the "needs citation labels"
            df = make_only_needs_citation(df)

            # check that only the relevant sentences are kept
            check_needs_citation(df)
            
            df_dict[data_str] = df
    print('Discard coverage, precision, and T2V annotations for sentences that do not require citation')

    # Filter the baseline quoted T2V annotations to only those that require citation
    columns_to_remove = ['n-gram precision', 'Citation Count', 'n sentences', 'uuid', 'first_human_fluency_rating', 'first_human_utility_rating', 't2v_precision', 'Fluency Rating', 'Perceived Utility Rating']
    datasets = ['nq', 'eli3', 'mash', 'mh']
    dataset_names = ['nq', 'eta3g', 'mash', 'mh']
    for data_str, data_name in zip(datasets, dataset_names):
        fp = baseline_fps[data_str]
        baseline_df = pd.read_csv(fp+'.csv', index_col=False).reset_index(drop=True)
        quoted_baseline_df = baseline_df[baseline_df['op']=='Quoted'].reset_index(drop=True)
        # Fix the mismatches between vertex and the current sentence count
        quoted_baseline_df = fix_mismatches(quoted_baseline_df, quoted_baseline_corrections[data_str])
        # remove the T2V, coverage, and precision annotations
        quoted_baseline_df = make_only_needs_citation(quoted_baseline_df)
        check_needs_citation(quoted_baseline_df)
        # rename 'Quoted' to 'Quoted Reeval'
        quoted_baseline_df['op'] = 'Quoted Reeval'
        # concatenate with the other baseline results
        # baseline_df = pd.read_csv(baseline_fps[data_str]+'_cleaned_trimmed_needs_citation_only_NEW.csv', index_col=False)
        baseline_df = baseline_dfs[data_str]
        baseline_df = pd.concat([baseline_df, quoted_baseline_df])
        # drop unused columns
        baseline_df = baseline_df.drop(columns=columns_to_remove)
        baseline_df = baseline_df.loc[:, ~baseline_df.columns.str.startswith("Unnamed")]
        # check again that only sentences requiring citation are kept
        check_needs_citation(baseline_df)
        # save
        save_path = '../mturk_results/processed_results/'+data_name+'_mturk_eval_byQueryOP_baseline_needs_citation.csv' # this is a file used in plotting_by_metric
        if (not os.path.isfile(save_path)):
            baseline_df.to_csv(save_path)
            print('Saved to '+save_path)

    # Save op files in the right folder with consistent naming
    for data_str, data_name in zip(datasets, dataset_names):
        op_df = op_dfs[data_str]
        op_df = op_df.drop(columns=columns_to_remove)
        op_df = op_df.loc[:, ~op_df.columns.str.startswith("Unnamed")]
        check_needs_citation(op_df) # check again that only sentences requiring citation are kept
        save_path = '../mturk_results/processed_results/'+data_name+'_mturk_eval_byQueryOP_ops_needs_citation.csv' # this is a file used in plotting_by_metric
        if (not os.path.isfile(save_path)):
            op_df.to_csv(save_path)
            print('Saved to '+save_path)

    # Add back the Quoted Reeval to the results over all sentences (not just those that require citation)
    tag = '_cleaned_minus_one_coverage_UF'
    mturk_baseline_fps_all = {
        'NQ': '../mturk_results/intermediate_results/nq_baseline_mturk_with_needs_citation_labels'+tag,
        'Eta3G': '../mturk_results/intermediate_results/eli3_baseline_mturk_with_needs_citation_labels'+tag,
        'MH': '../mturk_results/intermediate_results/mh_baseline_mturk_with_needs_citation_labels'+tag,
        'MASH': '../mturk_results/intermediate_results/mash_baseline_mturk_with_needs_citation_labels'+tag,
    }

    mturk_op_fps_all = {
        'NQ': '../mturk_results/intermediate_results/nq_mturk_with_needs_citation_labels2'+tag,
        'Eta3G': '../mturk_results/intermediate_results/eli3_mturk_with_needs_citation_labels'+tag,
        'MH': '../mturk_results/intermediate_results/mh_mturk_with_needs_citation_labels'+tag,
        'MASH': '../mturk_results/intermediate_results/mash_mturk_with_needs_citation_labels'+tag,
    }
    mturk_fp = '../mturk_results/'
    for k1, k2 in zip(datasets, list(mturk_baseline_fps_all.keys())):
        baseline_df = pd.read_csv(baseline_fps[k1]+'.csv', index_col=False).reset_index(drop=True)
        quoted_baseline_df = baseline_df[baseline_df['op']=='Quoted'].reset_index(drop=True)
        quoted_baseline_df['op'] = 'Quoted Reeval'
        baseline_df = pd.read_csv(mturk_fp+mturk_baseline_fps_all[k2]+'.csv', index_col=False)
        baseline_df = pd.concat([baseline_df, quoted_baseline_df])
        baseline_df = baseline_df.drop(columns=columns_to_remove)
        baseline_df = baseline_df.loc[:, ~baseline_df.columns.str.startswith("Unnamed")]
        save_path = '../mturk_results/processed_results/'+k2.lower()+'_mturk_eval_byQueryOP_baseline_all.csv' # this is a file used in plotting_by_metric
        baseline_df.to_csv(save_path)
        print('Saved to '+save_path)
        op_df = pd.read_csv(mturk_fp+mturk_op_fps_all[k2]+'.csv', index_col=False)
        save_path = '../mturk_results/processed_results/'+k2.lower()+'_mturk_eval_byQueryOP_ops_all.csv' # this is a file used in plotting_by_metric
        op_df.to_csv(save_path)
        print('Saved to '+save_path)
        print()

if __name__ == "__main__":
    main()


