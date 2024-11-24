import os
import pandas as pd

datasets = ['NQ', 'Eta3G', 'MH', 'MASH']
all_ops = ['Snippet', 'Quoted', 'Quoted Reeval', 'Paraphrased', 'Entailed', 'Abstractive', 'Gemini', 'Post Hoc']

pilot_fp = '../pilot_results/'
mturk_fp = '../mturk_results/'

pilot_op_input_fps = {
    'NQ': 'nq_pilot_eval_0_20_byQueryOP_inputs.csv',
    'Eta3G': 'eta3g_pilot_eval_0_20_byQueryOP_inputs.csv',
    'MH': 'mh_pilot_eval_0_20_byQueryOP_inputs.csv',
    'MASH': 'mash_pilot_eval_0_20_byQueryOP_inputs.csv',
}

pilot_op_output_fps = {
    'NQ': 'nq_pilot_eval_0_20_byQueryOP_outputs.csv',
    'Eta3G': 'eta3g_pilot_eval_0_20_byQueryOP_outputs.csv',
    'MH': 'mh_pilot_eval_0_20_byQueryOP_outputs.csv',
    'MASH': 'mash_pilot_eval_0_20_byQueryOP_outputs.csv',
}

pilot_baseline_input_fps = {
    'NQ': 'nq_pilot_eval_0_20_byQueryOP_baseline_inputs.csv',
    'Eta3G': 'eta3g_pilot_eval_0_20_byQueryOP_baseline_inputs.csv',
    'MH': 'mh_pilot_eval_0_20_byQueryOP_baseline_inputs.csv',
    'MASH': 'mash_pilot_eval_0_20_byQueryOP_baseline_inputs.csv',
}

pilot_baseline_output_fps = {
    'NQ': 'nq_pilot_eval_0_20_byQueryOP_baseline_outputs.csv',
    'Eta3G': 'eta3g_pilot_eval_0_20_byQueryOP_baseline_outputs.csv',
    'MH': 'mh_pilot_eval_0_20_byQueryOP_baseline_outputs.csv',
    'MASH': 'mash_pilot_eval_0_20_byQueryOP_baseline_outputs.csv',
}

mturk_op_fps = {
    'NQ': 'nq_mturk_eval_byQueryOP_ops',
    'Eta3G': 'eta3g_mturk_eval_byQueryOP_ops',
    'MH': 'mh_mturk_eval_byQueryOP_ops',
    'MASH': 'mash_mturk_eval_byQueryOP_ops',
}

mturk_baseline_fps = {
    'NQ': 'nq_mturk_eval_byQueryOP_baseline',
    'Eta3G': 'eta3g_mturk_eval_byQueryOP_baseline',
    'MH': 'mh_mturk_eval_byQueryOP_baseline',
    'MASH': 'mash_mturk_eval_byQueryOP_baseline',
}
# # _cleaned_minus_one_coverage_UF
# mturk_op_fps_all = {
#     'NQ': 'nq_mturk_with_needs_citation_labels2_cleaned_minus_one_coverage_UF',
#     'Eta3G': 'eli3_mturk_with_needs_citation_labels_cleaned_minus_one_coverage_UF',
#     'MH': 'mh_mturk_with_needs_citation_labels_cleaned_minus_one_coverage_UF',
#     'MASH': 'mash_mturk_with_needs_citation_labels_cleaned_minus_one_coverage_UF',
# }

# mturk_baseline_fps_all = {
#     'NQ': 'nq_baseline_mturk_with_needs_citation_labels_cleaned_minus_one_coverage_UF',
#     'Eta3G': 'eli3_baseline_mturk_with_needs_citation_labels_cleaned_minus_one_coverage_UF',
#     'MH': 'mh_baseline_mturk_with_needs_citation_labels_cleaned_minus_one_coverage_UF',
#     'MASH': 'mash_baseline_mturk_with_needs_citation_labels_cleaned_minus_one_coverage_UF',
# }

def load_pilot_results_for_ds(k):
    # Returns a df of the pilot results for the specified dataset
    op_df_inputs = pd.read_csv(os.path.join(pilot_fp, pilot_op_input_fps[k]), index_col=False) 
    op_df_inputs.rename(columns={'ID': 'query_id'}, inplace=True)
    
    baseline_df_inputs = pd.read_csv(os.path.join(pilot_fp, pilot_baseline_input_fps[k]), index_col=False) 
    baseline_df_inputs.rename(columns={'ID': 'query_id'}, inplace=True)
    # baseline_df_inputs = add_pilot_baseline_quoted_rows(baseline_df_inputs)

    df_inputs = pd.concat([op_df_inputs, baseline_df_inputs])
    
    op_df_outputs = pd.read_csv(os.path.join(pilot_fp, pilot_op_output_fps[k]), index_col=False) 
    baseline_df_outputs = pd.read_csv(os.path.join(pilot_fp, pilot_baseline_output_fps[k]), index_col=False) 
    # baseline_df_outputs = add_pilot_baseline_quoted_rows(baseline_df_outputs)
    
    df_outputs = pd.concat([op_df_outputs, baseline_df_outputs])
    
    results = pd.merge(df_inputs, df_outputs, how='inner', on=['query_id', 'op'])
    results['dataset'] = k
    return results

def load_all_pilot_results():
    all_results = pd.DataFrame()
    results_dict = {}
    for k in datasets:
        results_k = load_pilot_results_for_ds(k)
        all_results = pd.concat([all_results, results_k])
        results_dict[k] = results_k
    return all_results, results_dict

def load_mturk_results_for_ds(k, needs_citation_only=True):
    if (needs_citation_only):
        tag = '_needs_citation'
    else:
        tag = '_all'
        
    op_df = pd.read_csv(mturk_fp+mturk_op_fps[k]+tag+'.csv', index_col=False)
    baseline_df = pd.read_csv(mturk_fp+mturk_baseline_fps[k]+tag+'.csv', index_col=False)
    dataset_results = pd.concat([op_df, baseline_df])
    new_dataset_results = pd.DataFrame()
    for op in all_ops:
        dataset_op_results = dataset_results[dataset_results['op']==op]
        dataset_op_results = dataset_op_results.sort_values('query_id')
        if (op != 'Quoted Reeval'):
            dataset_op_results = dataset_op_results.iloc[:120]
        new_dataset_results = pd.concat([new_dataset_results, dataset_op_results], ignore_index=True)
    if (needs_citation_only):
        assert len(new_dataset_results) >= 120*len(all_ops)
    else:
        assert len(new_dataset_results) >= 120*(len(all_ops)-1)
    new_dataset_results['dataset'] = k
    return new_dataset_results

def load_all_mturk_results(needs_citation_only=True):
    all_results = pd.DataFrame()
    results_dict = {}
    for k in datasets:
        results_k = load_mturk_results_for_ds(k, needs_citation_only)
        all_results = pd.concat([all_results, results_k])
        results_dict[k] = results_k
    return all_results, results_dict