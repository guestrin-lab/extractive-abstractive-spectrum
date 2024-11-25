import os
import argparse
import json
import pandas as pd
import numpy as np

def main(args):

    # Load file
    with open(args.filename, "r") as f:
        results = [json.loads(line) for line in f]
    df = pd.DataFrame(results)

    abstained = df[df['Quoted Output'].apply(lambda x: 'Insufficient information to generate a grounded response' in x)]
    print('Abstention rate:', len(abstained)/len(df))
    df = df[df['Quoted Output'].apply(lambda x: 'Insufficient information to generate a grounded response' not in x)]

    # calculate number of sentences
    df['Quoted n sentences'] = df['Quoted Sent'].apply(len)
    df['Paraphrased n sentences'] = df['Paraphrased Sent'].apply(len)
    df['Entailed n sentences'] = df['Entailed Sent'].apply(len)
    df['Abstractive n sentences'] = df['Abstractive Sent'].apply(len)

    # Expand each query into its operating point responses
    op_strs = ['Snippet ', 'Quoted ', 'Paraphrased ', 'Entailed ', 'Abstractive ']
    op_col_suffixes_with_snippet = ['Output (cited)', 'Output', 'Fluency Rating', 'Perceived Utility Rating', 'n-gram precision']
    op_col_suffixes_without_snippet = ['Sent (cited)', 'Sent', 'Citation Dict', 'Citation Count', 'n sentences']
    cols_for_all_op = ['ID', 'All Sources', 'All URLs', 'All Sources (cited)', 'Used Sources (cited)', 'Question']
    snippet_df = df[['ID']]
    quoted_df = df[['ID']]
    pp_df = df[['ID']]
    ent_df = df[['ID']]
    abs_df = df[['ID']]
    op_dfs_without_snippet = [quoted_df, pp_df, ent_df, abs_df]

    # Add shared columns to all OPs
    for op_df in [snippet_df]+op_dfs_without_snippet:
        for col in cols_for_all_op:
            op_df[col] = df[col]

    # Add shared columns over non-snippet OPs
    for i in range(len(op_dfs_without_snippet)):
        op_df = op_dfs_without_snippet[i]
        op_str = op_strs[1:][i]
        for col_suffix in op_col_suffixes_with_snippet:
            op_df[col_suffix] = df[op_str+col_suffix]
        for col_suffix in op_col_suffixes_without_snippet:
            op_df[col_suffix] = df[op_str+col_suffix]
        op_df['op'] = op_str[:-1] # just cut out the space

    # Add shared columns to snippet OPs. Some columns will be nans, and that's ok!
    for col_suffix in op_col_suffixes_with_snippet:
        snippet_df[col_suffix] = df['Snippet '+col_suffix]
    for col_suffix in op_col_suffixes_without_snippet:
        snippet_df[col_suffix] = None
    snippet_df['op'] = 'Snippet'

    # Concatenate all of the op dfs into one df of responses
    combined_df = pd.concat([snippet_df]+op_dfs_without_snippet, ignore_index=True)

    # shuffle
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    # save new file
    save_path = args.filename[:-6]+'_byQueryOP.csv'
    if (os.path.isfile(save_path)):
        print('Did not save file; it would overwrite the file '+save_path+' that potentially contains annotations. Remove or rename that file and rerun this script.')
    else:
        combined_df.to_csv(save_path)
    print('Saved to '+save_path)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default=None, type=str)
    args = parser.parse_args()
    main(args)