import os
import re 
import ast
import numpy as np
import pandas as pd 
from tqdm import tqdm

import asyncio
import time
import argparse

from statsmodels.stats.proportion import proportion_confint


def prompt_model(prompt, answer_choices):
    message=[{"role": "assistant", "content": "a helpful expert"}, {"role": "user", "content": prompt}]
    rating_output_text = -1
    num_attempts = 0
    while ((rating_output_text not in answer_choices) and (num_attempts < 5)):
        rating_output = model.chat.completions.create(
                model="gpt-5-2025-08-07", 
                messages = message,
        )
        rating_output_text = rating_output.choices[0].message.content
        num_attempts += 1
    return rating_output_text

def run_pairwise_comparisons(k, model):

    # Open up the human-generated and model-scored quoted responses
    human_quoted_df = pd.read_csv("fluency_utility_for_human_quoted_responses.csv")
    human_quoted_df = human_quoted_df[human_quoted_df['Answer_y']==human_quoted_df['Answer_x']]
    human_quoted_df = human_quoted_df.rename(columns={'Answer_y': 'Human Output'})
    human_quoted_df = human_quoted_df.drop(columns=['Answer_x'])
    human_quoted_df = human_quoted_df[human_quoted_df['Reject']=='False']

    human_quoted_df = human_quoted_df.groupby('Question').filter(lambda g: len(g) >= k)
    human_quoted_df = human_quoted_df.groupby('Question').sample(n=k, random_state=0)

    human_quoted_df = human_quoted_df.groupby('Dataset').sample(n=90, random_state=0)

    print("Number of unique questions per dataset:", human_quoted_df.groupby('Dataset').count())

    human_quoted_df['avg_auto_rating'] = (human_quoted_df['auto_fluency_rating'] + human_quoted_df['auto_utility_rating']) / 2

    idx = human_quoted_df.groupby('Question')['avg_auto_rating'].idxmax()
    fluency_human_quoted_df = human_quoted_df.loc[idx, ['Question', 'auto_fluency_rating', 'Human Output', 'Dataset']].reset_index(drop=True)

    idx = human_quoted_df.groupby('Question')['avg_auto_rating'].idxmax()
    utility_human_quoted_df = human_quoted_df.loc[idx, ['Question', 'auto_utility_rating', 'Human Output', 'Dataset']].reset_index(drop=True)

    print(utility_human_quoted_df.groupby('Dataset').size())

    fluency_human_quoted_df = fluency_human_quoted_df.rename(columns={'auto_utility_rating': 'Human auto_utility_rating',
                                                      'auto_fluency_rating': 'Human auto_fluency_rating'})

    utility_human_quoted_df = utility_human_quoted_df.rename(columns={'auto_utility_rating': 'Human auto_utility_rating',
                                                      'auto_fluency_rating': 'Human auto_fluency_rating'})

    # Open up the model-generated and model-scored quoted responses
    model_quoted_df = pd.read_csv(f"mturk_results/autoEval_{model}_eli5_nq_byQueryOP.csv")
    model_quoted_df = pd.concat([model_quoted_df, pd.read_csv(f"mturk_results/autoEval_{model}_mash_byQueryOP.csv")])
    model_quoted_df = pd.concat([model_quoted_df, pd.read_csv(f"mturk_results/autoEval_{model}_multihop_byQueryOP.csv")])
    model_quoted_df = pd.concat([model_quoted_df, pd.read_csv(f"mturk_results/autoEval_{model}_nq_byQueryOP.csv")])
    model_quoted_df = model_quoted_df[model_quoted_df['op']=='Quoted']
    model_quoted_df = model_quoted_df.rename(columns={'Output': 'Model Output'})
    model_quoted_df = model_quoted_df[['Question', 'Model Output', 'auto_fluency_rating', 'auto_utility_rating']]
    model_quoted_df = model_quoted_df.rename(columns={'auto_utility_rating': 'Model auto_utility_rating',
                                                      'auto_fluency_rating': 'Model auto_fluency_rating'})
    
    # Merge
    utility_df = utility_human_quoted_df.merge(model_quoted_df, on='Question', how='left')
    fluency_df = fluency_human_quoted_df.merge(model_quoted_df, on='Question', how='left')
    
    print(f'Average model fluency score: {np.mean(fluency_df["Model auto_fluency_rating"])}')
    print(f'Average model utility score: {np.mean(utility_df["Model auto_utility_rating"])}')
    print(f'Average human fluency score: {np.mean(fluency_df["Human auto_fluency_rating"])}')
    print(f'Average human utility score: {np.mean(utility_df["Human auto_utility_rating"])}')

    # Tabulate win rates from pre-computed automatic scores
    num_model_utility_wins = np.sum(utility_df['Model auto_utility_rating'] > utility_df['Human auto_utility_rating'])
    num_model_fluency_wins = np.sum(fluency_df['Model auto_fluency_rating'] > fluency_df['Human auto_fluency_rating'])

    num_utility_ties = np.sum(utility_df['Model auto_utility_rating'] == utility_df['Human auto_utility_rating'])
    num_fluency_ties = np.sum(fluency_df['Model auto_fluency_rating'] == fluency_df['Human auto_fluency_rating'])

    model_utility_win_rate = (num_model_utility_wins + 0.5 * num_utility_ties) / len(utility_df)
    model_fluency_win_rate = (num_model_fluency_wins + 0.5 * num_fluency_ties) / len(fluency_df)
    n = len(utility_df)

    utility_ci_low, utility_ci_high = proportion_confint(
                                    count=int(num_model_utility_wins + 0.5 * num_utility_ties),
                                    nobs=n,
                                    alpha=0.05,
                                    method='normal',
                                )
    fluency_ci_low, fluency_ci_high = proportion_confint(
                                    count=int(num_model_fluency_wins + 0.5 * num_fluency_ties),
                                    nobs=n,
                                    alpha=0.05,
                                    method='normal',
                                )

    print(f"Model Utility Win Rate: {model_utility_win_rate} (95% CI: {utility_ci_low}, {utility_ci_high})")
    print(f"Model Fluency Win Rate: {model_fluency_win_rate} (95% CI: {fluency_ci_low}, {fluency_ci_high})")

    return model_utility_win_rate, utility_ci_low, utility_ci_high, model_fluency_win_rate, fluency_ci_low, fluency_ci_high

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=1, help='Number of responses per query to compare')
    args = parser.parse_args()

    results_dict = {'Model': [],
                    'Fluency': [],
                    'Perceived Utility': [],
                    }
    for model_str in ['gpt4', 'gpt5', 'sonnet4.5']:
        utility_model_win_rate, u_low, u_high, fluency_model_win_rate, f_low, f_high = run_pairwise_comparisons(args.k, model_str)
        results_dict['Model'].append(model_str)
        results_dict['Perceived Utility'].append(f"{utility_model_win_rate:.2f} ({u_low:.2f}, {u_high:.2f})")
        results_dict['Fluency'].append(f"{fluency_model_win_rate:.2f} ({f_low:.2f}, {f_high:.2f})")
    
    df = pd.DataFrame(results_dict)

    parent = "Model Quoted Generation Win Rate"

    df.columns = pd.MultiIndex.from_tuples([
        ("", "Model"),
        (parent, "Fluency"),
        (parent, "Perceived Utility"),
    ])

    print(df.to_latex(index=False))