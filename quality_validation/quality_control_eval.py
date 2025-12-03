import os
import re 
import ast
import numpy as np
import pandas as pd 
from tqdm import tqdm

import asyncio
from openai import AsyncOpenAI
import time
import argparse

from eval import AutoEvaluator

async def auto_eval():
    overall_start_time = time.time()
    evaluator = AutoEvaluator()

    # Open up results file and read in csv results
    results_df1 = pd.read_csv(f"human_quoted_generations/rejected_and_accepted_quoted_responses_NQ.csv")
    results_df2 = pd.read_csv(f"human_quoted_generations/rejected_and_accepted_quoted_responses_ETA3G.csv")
    results_df3 = pd.read_csv(f"human_quoted_generations/rejected_and_accepted_quoted_responses_MH.csv")
    results_df4 = pd.read_csv(f"human_quoted_generations/rejected_and_accepted_quoted_responses_MASH.csv")
    
    results_df = pd.concat([results_df1, results_df2, results_df3, results_df4])
    # Remove NaN responses
    results_df = results_df[results_df['Answer'].notna()]

    # Make sure to remove double-quotes 
    results_df['Answer'] = results_df['Answer'].apply(lambda x: x.replace('\"', ''))
    
    results_df = results_df.iloc[0:10] # TODO remove
    
    # Evaluate all quotes responses
    queries_to_send = results_df['Question'].tolist()
    responses_to_send = results_df['Answer'].tolist()
        
    semaphore = asyncio.Semaphore(5)  
    async def sem_task(query, response):
        async with semaphore:
            fluency_scores = await evaluator.evaluate_fluency(query, response)
            perceived_utility_scores = await evaluator.evaluate_perceived_utility(query, response)
            return fluency_scores, perceived_utility_scores, query, response
    results = await asyncio.gather(*[sem_task(query, response) for (query, response) in zip(queries_to_send, responses_to_send)], return_exceptions=True)

    # Make a dataframe to store the auto-eval results and merge into the original results_df
    fluency_score_ls = []
    perceived_utility_score_ls = []
    query_ls = []
    response_ls = []

    for result in results:
        if isinstance(result, Exception):
            print("Error during evaluation:", repr(result))
            fluency_score_ls.append(None)
            perceived_utility_score_ls.append(None)
            continue

        fluency_score, perceived_utility_score, query, response = result
        fluency_score_ls.append(fluency_score)
        perceived_utility_score_ls.append(perceived_utility_score)
        query_ls.append(query)
        response_ls.append(response)

    auto_eval_dict = {
        'Question': query_ls,
        'Answer': response_ls,
        'auto_fluency_rating': fluency_score_ls,
        'auto_utility_rating': perceived_utility_score_ls,
    }

    auto_eval_df = pd.DataFrame(auto_eval_dict)

    merged_results_df = results_df.merge(auto_eval_df, on=['Question'])

    print("Total time taken for evaluation: %s seconds" % (time.time() - overall_start_time))
    save_path = "fluency_utility_for_human_quoted_responses.csv" 
    print("Results saved to: ", save_path)
    merged_results_df.to_csv(save_path, index=False) 
    
    print()
    print("Fluency")
    print(f"{np.mean(merged_results_df['auto_fluency_rating'])}")

    print()

    print("Perceived Utility")
    print(f"{np.mean(merged_results_df['auto_utility_rating'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    evaluator = AutoEvaluator()
    print()
    asyncio.run(auto_eval())

    print()
    print()

# python eval.py --results_dir autoEval_results --results_file gpt5_nq_byQueryOP
    
