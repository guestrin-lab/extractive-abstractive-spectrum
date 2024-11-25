from instructions import *
from few_shot_examples import *
from eval import evaluate_quote_coverage
from utils import format_remove_quotation_marks, generate_from_model, format_source_list, get_quotes
import global_vars

# Operating Point: Inline Quotes
def best_of_k_quoted_answer(question, sources, backbone_model, k, dataset=None, using_gold=False):
    best_quoted_response = ""
    fewest_unquoted_words = float('inf')
    best_response_num_words = float('inf')
    best_num_quoted_words = -float('inf')
    best_details_dict = {}
    sub_questions = get_sub_questions(question, backbone_model)
    print('sub_questions:', sub_questions)
    for _ in range(k):
        quoted_response, details_dict = generate_quoted_answer(question, sub_questions, sources, backbone_model, dataset, using_gold)
        num_quoted_words, num_words, _ = evaluate_quote_coverage(quoted_response, sources)
        unquoted_words = num_words - num_quoted_words
        print('Unquoted words:', unquoted_words)
        if ('Insufficient information to generate a grounded response' in quoted_response):
            unquoted_words = float('inf') # this prioritizes a response with >7 unquoted words over an abstention
        if (unquoted_words <= fewest_unquoted_words):
            if ((unquoted_words != fewest_unquoted_words) or (num_quoted_words > best_num_quoted_words)): # this prioritizes quoting
                fewest_unquoted_words = unquoted_words
                best_quoted_response = quoted_response
                best_details_dict = details_dict
                best_response_num_words = num_words
                best_num_quoted_words = num_quoted_words
    return best_quoted_response, best_details_dict

def get_sub_questions(question, backbone_model):
    question_str = "Question: "+question
    prompt = construct_prompt(subquestion_instruction_str, subquestions_few_shot_examples_dict, question_str, "Sub-questions:\n")
    response, details_dict = generate_from_model(backbone_model, prompt)
    return response    

def generate_quoted_answer(question, subquestions, sources, backbone_model, dataset, using_gold=False):
    sources_str = "Sources:\n"+format_source_list(sources)
    question_str = "Query: "+question
    subquestions_str = 'Sub-questions: '+subquestions
    if (using_gold):
        instruction_str = gold_quote_cot_instruction_str
    else: # using retrieved sources
        instruction_str = retrieval_quote_cot_instruction_str
    
    if (dataset == 'multihop'):
        prompt = construct_prompt(instruction_str, multihop_quote_cot_few_shot_examples_dict, [question_str, subquestions_str, sources_str], quote_response_str)
        prompt_box = construct_quoted_prompt_box(instruction_str, multihop_quote_cot_few_shot_examples_dict, [question_str, subquestions_str, sources_str], quote_response_str)
    elif (dataset == 'nq'):
        prompt = construct_prompt(instruction_str, quote_cot_few_shot_examples_dict, [question_str, subquestions_str, sources_str], quote_response_str) 
        prompt_box = construct_quoted_prompt_box(instruction_str, quote_cot_few_shot_examples_dict, [question_str, subquestions_str, sources_str], quote_response_str) 
    elif (dataset == 'eli5_nq'):
        prompt = construct_prompt(instruction_str, eli3g_quote_cot_few_shot_examples_dict, [question_str, subquestions_str, sources_str], quote_response_str) 
        prompt_box = construct_quoted_prompt_box(instruction_str, eli3g_quote_cot_few_shot_examples_dict, [question_str, subquestions_str, sources_str], quote_response_str) 
    elif (dataset == 'mash'):
        prompt = construct_prompt(instruction_str, mash_quote_cot_few_shot_examples_dict, [question_str, subquestions_str, sources_str], quote_response_str) 
        prompt_box = construct_quoted_prompt_box(instruction_str, mash_quote_cot_few_shot_examples_dict, [question_str, subquestions_str, sources_str], quote_response_str) 
    else:
        print('Quoted OP not implemented for this dataset.')
        exit()

    print('... Generating Quoted Answer ...')
    print('--------------------------------------------------------------------------------------------------------------------------------------------')
    print(prompt_box)
    print('--------------------------------------------------------------------------------------------------------------------------------------------')
    breakpoint()
    
    response, details_dict = generate_from_model(backbone_model, prompt)
    return response, details_dict

# Operating Point: Paraphrase
def generate_paraphrased_answer(query, quoted_output, backbone_model, dataset=None):
    query_str = "Query: "+query
    response_str = "Response: "+format_remove_quotation_marks(quoted_output)
    if (dataset == 'multihop'):
        prompt = construct_prompt(paraphrase_instruction_str, multihop_paraphrased_few_shot_examples_dict, [query_str, response_str], paraphrased_response_str)
        prompt_box = construct_pp_ent_abs_prompt_box(paraphrase_instruction_str, multihop_paraphrased_few_shot_examples_dict, [query_str, response_str], paraphrased_response_str)
    elif ((dataset == 'nq') | (dataset == 'eli5_nq')):
        prompt = construct_prompt(paraphrase_instruction_str, paraphrase_few_shot_examples_dict, [query_str, response_str], paraphrased_response_str)
        prompt_box = construct_pp_ent_abs_prompt_box(paraphrase_instruction_str, paraphrase_few_shot_examples_dict, [query_str, response_str], paraphrased_response_str)
    elif (dataset == 'mash'):
        prompt = construct_prompt(paraphrase_instruction_str, mash_paraphrase_few_shot_examples_dict, [query_str, response_str], paraphrased_response_str)
        prompt_box = construct_pp_ent_abs_prompt_box(paraphrase_instruction_str, mash_paraphrase_few_shot_examples_dict, [query_str, response_str], paraphrased_response_str)
    else:
        print('PP OP not implemented for this dataset.')
        exit()

    print('... Generating Paraphrased Answer ...')
    print('--------------------------------------------------------------------------------------------------------------------------------------------')
    print(prompt_box)
    print('--------------------------------------------------------------------------------------------------------------------------------------------')
    breakpoint()

    response, details_dict = generate_from_model(backbone_model, prompt)
    return response, details_dict

# Operating Point: Entailment
def generate_entailed_answer(query, quoted_output, backbone_model, dataset=None):
    unmarked_quoted_output = format_remove_quotation_marks(quoted_output) # of using quoted output
    query_str = "Query: "+query
    response_str = "\nResponse: "+unmarked_quoted_output
    if (dataset == 'multihop'):
        prompt = construct_prompt(entailment_instruction_str, multihop_entailment_few_shot_examples_dict, [query_str, response_str], entailed_response_str)
        prompt_box = construct_pp_ent_abs_prompt_box(entailment_instruction_str, multihop_entailment_few_shot_examples_dict, [query_str, response_str], entailed_response_str)
    elif ((dataset == 'nq') | (dataset == 'eli5_nq')):
        prompt = construct_prompt(entailment_instruction_str, entailment_few_shot_examples_dict, [query_str, response_str], entailed_response_str)
        prompt_box = construct_pp_ent_abs_prompt_box(entailment_instruction_str, entailment_few_shot_examples_dict, [query_str, response_str], entailed_response_str)
    elif (dataset == 'mash'):
        prompt = construct_prompt(entailment_instruction_str, mash_entailment_few_shot_examples_dict, [query_str, response_str], entailed_response_str)
        prompt_box = construct_pp_ent_abs_prompt_box(entailment_instruction_str, mash_entailment_few_shot_examples_dict, [query_str, response_str], entailed_response_str)
    else:
        print('Ent OP not implemented for this dataset.')
        exit()

    print('... Generating Entailed Answer ...')
    print('--------------------------------------------------------------------------------------------------------------------------------------------')
    print(prompt_box)
    print('--------------------------------------------------------------------------------------------------------------------------------------------')
    breakpoint()

    response, details_dict = generate_from_model(backbone_model, prompt)
    return response, details_dict

# Operating Point: Entailment (inconsistent format)
def generate_citeable_abstractive_answer(question, quoted_output, backbone_model, dataset=None):
    question_str = "Query: "+question
    response_str = "Response: "+format_remove_quotation_marks(quoted_output)
    if (dataset == 'multihop'):
        prompt = construct_prompt(abstractive_instruction_str, multihop_abstractive_few_shot_examples_dict, [question_str, response_str], abstractive_response_str)
        prompt_box = construct_pp_ent_abs_prompt_box(abstractive_instruction_str, multihop_abstractive_few_shot_examples_dict, [question_str, response_str], abstractive_response_str)
    elif ((dataset == 'nq') | (dataset == 'eli5_nq')):
        prompt = construct_prompt(abstractive_instruction_str, abstractive_few_shot_examples_dict, [question_str, response_str], abstractive_response_str)
        prompt_box = construct_pp_ent_abs_prompt_box(abstractive_instruction_str, abstractive_few_shot_examples_dict, [question_str, response_str], abstractive_response_str)
    elif (dataset == 'mash'):
        prompt = construct_prompt(abstractive_instruction_str, mash_abstractive_few_shot_examples_dict, [question_str, response_str], abstractive_response_str)
        prompt_box = construct_pp_ent_abs_prompt_box(abstractive_instruction_str, mash_abstractive_few_shot_examples_dict, [question_str, response_str], abstractive_response_str)
    else: 
        print('Abs OP not implemented for this dataset.')
        exit()

    print('... Generating Abstractive Answer ...')
    print('--------------------------------------------------------------------------------------------------------------------------------------------')
    print(prompt_box)
    print('--------------------------------------------------------------------------------------------------------------------------------------------')
    breakpoint()

    response, details_dict = generate_from_model(backbone_model, prompt)
    return response, details_dict

# Operating Point: Entailment (inconsistent format)
def generate_vanilla_answer(question, sources, backbone_model):
    instruction_str = 'Use the sources below to help respond to the query.'
    sources_str = "\nSources:\n"+format_source_list(sources)
    question_str = "\nQuery: "+question
    prompt = instruction_str + question_str + sources_str + 'Response: '
    response, details_dict = generate_from_model(backbone_model, prompt)
    return response, details_dict

# Operating Point: Fully abstractive
def generate_backbone_model_answer(question, backbone_model):
    response, details_dict = generate_from_model(backbone_model, question)
    return response, details_dict


def main():
    backbone_model = None # OpenAI()
    generate_paraphrased_answer('test query', '\"test\" response test \"response\"', backbone_model)
    return

if __name__ == "__main__":
    main()