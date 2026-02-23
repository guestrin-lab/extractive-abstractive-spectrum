from instructions import *
from few_shot_examples import *
from eval import evaluate_quote_coverage
from utils import format_remove_quotation_marks, format_remove_highlights, generate_from_model, format_source_list, get_quotes
from one_step_instructions import *
import global_vars

# Operating Point: Inline Quotes
def best_of_k_quoted_answer(question, sources_with_sentence_numbers, backbone_model, k, dataset):
    best_quoted_response = ""
    fewest_unquoted_words = float('inf')
    best_response_num_words = float('inf')
    best_num_quoted_words = -float('inf')
    best_details_dict = {}
    
    for _ in range(k):
        quoted_response, details_dict = generate_quoted_answer(question, sources_with_sentence_numbers, backbone_model, dataset)
        print(f"Quoted response (attempt {_+1}/{k}):", quoted_response)
        num_quoted_words, num_words, _ = evaluate_quote_coverage(quoted_response, sources_with_sentence_numbers)
        unquoted_words = num_words - num_quoted_words
        print('Unquoted words:', unquoted_words)
        if ('Insufficient information to generate a grounded response' in quoted_response):
            unquoted_words = float('inf') # this prioritizes a response with >7 unquoted words over an abstention
        if (unquoted_words <= fewest_unquoted_words):
            if ((unquoted_words != fewest_unquoted_words) or (num_quoted_words > best_num_quoted_words)): # (num_words < best_response_num_words)): # this prioritizes shorter responses
                fewest_unquoted_words = unquoted_words
                best_quoted_response = quoted_response
                best_details_dict = details_dict
                best_response_num_words = num_words
                best_num_quoted_words = num_quoted_words
    return best_quoted_response, best_details_dict

def get_sub_questions(question, backbone_model):
    question_str = "Question: "+question
    prompt = construct_prompt(subquestion_instruction_str, subquestions_few_shot_examples_dict, [question_str], "Sub-questions:\n")
    response, details_dict = generate_from_model(backbone_model, prompt)
    return response  

def generate_quoted_answer(question, sources_with_sentence_numbers, backbone_model, data_str):

    op_prompt = one_step_op_instruction_dict['Quoted']

    fewshot_examples = get_quoted_few_shot_examples(data_str)  

    prompt = f"{tg_one_step_outer_prompt_str}\n\n{fewshot_examples}\n\nInstructions: {op_prompt}\nQuery: {question}\n\nSources: {sources_with_sentence_numbers}"

    response, details_dict = generate_from_model(backbone_model, prompt)

    return response, details_dict

# Operating Point: Paraphrase
def generate_paraphrased_answer(query, quoted_output, backbone_model, dataset):

    quoted_output = format_remove_quotation_marks(quoted_output)
    quoted_output = format_remove_highlights(quoted_output)

    op_prompt = one_step_op_instruction_dict['Paraphrased']

    fewshot_examples = get_paraphrased_few_shot_examples(dataset)  

    prompt = f"{tg_one_step_outer_prompt_str}\n\n{fewshot_examples}\n\nInstructions: {op_prompt}\nQuery: {query}\nResponse: {quoted_output}\nParaphrased Response: "

    response, details_dict = generate_from_model(backbone_model, prompt)

    return response, details_dict

# Operating Point: Entailment
def generate_entailed_answer(query, quoted_output, backbone_model, dataset):

    quoted_output = format_remove_quotation_marks(quoted_output)
    quoted_output = format_remove_highlights(quoted_output)

    op_prompt = one_step_op_instruction_dict['Entailed']

    fewshot_examples = get_entailed_few_shot_examples(dataset)  

    prompt = f"{tg_one_step_outer_prompt_str}\n\n{fewshot_examples}\n\nInstructions: {op_prompt}\nQuery: {query}\nResponse: {quoted_output}\nEntailed Response: "

    response, details_dict = generate_from_model(backbone_model, prompt)

    return response, details_dict

# Operating Point: Entailment (inconsistent format)
def generate_citeable_abstractive_answer(query, quoted_output, backbone_model, dataset):

    quoted_output = format_remove_quotation_marks(quoted_output)
    quoted_output = format_remove_highlights(quoted_output)

    op_prompt = one_step_op_instruction_dict['Abstractive']

    fewshot_examples = get_abstractive_few_shot_examples(dataset)  

    prompt = f"{tg_one_step_outer_prompt_str}\n\n{fewshot_examples}\n\nInstructions: {op_prompt}\nQuery: {query}\nResponse: {quoted_output}\nAbstractive Response: "

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