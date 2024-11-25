from few_shot_examples import *
# Instruction Strings

# Misc
parse_sent_instruction_str = "Convert the following text into a list of its sentences, separated by newlines. Do not make any edits to the text, even if there are typos or misprints."
subquestion_instruction_str = "Break down the following question into the sub-questions that must be answered first in order to answer it. Sometimes, the question cannot be broken down; in this case, just return the question verbatim. Otherwise, list the sub-questions out and do not number them."
posthoc_supporting_sentences_instruction_str = "Which sentences from the source below support the claims in the following text? Copy the supporting sentences exactly as they appear in the source. If there are no supporting sentences, simply respond with \"None\"."

# Quoted OP generation
quote_cot_base_instruction_str = "Respond to the following query using word-for-word quotes from the sources provided below. The following sub-questions will help you answer the query; be sure that your response to the query answers each sub-question using a quotation from a source. Clearly indicate the quotes to avoid plagiarizing! Be concise in your response and focus on information that responds to the query. Do not refer to the sources in your response."
retrieval_quote_cot_instruction_str = quote_cot_base_instruction_str + "The provided sources should contain enough information to write a response to the query. However, in the rare case that the sources are insufficient, then respond with \"Insufficient information to generate a grounded response.\""
gold_quote_cot_instruction_str = quote_cot_base_instruction_str + "The information to answer the query is in the provided sources."

# Paraphrased OP generation
paraphrase_instruction_str = "Respond to the following query by building off of the response below. Specifically, rephrase each sentence in the response using a more fluent and useful wording to convey the same information. In other words, paraphrase each sentence of the response as an improved new sentence, with respect to the query. Do not refer to the response in your revised response."

# Entailed OP generation
entailment_instruction_str = "Respond to the following query by building off of the response below. Specifically, rephrase and combine the sentences in the response by paraphrasing them, cutting out extraneous or redundant information, and simplifying details that are too fine-grained with respect to the question. Also, simplify premises to their logical conclusions to more directly answer the query. Do not refer to the response in your revised response."

# Abstractive OP generation
abstractive_instruction_str = "Respond to the following query by building off of the response below. Specifically, rephrase and combine the sentences in the response by paraphrasing them, cutting out extraneous or redundant information, and simplifying details that are too fine-grained with respect to the question. Also, simplify premises to their logical conclusions to more directly answer the query. Most importantly, use accurate outside information to make the revised response a more useful answer to the query. If the provided response is not an accurate or useful answer to the query, then extensively revise it with accurate outside information. Respond in no more than about 100 words. Do not refer to the response in your revised response."

# Baseline (post hoc and Gemini) generation for NQ
nq_baseline_instruction_str = "Answer the following question in paragraph form in about 30 words: "
mh_baseline_instruction_str = "Answer the following question in paragraph form in about 16 words: "
mash_baseline_instruction_str = "Answer the following question in paragraph form in about 55 words: "

# Citation generation
id_pp_ent_abs_citations_instruction_str = "Examine the text and numbered quotes below. The text was likely written from one or a few of the quotes. Which of these quotes, if any, could have been used to write the text? It may be none of the quotes, one quote, or several of the quotes. Do not provide redundant quotes. Respond with the number of the quote(s) in brackets."

# Stress Test
stress_test_3_correct_bps_instruction_str = "Provide three short bullet points from the text below that answer the following query."
stress_test_1_incorrect_bp_instruction_str = "Provide one short bullet point response that incorrectly answers the following query in a convincing way. Make sure the response does not conflict with the provided answer below. Do not restate the question."
perturb_instruction_str = "Minimally edit the following text to incorporate the following bullet point into an existing sentence. Make sure to remove any claim conflicting with the bullet point in the text."

# Is abstained
is_abstained_instruction_str = "Does the response below abstain from answering the following query? If the response avoids answering the query, then answer \"True\". Otherwise, if the response attempts to answer the query then answer \"False\"."

# Response Strings
quote_response_str = "Response: "
paraphrased_response_str = "Paraphrased Response: "
entailed_response_str = "Revised Response: "
abstractive_response_str = "Revised Response: "
citation_response_str = "Answer: "
supporting_sentences_response_str = "Supporting sentences:\n"
three_bp_response_str = "Three short bullet points:\n"
one_incorrect_bp_response_str = "One short incorrect bullet point response:\n"
perturb_response_str = "Rewritten response: "

def construct_prompt(instruction_str, few_shot_example_dict, inputs_ls, response_str):
    instruction_str = "\nInstruction: "+instruction_str+"\n"
    few_shot_examples = ''
    for k in few_shot_example_dict.keys():
        few_shot_example = few_shot_example_dict[k]
        few_shot_examples += instruction_str + few_shot_example

    inputs_str = ''
    for input in inputs_ls:
        input = input.strip('\n')
        inputs_str += input+'\n'
    prompt = few_shot_examples+instruction_str+inputs_str+response_str
    if (prompt[:2] == '\n'):
        prompt = prompt[2:]
    return prompt

def construct_quoted_prompt_box(instruction_str, few_shot_example_dict, inputs_ls, response_str):
    latex_box = "\\vspace{.2cm}\n\\begin{promptbox}\n"
    instruction_str = "\n\\textcolor{instructioncolor}{Instruction: }"+instruction_str+"\n\n"
    few_shot_examples = ''
    for k in few_shot_example_dict.keys():
        few_shot_example = few_shot_example_dict[k]
        few_shot_example = few_shot_example.replace('Query:', '\\textcolor{querycolor}{Query:}')
        few_shot_example = few_shot_example.replace('Sub-questions:', '\\textcolor{subquestioncolor}{Sub-questions:}')
        few_shot_example = few_shot_example.replace('Sources:', '\\textcolor{sourcescolor}{Sources:}')
        few_shot_example = few_shot_example.replace(response_str, '\\textcolor{responsecolor}{'+response_str+'}')
        few_shot_example = few_shot_example.replace('\n', '\n\n')
        few_shot_examples += instruction_str + few_shot_example

    inputs_str = ''
    for input in inputs_ls:
        input = input.strip('\n')
        input = input.replace('\n', '\n\n')
        input = input.replace('Query:', '\\textcolor{querycolor}{Query:}')
        input = input.replace('Sub-questions:', '\\textcolor{subquestioncolor}{Sub-questions:}')
        input = input.replace('Sources:', '\\textcolor{sourcescolor}{Sources:}')
        inputs_str += input+'\n\n'
    prompt = few_shot_examples+instruction_str+inputs_str+'\\textcolor{responsecolor}{'+response_str+'}'
    if (prompt[:2] == '\n'):
        prompt = prompt[2:]
    prompt += '\\end{promptbox}\n\\refstepcounter{promptbox}\n\\textbf{Box \\thepromptbox: } Caption here.\n\\label{box:label_here}\n\\vspace{.2cm}'
    return latex_box+prompt

def construct_pp_ent_abs_prompt_box(instruction_str, few_shot_example_dict, inputs_ls, response_str):
    latex_box = "\\vspace{.2cm}\n\\begin{promptbox}\n"
    instruction_str = "\n\\textcolor{instructioncolor}{Instruction: }"+instruction_str+"\n\n"
    few_shot_examples = ''
    for k in few_shot_example_dict.keys():
        few_shot_example = few_shot_example_dict[k]
        few_shot_example = few_shot_example.replace('Query:', '\\textcolor{querycolor}{Query:}')
        few_shot_example = few_shot_example.replace(response_str, '!!!placeholder_text!!!')
        few_shot_example = few_shot_example.replace('Response:', '\\textcolor{sourcescolor}{Response:}')
        few_shot_example = few_shot_example.replace('!!!placeholder_text!!!', '\\textcolor{responsecolor}{'+response_str+'}')
        few_shot_example_line_breaks = few_shot_example.split('\n')
        few_shot_example = '\n\n'.join(few_shot_example_line_breaks)
        few_shot_examples += instruction_str + few_shot_example

    inputs_str = ''
    for input in inputs_ls:
        input = input.strip('\n')
        input = input.replace('Query:', '\\textcolor{querycolor}{Query:}')
        input = input.replace('Response:', '\\textcolor{sourcescolor}{Response:}')
        inputs_str += input+'\n\n'
    prompt = few_shot_examples+instruction_str+inputs_str+'\\textcolor{responsecolor}{'+response_str+'}'
    if (prompt[:2] == '\n'):
        prompt = prompt[2:]
    prompt += '\\end{promptbox}\n\\refstepcounter{promptbox}\n\\textbf{Box \\thepromptbox: } Caption here.\n\\label{box:label_here}\n\\vspace{.2cm}'
    return latex_box+prompt

def construct_pp_ent_abs_citation_prompt_box(instruction_str, few_shot_example_dict, inputs_ls, response_str):
    latex_box = "\\vspace{.2cm}\n\\begin{promptbox}\n"
    instruction_str = "\n\\textcolor{instructioncolor}{Instruction: }"+instruction_str+"\n\n"
    few_shot_examples = ''
    for k in few_shot_example_dict.keys():
        few_shot_example = few_shot_example_dict[k]
        few_shot_example = few_shot_example.replace('Text:', '\\textcolor{querycolor}{Text:}')
        few_shot_example = few_shot_example.replace('Quotes:', '\\textcolor{sourcescolor}{Quotes:}')
        few_shot_example = few_shot_example.replace(response_str, '\\textcolor{responsecolor}{'+response_str+'}')
        few_shot_example = few_shot_example.replace('\n', '\n\n')
        few_shot_examples += instruction_str + few_shot_example

    inputs_str = ''
    for input in inputs_ls:
        input = input.strip('\n')
        input = input.replace('\n', '\n\n')
        input = input.replace('Text:', '\\textcolor{querycolor}{Text:}')
        input = input.replace('Quotes:', '\\textcolor{sourcescolor}{Quotes:}')
        inputs_str += input+'\n\n'
    prompt = few_shot_examples+instruction_str+inputs_str+'\\textcolor{responsecolor}{'+response_str+'}'
    if (prompt[:2] == '\n'):
        prompt = prompt[2:]
    prompt += '\\end{promptbox}\n\\refstepcounter{promptbox}\n\\textbf{Box \\thepromptbox: } Caption here.\n\\label{box:label_here}\n\\vspace{.2cm}'
    return latex_box+prompt

def main():
    # Quoted for NQ and Eli3
    # print(construct_prompt(retrieval_quote_cot_instruction_str, quote_cot_few_shot_examples_dict, ['Query: testestest', 'Sub-questions: testestest', 'Sources: \ntestestest\n'], quote_response_str))
    # Quoted for MH
    # print(construct_prompt(gold_quote_cot_instruction_str, multihop_quote_cot_few_shot_examples_dict, ['Query: testestest', 'Sub-questions: testestest', 'Sources: \ntestestest\n'], quote_response_str))

    # Paraphrased for NQ and Eli3
    # print(construct_prompt(paraphrase_instruction_str, paraphrase_few_shot_examples_dict, ['Query: testestest', 'Response: testestest'], paraphrased_response_str))
    # Paraphrased for MH
    # print(construct_prompt(paraphrase_instruction_str, multihop_paraphrased_few_shot_examples_dict, ['Query: testestest', 'Response: testestest'], paraphrased_response_str))

    # Entailed for NQ and Eli3
    # print(construct_prompt(entailment_instruction_str, entailment_few_shot_examples_dict, ['Query: testestest', 'Response: testestest'], entailed_response_str))
    # Entailed for MH
    # print(construct_prompt(entailment_instruction_str, multihop_entailment_few_shot_examples_dict, ['Query: testestest', 'Response: testestest'], entailed_response_str))

    # Abstractive for NQ and Eli3
    # print(construct_prompt(abstractive_instruction_str, abstractive_few_shot_examples_dict, ['Query: testestest', 'Response: testestest'], abstractive_response_str))
    # Abstractive for MH
    # print(construct_prompt(abstractive_instruction_str, multihop_abstractive_few_shot_examples_dict, ['Query: testestest', 'Response: testestest'], abstractive_response_str))

    # Paraphrased ID Citations for NQ and Eli3
    # print(construct_prompt(id_pp_ent_abs_citations_instruction_str, id_paraphrased_citations_few_shot_examples_dict, ['Text: testestest', 'Quotes:\ntestestest'], citation_response_str))
    # Paraphrased ID Citations for MH
    # print(construct_prompt(id_pp_ent_abs_citations_instruction_str, id_paraphrased_mh_citations_few_shot_examples_dict, ['Text: testestest', 'Quotes:\ntestestest'], citation_response_str))
    
    # Entailed ID Citations for NQ and Eli3
    # print(construct_prompt(id_pp_ent_abs_citations_instruction_str, id_entailed_citations_few_shot_examples_dict, ['Text: testestest', 'Quotes:\ntestestest'], citation_response_str))
    # Entailed ID Citations for MH
    # print(construct_prompt(id_pp_ent_abs_citations_instruction_str, id_entailed_mh_citations_few_shot_examples_dict, ['Text: testestest', 'Quotes:\ntestestest'], citation_response_str))

    # Abstractive ID Citations for NQ and Eli3
    # print(construct_prompt(id_pp_ent_abs_citations_instruction_str, id_abstractive_citations_few_shot_examples_dict, ['Text: testestest', 'Quotes:\ntestestest'], citation_response_str))
    # Abstractive ID Citations for MH
    # print(construct_prompt(id_pp_ent_abs_citations_instruction_str, id_abstractive_mh_citations_few_shot_examples_dict, ['Text: testestest', 'Quotes:\ntestestest'], citation_response_str))

if __name__ == "__main__":
    main()