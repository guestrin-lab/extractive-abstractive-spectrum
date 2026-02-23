from few_shot_examples import *
from eval_instructions import *
# Instruction Strings

# Misc
# parse_sent_instruction_str = "Convert the following text into a list of its sentences, separated by newlines. Do not make any edits to the text, even if there are typos or misprints. Do not infer missing punctuaton marks."
parse_sent_instruction_str = "Convert the following text into a list of its sentences, separated by newlines. Do not make any edits to the text, even if there are typos or misprints."
subquestion_instruction_str = "Break down the following question into the sub-questions that must be answered first in order to answer it. Sometimes, the question cannot be broken down; in this case, just return the question verbatim. Otherwise, list the sub-questions out and do not number them."
posthoc_supporting_sentences_instruction_str = "Which sentences from the source below support the claims in the following text? Copy the supporting sentences exactly as they appear in the source. If there are no supporting sentences, simply respond with \"None\"."

# Optimized prompt for one-step operating points
tg_one_step_outer_prompt_str = """You are an attentive assistant. Follow all instructions carefully. Your responses must rigorously maximize the following metrics and comply precisely with all operational requirements below:

**Fluency:** Write in clear, concise, idiomatic English with logical sequencing and varied sentence structures. Explicitly state all entities and facts—avoid ambiguous, vague, or redundant language, as well as under- or over-repetition of names or terms within close proximity. Carefully restructure sentences to ensure each statement remains clear and coherent even if read in isolation. When groupings or lists are present, organize information logically and employ transitions as needed for readability and natural flow.

**Perceived Utility:** Directly and efficiently answer the user's query. Organize key takeaways, facts, or advice in skimmable, logically ordered, and thematically grouped sections, adjusting explanations for the likely audience. Every factual referent (names, places, entities, relationships) must be explicit. Provide concise rationales (why/how) for advice, practical steps, or procedural instructions, and briefly explain the importance of non-obvious terms or details. When discussing little-known entities or comparative facts, append a succinct, relevant description or context for user orientation. Where a minor detail (such as an alternate title, spelling, or credit) is included, clarify its relevance to the user’s query.

**Citation Precision:** Insert citation markers, direct quotes, or paraphrased statements only when the claim is directly, word-for-word, or unambiguously supported by the provided source(s). Place each citation marker immediately after the specific claim, fact, or data point it supports—*never* at the end of a sentence or grouping unless all elements are fully, equally, and explicitly covered by the source(s). Do not assign a single citation to multiple facts unless every fact is supported by the source; avoid clustered or grouped citations for composite claims. Never fabricate citations, use placeholders, or infer evidence. Use only genuine, checkable sources from the input/context.

**Citation Coverage and Claim Splitting:** Every distinct factual claim, component, or actionable statement requiring support must be individually substantiated by a citation or direct quote. For any sentence containing multiple distinct factual elements (names, dates, mechanisms, outcomes), split or rephrase so each part is mapped to its own citation, or clearly indicate if part is based on general/background knowledge (“Based on general knowledge” / “No source provided”). Never cite a source for information it does not explicitly contain or confirm—e.g., do not cite a provincial source as proof of a national location unless the national context is present in the source.

**Citation-Claim Mapping and Internal Audit:** As you draft, audit each atomic factual segment to guarantee that its citation marker appears *immediately after* the claim it validates and that the marker covers only that specific content. If two or more claims share a citation, this must only occur if the source explicitly and entirely supports all claims at once—otherwise, restructure and recite. Strictly prohibit citation “bundling” (multiple citations after multi-claim sentences) unless each citation independently covers the full statement. For every output, review that citation markers and source support precisely match at the claim-by-claim level; edit or split sentences as necessary for maximum traceability.

**Handling Logical Conclusions, Inference, and Background Knowledge:** When drawing conclusions that require inference (e.g., deducing that two villages in different provinces are in different countries when only the provinces are named in the sources), transparently indicate which steps are evidenced by sources and which are based on general or background knowledge. Flag any logical or connecting steps without a direct source as “Based on general knowledge” or “No source provided.”

**Handling Redundancy, Synthesis, and Multiple Claims:** Actively eliminate redundancy. If different sources contain parallel information, synthesize into a single, well-attributed unit unless each genuinely adds unique value (then clarify the distinction). Never quote or paraphrase similar facts multiple times without added usefulness. Clarify or note distinctions for multiple citations, and group highly related facts for user clarity (provided citation mapping remains granular and accurate).

**Quotation, Paraphrasing, and Editing:** Use quotation marks exclusively for word-for-word source text and mark any modifications (abridgments, elisions) clearly. Paraphrase for clarity or conciseness only if the new language remains fully faithful to the source and can be directly traced; do not paraphrase in a way that changes meaning or introduces ambiguity. For all paraphrased statements, double-check semantic equivalence (e.g., “was” ≠ “worked as”). Avoid non-standard or awkward phrasing—especially with technical terms, dates, currencies, or attributions.
"""

# Hand-written Outer Prompt for response generation
hand_written_outer_prompt_generation_str = f"""You are an attentive assistant who follows all instructions carefully. Your responses must maximize the following metrics.

**Fluency:** Write in clear, concise, idiomatic English with logical sequencing and varied sentence structures. Explicitly state all entities and facts—avoid ambiguous, vague, or redundant language, as well as under- or over-repetition of names or terms within close proximity. Carefully restructure sentences to ensure each statement remains clear and coherent. When groupings or lists are present, organize information logically and employ transitions as needed for readability and natural flow. Use proper punctuation, grammar, and capitalization. You will be graded on fluency by the following rubric:
{fluency_answer_choices}

**Perceived Utility:** Directly and concisely answer the user's query in the appropriate style. Be sure to directly address the query and exclude any information that is redundant or not relevant to the query. Carefully choose vocabulary, structure sentences, and present information in an appropriate way for the query. You will be graded on perceived utility by the following rubric:
{perceived_utility_answer_choices}

**Citation Coverage and Precision** Every distinct factual claim, component, or actionable statement requiring support must be individually substantiated by a direct quote from the provided sources or response. When paraphrasing information, be sure that the new language remains fully faithful to the original language and can be directly traced; do not paraphrase in a way that changes meaning or introduces ambiguity. For all paraphrased statements, double-check semantic equivalence (e.g., “performed” ≠ “sang”). You will be graded on coverage for each sentence by the following rubric:
{citation_coverage_answer_choices}

You will be graded on precision for each quote used to support a claim in the sentence by the following rubric:
{citation_precision_answer_choices}

**Instruction Prioritization:** The instructions in the following section take precedence over maximizing the metrics above. Of course, still adhere to the metrics above as much as possible considering the instructions below. In particular:
- If the instructions below ask you to construct a response using direct quotes from the sources, then you must avoid using any non-quoted words.
- If the instructions below ask you to paraphrase information, then you must do so.
- If the instructions below ask you to otherwise edit the response, then you must do so.
- If the instructions below ask you to add additional knowledge, then you must do so.

"""
# 

# Hand-written Outer Prompt for citation identification
hand_written_outer_prompt_citation_id_str = f"""You are an attentive assistant who follows all instructions carefully. Your responses must maximize the following metrics.

**Precision:** Every quote must support at least one claim in the text (sentence). You will be graded on precision for each quote you choose by the following rubric:
{citation_precision_answer_choices}

**Coverage:** Every distinct factual claim, component, or actionable statement in the text requiring support must be substantiated by a direct quote from the provided sources or response. You will be graded on coverage for the text (sentence) by the following rubric:
{citation_coverage_answer_choices}

"""

# Textgrad prompts
tg_rubric_prompt = 'You are an attentive assistant who strictly follows all user instructions and maximizes fluency, perceived utility, citation coverage, and citation precision. Use only information found in the provided sources (or prior responses, if explicitly allowed); never hallucinate or invent information, and do not refer to prior responses or input meta-information unless instructed.\n\n**General Principles:**\n- Build every response *exclusively* from claims directly traceable to the provided sources. Do not use outside knowledge unless explicitly allowed.\n- **For every output, regardless of length:**  \n  - **NEVER submit a blank, fragmentary, or verbatim output.** Even ultra-short, single-fact, or minimal responses require at least one overt paraphrase, format change, or direct answer structure (e.g., turning “1995” or “‘Song’ (2003)” into “The song was released in 2003.”). Never produce a blank or unchanged output—retry or rephrase until this minimum is achieved.\n  - If a maximally fluent, accessible form is already present, produce a minor format or syntactic change, or supply a one-line rationale explaining its maximality.\n- Each factual statement, list item, or actionable phrase must show a strict, one-to-one mapping to a source claim unless user instructions specify condensing, combining, or logical conclusion. Never merge, split, generalize, or omit claims—including qualifiers or exceptions—unless *explicitly* instructed, and do not supplement with extraneous details (e.g., professions, roles) unless the query requires them for clarity.\n- **Qualifiers and Certainty:** Always preserve the original meaning, scope, qualifiers (“may,” “can,” “in severe cases”), and conditionality. Do not strengthen, drop, or blur them in paraphrase or condensation.\n- **Condensation, Simplification, and Logical Inference:** Only condense or combine information when user instructions explicitly direct you to do so (e.g., “simplify to logical conclusions”). In such cases:\n  - Aggressively minimize redundancy and condense to the shortest, clearest phrasing that covers *all* original claims.\n  - Group or reorganize lists for conciseness and audience needs, but ensure all substantive, distinct claims are preserved; omit only genuinely redundant/repetitive elements.\n  - Always cross-check input and source for overlap to prevent repetition or omission.\n  - When merging or condensing, ensure every condensed claim remains traceable to its source(s), and if necessary, briefly indicate source mapping or logical rationale.\n  - Never introduce unsupported inferences or unstated implications.\n- For ambiguous, contradictory, or incomplete inputs, paraphrase faithfully and note the ambiguity or missing coverage.\n\n**Fluency and Utility:**\n- Always write in clear, idiomatic, and fully-formed English sentences suited to the specified target audience.\n- For factoid, closed, or ultra-minimal queries (“who,” “which country,” “what year”):\n  - Always convert fragments or one/two-word facts into a direct, complete sentence that unmistakably references the answer’s subject/entity (even for children).\n  - Match answer surface form to user query (e.g., for “Which country...?”, directly provide the country name: “United States.” Prefer “from the United States” over “was an American” for clarity unless otherwise requested.)\n  - Only add further context (names, roles, explanations) if essential for disambiguation.\n  - For tables, lists, or titles, always render the answer as a complete sentence.\n- For child or audience adaptation, adjust language register as specified (e.g., use simple words, short sentences, direct style for children).\n\n**Citation Coverage and Precision:**\n- For *all* claims, list items, or paraphrases, source traceability is required. If an answer contains multiple claims (e.g., symptom lists), each should be individually mapped or, where source structure allows, cited collectively.\n- When required by user or task format, attach explicit source attributions (parenthetical, inline, or list markers).\n- For any output, ensure no padding, no unsupported additions, and every detail is justifiable by a source.\n- If information is unavailable, clearly state its absence.\n\n**Redundancy and Compactness:**\n- Aggressively remove or combine only *genuinely* redundant information—never compress or omit informational content present in the source, especially within technical lists (e.g., disease symptoms) unless instructed.\n- For tasks requiring condensation, minimize relisting or overlapping items while preserving traceable detail.\n\n**Edge/Special Cases and Self-Check:**\n- For fragmentary, minimally informative, or list/table answers, enforce conversion to full sentences with entity/context.\n- For technical, medical, or list-dense content: retain original order/grouping unless reorganization clearly improves clarity; always preserve qualifiers and do not collapse distinct claims unless requested.\n- Before submitting:\n  - Confirm every claim (including condensed ones) is traceable and source-supported.\n  - Check that all user instructions, output format, and audience expectations are met.\n  - Ensure brevity, clarity, and maximal utility—no blank, fragment, verbatim, or off-format outputs.\n\n**Examples of Good/Bad Output:**\n- Query: What year did “Seven Nation Army” come out?  \n  - Good: “The song ‘Seven Nation Army’ was released in 2003.”\n  - Bad: “‘Seven Nation Army’ (2003).” (fragment); blank.\n- Query: Which country is the director of “Inga” from?\n  - Good: “The director of ‘Inga,’ Joseph W. Sarno, was from the United States.”\n  - Bad: “He was an American film director and screenwriter.” (contains extraneous detail); “United States.” (if not clearly referencing subject).\n- Query: What are symptoms of shellfish poisoning?\n  - Good: “Symptoms may include nausea, vomiting, diarrhea, abdominal pain, and numbness of the lips, tongue, and fingertips. Other possible effects are muscle paralysis, headache, lower-back pain, vertigo, loss of balance, drooling, blurred vision, temporary blindness, elevated heart rate, low blood pressure, and altered temperature perception. In severe cases, coma and respiratory failure can occur.” (preserves qualifiers, full coverage, and traceability)\n  - Bad: “Symptoms include nausea, diarrhea, and more.” (over-compressed); listed without qualifiers or merged items.\n\nRemember: If in doubt about paraphrasing or condensation, prioritize clarity, complete coverage, source traceability, and instruction compliance—never sacrifice necessary qualifiers or introduce new information. Always ensure every response is a faithful, explicit, and maximally useful re-expression of the sourced answer, tailored precisely to the user’s question and instructions.'

tg_cite_rubric_prompt = 'You are an expert citation assistant. Your task is to carefully analyze each sentence provided and determine which segments require citations based on the presence of direct quotations, paraphrased content, ideas, statistics, or claims that are not common knowledge and originate from another source. For every relevant part of the sentence, clearly identify the text that should be cited and specify the type of source or quote that would be appropriate to support it. Do not cite information that is considered common knowledge. Make your decisions based solely on the content and context of the sentence provided.'

# Quoted OP generation
# Used for GPT-4
quote_cot_base_instruction_str = "Respond to the following query using word-for-word quotes from the sources provided below. The following sub-questions will help you answer the query; be sure that your response to the query answers each sub-question using a quotation from a source. Clearly indicate the quotes to avoid plagiarizing! Be concise in your response and focus on information that responds to the query. Do not refer to the sources in your response."

# Used for GPT-5 and Sonnet 4.5
# quote_cot_base_instruction_str: "Respond to the following query using word-for-word quotes from the sources provided below. Clearly indicate the quotes with double quotes to avoid plagiarizing! Please present the quotes as inline quotes within grammatically correct sentences. Do not add new information with unquoted words. Be concise in your response and focus on information that responds to the query——do not repeat information already quoted. Do not refer to the sources in your response."

retrieval_quote_cot_instruction_str = quote_cot_base_instruction_str + "The provided sources should contain enough information to write a response to the query. However, in the rare case that the sources are insufficient, then respond with \"Insufficient information to generate a grounded response.\""
gold_quote_cot_instruction_str = quote_cot_base_instruction_str + "The information to answer the query is in the provided sources."

# Paraphrased OP generation
paraphrase_instruction_str = "Respond to the following query by building off of the response below. Specifically, rephrase each sentence in the response using a more fluent and useful wording to convey the same information. In other words, paraphrase each sentence of the response as an improved new sentence, with respect to the query. Do not refer to the response in your revised response."

# Entailed OP generation
entailment_instruction_str = "Respond to the following query by building off of the response below. Specifically, rephrase and combine the sentences in the response by paraphrasing them, cutting out extraneous or redundant information, and simplifying details that are too fine-grained with respect to the question. Also, simplify premises to their logical conclusions to more directly answer the query. Do not refer to the response in your revised response."

# Abstractive OP generation
abstractive_instruction_str = "Respond to the following query by building off of the response below. Specifically, rephrase and combine the sentences in the response by paraphrasing them, cutting out extraneous or redundant information, and simplifying details that are too fine-grained with respect to the question. Also, simplify premises to their logical conclusions to more directly answer the query. Most importantly, use accurate outside information to make the revised response a more useful answer to the query. If the provided response is not an accurate or useful answer to the query, then extensively revise it with accurate outside information. Respond in no more than about 100 words. Do not refer to the response in your revised response."
# some sentences may not have citation. already, this is the case! in those cases of sentences w/o citation, it's either the case that:
    # sentence doesn't require citation: should NOT count as a coverage error
    # sentence needs citation: should count as a coverage error

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

def construct_prompt(instruction_str, few_shot_example_dict, inputs_ls=[], response_str='', use_optimized_prompt=False, use_optimized_prompt_for_citation_id=False, just_prompt_and_fewshot=False):
    assert type(inputs_ls)==list
    instruction_str = "\nInstruction: "+instruction_str+"\n"
    few_shot_examples = ''
    for k in few_shot_example_dict.keys():
        few_shot_example = few_shot_example_dict[k]
        few_shot_examples += instruction_str + few_shot_example

    if just_prompt_and_fewshot:
        return few_shot_examples+instruction_str

    inputs_str = ''
    for input in inputs_ls:
        input = input.strip('\n')
        inputs_str += input+'\n'
    prompt = few_shot_examples+instruction_str+inputs_str+response_str
    if (prompt[:2] == '\n'):
        prompt = prompt[2:]

    if use_optimized_prompt_for_citation_id:
        # prompt = hand_written_outer_prompt_citprompt = tg_cite_rubric_prompt + '\n\n' + promptation_id_str + prompt
        prompt = tg_cite_rubric_prompt + '\n\n' + prompt
    elif use_optimized_prompt:
        # prompt = optimized_outer_prompt_str + prompt
        # prompt = hand_written_outer_prompt_generation_str + prompt
        prompt = tg_rubric_prompt + '\n\n' + prompt
    
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

    # print(construct_prompt(stress_test_3_correct_bps_instruction_str, stress_test_3_correct_bps_few_shot_examples_dict, ['Query: testestest', 'Text: testestest'], three_bp_response_str))
    
    
    # entailment_instruction_str
    # abstractive_instruction_str
    # print(quote_cot_base_instruction_str)
    # print(paraphrase_instruction_str)
    # print(entailment_instruction_str)
    # print(abstractive_instruction_str)
    print(id_pp_ent_abs_citations_instruction_str)

if __name__ == "__main__":
    main()