import streamlit as st
import pandas as pd
import sys
import time

COLORS = {'\x1b[92m':':green[', '\x1b[96m':':orange[', '\x1b[95m':':red[', '\x1b[1;31;60m':':blue[', '\x1b[102m':':violet[', '\x1b[1;35;40m':':grey[', '\x1b[0;30;47m':':rainbow[', '\x1b[0;33;47m':':orange[', '\x1b[0;34;47m':':blue[', '\x1b[0;31;47m':':red[', '\x1b[0m':']'}
MD_IDX_TO_MD_COLORS = {0:':orange[', 1:':green[', 2:':blue[', 3:':red[', 4:':rainbow[', 5:':violet[', 6:':orange[', 7:':blue[', 8:':green[', 9:':red['}
NUM_TO_ALPHA = {0: 'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J'}

def get_md_color(ansi_escape_sequence):
    if ('ANSI_TO_MD_IDX' not in st.session_state):
        st.session_state['ANSI_TO_MD_IDX'] = {}
    if (ansi_escape_sequence == '\x1b[0m'):
        return '] '
    if (len(st.session_state["ANSI_TO_MD_IDX"]) == 0):
        st.session_state["ANSI_TO_MD_IDX"][ansi_escape_sequence] = 0
    if (ansi_escape_sequence not in st.session_state["ANSI_TO_MD_IDX"].keys()):
        st.session_state["ANSI_TO_MD_IDX"][ansi_escape_sequence] = max(st.session_state["ANSI_TO_MD_IDX"].values())+1
    md_idx = st.session_state["ANSI_TO_MD_IDX"][ansi_escape_sequence]
    return MD_IDX_TO_MD_COLORS[md_idx]

def clear_ansi(text):
    for ansi_escape_sequence in COLORS.keys():
        if (ansi_escape_sequence in text):
            text = text.replace(ansi_escape_sequence, '')
    return text

def highlight(text):
    for ansi_escape_sequence in COLORS.keys():
        if (ansi_escape_sequence in text):
            md_color = get_md_color(ansi_escape_sequence)
            text = text.replace(ansi_escape_sequence, '')
    return text

def format_remove_quotation_marks(output):
    list_output = list(output)
    i=0
    while i < len(list_output):
        if (list_output[i] == "\""):
            del list_output[i]
            i -= 1
        i += 1
    fixed_output = "".join(list_output)
    return fixed_output

def save_start_time():
    st.session_state["start_time"] = time.time()

def save_time(i, task_str):
    seconds_elapsed = time.time() - st.session_state["start_time"]
    if (task_str == 'prec'):
        if ('prec_t2v' in st.session_state):
            st.session_state['prec_t2v'].append(seconds_elapsed)
        else:
            st.session_state['prec_t2v'] = [seconds_elapsed]
        st.session_state["start_time"] = time.time()
    else:
        if ('cov_t2v' in st.session_state):
            st.session_state['cov_t2v'].append(seconds_elapsed)
        else:
            st.session_state['cov_t2v'] = [seconds_elapsed]
        st.session_state["start_time"] = time.time()
    return

def get_substring_indices(text, substring):
    i = 0
    ocurrences = []
    for i in range(len(text)-len(substring)+1):
        if (substring == text[i:i+len(substring)]):
            ocurrences.append((i,i+len(substring)))
    return ocurrences

def get_highlighted_snippet(snippet_ls):
    for i in range(len(snippet_ls)):
        for ansi_escape_sequence in COLORS.keys():
            if (ansi_escape_sequence == '\x1b[0m'):
                snippet_ls[i] = snippet_ls[i].replace(ansi_escape_sequence, "</span>")
            else:
                snippet_ls[i] = snippet_ls[i].replace(ansi_escape_sequence, "<span class='orange-highlight'>")
    return snippet_ls

def get_cited_sources_for_sentence(cited_sources_ls, citations):
    sources_idxs_to_show = {}
    for citation_num in citations:
        citation = '['+str(citation_num)+']'
        highlighted_citation = "<span class='orange-highlight'>"+citation
        found_citation = False
        for ansi_escape_sequence in COLORS.keys():
            if (found_citation):
                break
            ansi_citation = ansi_escape_sequence+citation

            for j in range(len(cited_sources_ls)):
                source = cited_sources_ls[j]
                if (ansi_citation in source):
                    found_citation = True
                    start_citation_occurrences = get_substring_indices(source, ansi_citation)
                    source_to_use = source[:start_citation_occurrences[0][0]]+highlighted_citation+source[start_citation_occurrences[0][1]:]
                    end_citation_occurrences = get_substring_indices(source_to_use, '\x1b[0m')
                    slice_to_use = None
                    for slice in end_citation_occurrences:
                        if (slice[0] > start_citation_occurrences[0][0]):
                            slice_to_use = slice
                            break
                    source_to_use = source_to_use[:slice_to_use[0]]+"</span>"+source_to_use[slice_to_use[1]:]
                    cited_sources_ls[j] = source_to_use
                    if (j not in sources_idxs_to_show.keys()):
                        sources_idxs_to_show[j] = None
                    break

    sources_to_show = []
    for source_idx in sources_idxs_to_show.keys(): 
        curr_source = clear_ansi(cited_sources_ls[source_idx]).strip()
        curr_source_ls = curr_source.split('\n')
        if ('https://' in curr_source_ls[0]):
            curr_source_ls = curr_source.split('\n')
            curr_url = curr_source_ls[0][8:]
            if (curr_url[:4] == 'www.'):
                curr_url = curr_url[4:]
            curr_source = ' '.join(curr_source_ls[1:]).strip()
            sources_to_show.append(replace_dollar_signs('<b>Source: </b>'+curr_url+'\n\n'+curr_source))
        else:
            sources_to_show.append(replace_dollar_signs('<b>Source: </b>\n\n'+curr_source))
    return sources_to_show

## Page configs
st.set_page_config(initial_sidebar_state="collapsed",layout="wide")

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }

    button.step-up {display: none;}
    button.step-down {display: none;}
    div[data-baseweb] {border-radius: 4px;}

    p {
    font-size:18px;
    }

    .orange-highlight {
        background-color: hsla(42, 100%, 51%, 0.75);
    }

    .big-font {
    font-size:14px !important;
    font-style:italic;  
    }

    .highlighted-font {
    background-color: green; 
    }

</style>
""", unsafe_allow_html=True,
) 

def replace_dollar_signs(text):
    return text.replace('$', '\$').strip()

def continue_from_snippet(response_id, fluency_rating, utility_rating):
    # write results to db
    st.session_state.db_conn.table(st.session_state['annotations_db']).insert({
    "annotator_id": st.session_state["username"], 
    "human_fluency_rating": int(fluency_rating),
    "human_utility_rating": int(utility_rating),
    "op": op,
    "query_id":int(response_id),
    }).execute()    
    st.session_state['touched_response_ids'] += [int(response_id)]
    st.session_state.db_conn.table(st.session_state['annotator_db_str']).update({'annotated_query_ids': st.session_state['touched_response_ids']}).eq('annotator_id', st.session_state["username"]).execute()
    
    # reset fluency/utility button press
    st.session_state["b1_press"] = False

    # increment to the next task
    if (st.session_state["task_n"]<st.session_state["total_tasks"]-1):
        st.session_state["task_n"] += 1
        st.session_state['prec_t2v'] = []
        st.session_state['cov_t2v'] = []
        st.session_state['ANSI_TO_MD_IDX'] = {}
        st.switch_page('pages/response_level.py')
    else:
        st.session_state["hit_finished"] = True
        st.switch_page('pages/done.py')

if ("hit_df" in st.session_state):
    st.header("Task "+str(st.session_state["task_n"]+1)+"/"+str(st.session_state["total_tasks"]))
    subheader_container = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        query_container = st.empty()
        query_container.markdown('''**User Query:**\n'''+st.session_state["hit_df"].iloc[st.session_state["task_n"]]['Question'])
        full_response_container = st.empty()
    op = st.session_state["hit_df"].iloc[st.session_state["task_n"]]['op']
    if (op == 'Snippet'):
        unmarked_response = eval(st.session_state["hit_df"].iloc[st.session_state["task_n"]]['Output'])
        snippets_to_show = get_highlighted_snippet(unmarked_response)
        
        if (len(snippets_to_show) == 0):
            response_id = st.session_state["hit_df"].iloc[st.session_state["task_n"]]['ID']
            continue_from_snippet(response_id, -1, -1)
    
        unmarked_response = "\n\n".join(snippets_to_show)
        
    else:
        unmarked_response = format_remove_quotation_marks(st.session_state["hit_df"].iloc[st.session_state["task_n"]]['Output'])
        cited_response = st.session_state["hit_df"].iloc[st.session_state["task_n"]]['Output (cited)']
        sentences = eval(st.session_state["hit_df"].iloc[st.session_state["task_n"]]['Sent (cited)'])
        if (clear_ansi(sentences[0].strip()) not in clear_ansi(cited_response)):
            if (clear_ansi(sentences[0].strip()) in clear_ansi("\'"+cited_response)):
                unmarked_response = "\'"+format_remove_quotation_marks(st.session_state["hit_df"].iloc[st.session_state["task_n"]]['Output'])
    with col1:
        fluency_container = st.empty()
        unmarked_response = replace_dollar_signs(unmarked_response) 
        full_response_container.write("<p><b>System Response:</b>\n\n"+unmarked_response+"</p>", unsafe_allow_html=True)    

    fluency_options = ['1: Response has misprints or disfluent transitions and sentences', 
                       '2: Response has no misprints and mostly smooth transitions and sentences', 
                       '3: Response has no misprints and all of the sentences flow nicely together']
    fluency_label = "**Fluency Question:** To what extent is the response fluent and coherent?"
    fluency_rating = fluency_container.radio(
                                        label=fluency_label,
                                        options=fluency_options,
                                        index=None,
                                        key="fluency_"+str(st.session_state["task_n"]+1),
                                        )
    if (fluency_rating):
        with col1:
            utility_container = st.empty()
            
        utility_options = ['1: Response includes too many irrelevant details or the query is not addressed', 
                           '2: Response is only a partially satisfying answer to the query', 
                           '3: The response is concise and seems to be a satisfying answer to the query']
        utility_label = "**Utility Question:** To what extent does the response seem to be a useful answer to the query?"
        utility_rating = utility_container.radio(
                            label=utility_label,
                            options=utility_options,
                            index=None,
                            key="utility_"+str(st.session_state["task_n"]+1),
                            )
        if (utility_rating):
            with col1:
                continue_container = st.empty()
                
            if ('b1_press' not in st.session_state):
                st.session_state["b1_press"] = False
            b1_press = continue_container.button('Continue task', on_click=save_start_time)
            if (st.session_state["b1_press"] or b1_press):
                st.session_state["utility_rating"] = int(utility_rating[0])
                st.session_state["fluency_rating"] = int(fluency_rating[0])
                st.session_state["b1_press"] = True
                fluency_container.empty()
                utility_container.empty()
                continue_container.empty()
                op = st.session_state["hit_df"].iloc[st.session_state["task_n"]]['op']
                response_id = st.session_state["hit_df"].iloc[st.session_state["task_n"]]['ID']
                if (op == 'Snippet'):
                    continue_from_snippet(response_id, st.session_state["fluency_rating"], st.session_state["utility_rating"])
                    
                sentences = eval(st.session_state["hit_df"].iloc[st.session_state["task_n"]]['Sent (cited)'])
                num_sentences = len(sentences)
                prec_results = []
                cov_results = []
                placeholders_prec = {}
                placeholders_prec_text = []
                placeholders_cov = []
                placeholders_prec_button = []
                placeholders_cov_button = []
                placeholders_sources = []
                citations_dict = eval(st.session_state["hit_df"].iloc[st.session_state["task_n"]]['Citation Dict'])
                for i in range(num_sentences):
                    with col1:
                        placeholder = st.empty()
                        placeholders_prec_text.append(placeholder)
                        num_citations = len(citations_dict[str(i)]['citation_numbers'])
                        placeholders_prec[i] = []
                        for j in range(num_citations):
                            placeholder = st.empty()
                            placeholders_prec[i].append(placeholder)
                        placeholder = st.empty()
                        placeholders_prec_button.append(placeholder)
                        placeholder = st.empty()
                        placeholder = st.empty()
                        placeholders_cov.append(placeholder)
                        placeholder = st.empty()
                        placeholders_cov_button.append(placeholder)

                def finish_up():
                    # write results to db that user annotated entire response
                    st.session_state.db_conn.table(st.session_state['annotations_db']).insert({
                    "annotator_id": st.session_state["username"], 
                    "human_fluency_rating": int(st.session_state["fluency_rating"]),
                    "human_utility_rating": int(st.session_state["utility_rating"]),
                    "precise_citations": st.session_state['prec_results'],
                    "is_covered": st.session_state['cov_results'],
                    "t2v_precision": st.session_state['prec_t2v'],
                    "t2v_coverage": st.session_state['cov_t2v'],
                    "op": op,
                    "query_id":int(response_id),
                    }).execute()  
                    st.session_state['touched_response_ids'] += [int(response_id)]
                    st.session_state.db_conn.table(st.session_state['annotator_db_str']).update({'annotated_query_ids': st.session_state['touched_response_ids']}).eq('annotator_id', st.session_state["username"]).execute()
                    
                    # reset button presses
                    st.session_state["b1_press"] = False
                    
                    # increment to the next task
                    if (st.session_state["task_n"]<st.session_state["total_tasks"]-1):
                        st.session_state["task_n"] += 1
                        st.session_state['prec_t2v'] = []
                        st.session_state['cov_t2v'] = []
                        st.session_state['ANSI_TO_MD_IDX'] = {}
                        st.switch_page('pages/response_level.py')
                    else:
                        st.session_state["hit_finished"] = True
                        st.switch_page('pages/done.py')
                    
                    return

                def eval_next_sentence(cov_pressed, cov_result, citations_dict, i, save_time, col2_container, sources_placeholder, highlighted_cited_sources):
                    # clear any previous coverage question container
                    if (i > 0):
                        placeholders_prec_text[i].empty()
                        for j in range(0, len(placeholders_prec[i])):
                            placeholders_prec[i][j].empty()
                    
                    # Set state variable for coverage submission button press
                    if ('cov_continue_press_sentence'+str(i)+'_task'+str(st.session_state["task_n"]) not in st.session_state):
                        st.session_state['cov_continue_press_sentence'+str(i)+'_task'+str(st.session_state["task_n"])] = False

                    # if the coverage submission button is pressed, then proceed
                    if ((cov_pressed and cov_result) or st.session_state['cov_continue_press_sentence'+str(i)+'_task'+str(st.session_state["task_n"])]):
                        if (not st.session_state['cov_continue_press_sentence'+str(i)+'_task'+str(st.session_state["task_n"])]):
                            # if pressed for the first time (not on a internal page re-run), record the time
                            save_time(i,'cov') 
                        
                        # Set coverage submission button variables and remove the button
                        cov_pressed = False
                        st.session_state['cov_continue_press_sentence'+str(i)+'_task'+str(st.session_state["task_n"])] = True
                        placeholders_cov_button[i].empty()

                        # Record the coverage results and remove the coverage text and checklist
                        if cov_result == "Yes":
                            cov_results.append({"sentence_id": i, "coverage": 1})
                        elif cov_result == "No":
                            cov_results.append({"sentence_id": i, "coverage": 0}) 
                        else: 
                            cov_results.append({"sentence_id": i, "coverage": -1})                           
                        
                        cov_result = None
                        placeholders_cov[i].empty()
                        placeholders_cov_button[i].empty()

                        # Now, prepare to ask about precision.
                        # Display precision prompt and checklist
                        placeholders_prec_text[i].markdown('<p class="big-font">2. Please select each citation below whose source supports information in the highlighted sentence above.</p>', unsafe_allow_html=True)
                        precision_checklist = []
                        citations = citations_dict[str(i)]['citation_numbers']
                        for j in range(len(citations)):
                            precision_checklist.append(placeholders_prec[i][j].checkbox('['+str(citations[j])+']', key='cb_sentence'+str(i)+'_citation'+str(j)))
                        # Set state variable for precision submission button press
                        if ('continue_press_sentence'+str(i)+'_task'+str(st.session_state["task_n"]) not in st.session_state):
                            st.session_state['continue_press_sentence'+str(i)+'_task'+str(st.session_state["task_n"])] = False
                    
                        # Display precision submission button
                        pressed = placeholders_prec_button[i].button('Continue task', key='continue_press_button_sentence'+str(i))
                        if ((len(citations) == 0) or (len(highlighted_cited_sources)==0)):
                            pressed = True
                        # Once we get the precision result...
                        if (pressed or st.session_state['continue_press_sentence'+str(i)+'_task'+str(st.session_state["task_n"])]):
                            # save T2V
                            if (not st.session_state['continue_press_sentence'+str(i)+'_task'+str(st.session_state["task_n"])]):
                                save_time(i,'prec')
                            
                            # remove prec button
                            placeholders_prec_button[i].empty()
                            st.session_state['continue_press_sentence'+str(i)+'_task'+str(st.session_state["task_n"])] = True
                            
                            # Stash the precision result
                            prec_results.append({"sentence_id": i, "annotations": [int(x) for x in precision_checklist]})
                            placeholders_prec_text[i].empty()
                            for j in range(0, len(placeholders_prec[i])):
                                placeholders_prec[i][j].empty()
                            
                            i += 1
                            # On to the next sentence, if there is one
                            if (i < num_sentences):
                                # Get next sentence
                                sentence = sentences[i]
                                # Update subheading
                                subheader_container.subheader("Sentence "+str(i+1)+"/"+str(len(sentences)))

                                # Get and display highlighted response
                                cited_response = st.session_state["hit_df"].iloc[st.session_state["task_n"]]['Output (cited)']
                                sentence = clear_ansi(sentence)
                                highlighted_response = clear_ansi(cited_response).replace(sentence.strip(), "<span class='orange-highlight'>"+sentence.strip()+"</span>")

                                highlighted_response = replace_dollar_signs(highlighted_response)
                                full_response_container.write("<p><b>Cited System Response:</b>\n\n"+highlighted_response+"</p>", unsafe_allow_html=True)
                                
                                # Get and display highlighted sources for this sentence
                                cited_sources_ls = eval(st.session_state["hit_df"].iloc[st.session_state["task_n"]]['Used Sources (cited)'])
                                citations = citations_dict[str(i)]['citation_numbers']
                                highlighted_cited_sources = get_cited_sources_for_sentence(cited_sources_ls, citations)
                                with col2_container:
                                    sources_placeholder.write("\n_____________________________________________________________\n".join(highlighted_cited_sources), unsafe_allow_html=True)
                                num_citations_in_sentence = len(citations)
                                if ((num_citations_in_sentence == 0) or (len(highlighted_cited_sources) == 0)):
                                    # If there are no citations in the first sentence, set up variables for the next sentence
                                    st.session_state['cov_continue_press_sentence'+str(i)+'_task'+str(st.session_state["task_n"])] = True
                                    st.session_state['continue_press_sentence'+str(i)+'_task'+str(st.session_state["task_n"])] = True
                                    cov_result = "NA"
                                    cov_pressed = True
                                else:
                                    # build the string citation list
                                    citations_str = ''
                                    for k in range(len(citations)):
                                        citation_num = citations[k]
                                        if (len(citations)==1):
                                            citations_str += '['+str(citation_num)+']'
                                            break
                                        if (k == len(citations)-1):
                                            citations_str += 'and ['+str(citation_num)+']'
                                        elif (len(citations)==2):
                                            citations_str += '['+str(citation_num)+'] '
                                        else:
                                            citations_str += '['+str(citation_num)+'], '
                                    if (num_citations_in_sentence == 1):
                                        coverage_text = '*1. Does the source of '+citations_str+' support **all** information in the sentence?*'
                                    else:
                                        coverage_text = '*1. Do the sources of '+citations_str+' together support **all** information in the sentence?*'
                                    # Show the coverage question and multiple choice
                                    cov_result = placeholders_cov[i].radio(
                                                label=coverage_text,
                                                options=["Yes", "No"],
                                                index=None,
                                                key=str(i)+'coverage',
                                                args=(i,'cov',))
                                    cov_pressed = placeholders_cov_button[i].button('Continue task', key='cov_continue_press_button_sentence'+str(i))

                                # Next call to eval
                                eval_next_sentence(cov_pressed, cov_result, citations_dict, i, save_time, col2_container, sources_placeholder, highlighted_cited_sources)
                            else:
                                # If no next sentence, save everything to the database
                                st.session_state['prec_results'] = prec_results
                                st.session_state['cov_results'] = cov_results
                                finish_up()
                                return
                i = 0
                # Get cited sentences and display subheader
                sentence = sentences[i]
                sentence = clear_ansi(sentence)
                subheader_container.subheader("Sentence "+str(i+1)+"/"+str(len(sentences)))

                # Get cited response and sources
                cited_response = st.session_state["hit_df"].iloc[st.session_state["task_n"]]['Output (cited)']
                cited_sources = st.session_state["hit_df"].iloc[st.session_state["task_n"]]['Used Sources (cited)']
                # Display cited response with highlighted sentence
                highlighted_response = clear_ansi(cited_response).replace(sentence.strip(), "<span class='orange-highlight'>"+sentence.strip()+"</span>")
                if (sentence.strip() in clear_ansi(cited_response)):
                    highlighted_response = clear_ansi(cited_response).replace(sentence.strip(), "<span class='orange-highlight'>"+sentence.strip()+"</span>")
                elif (sentence.strip() in clear_ansi("\'"+cited_response)):
                    highlighted_response = clear_ansi("\'"+cited_response).replace(sentence.strip(), "<span class='orange-highlight'>"+sentence.strip()+"</span>")
                highlighted_response = replace_dollar_signs(highlighted_response)
                full_response_container.write("<p><b>Cited System Response:</b>\n\n"+highlighted_response+"</p>", unsafe_allow_html=True)
                

                # Get highlighted sources for this sentence
                cited_sources_ls = eval(cited_sources)
                citations_dict = eval(st.session_state["hit_df"].iloc[st.session_state["task_n"]]['Citation Dict'])
                citations = citations_dict[str(i)]['citation_numbers']
                highlighted_cited_sources = get_cited_sources_for_sentence(cited_sources_ls, citations)
                # Display highlighted sources for this sentence
                col2_container = col2.container(height=600)
                with col2_container:
                    sources_placeholder = st.empty()
                    sources_placeholder.write("\n_____________________________________________________________\n".join(highlighted_cited_sources), unsafe_allow_html=True)
                if ((len(citations)==0) or (len(highlighted_cited_sources)==0)):
                    # If there are no citations in the first sentence, set up variables for the next sentence
                    st.session_state['cov_continue_press_sentence'+str(i)+'_task'+str(st.session_state["task_n"])] = True
                    st.session_state['continue_press_sentence'+str(i)+'_task'+str(st.session_state["task_n"])] = True
                    if ('prec_t2v' not in st.session_state):
                        st.session_state['prec_t2v'] = []
                    if ('cov_t2v' not in st.session_state):
                        st.session_state['cov_t2v'] = []
                    cov_result = "NA"
                    cov_pressed = True
                else:
                    # First, get citations
                    num_citations_in_sentence = len(citations)
                           
                    # Build the string citation list
                    citations_str = ''
                    for k in range(len(citations)):
                        citation_num = citations[k]
                        if (len(citations)==1):
                            citations_str += '['+str(citation_num)+']'
                            break
                        if (k == len(citations)-1):
                            citations_str += 'and ['+str(citation_num)+']'
                        elif (len(citations)==2):
                            citations_str += '['+str(citation_num)+'] '
                        else:
                            citations_str += '['+str(citation_num)+'], '
                    if (num_citations_in_sentence == 1):
                        coverage_text = '*1. Does the source of '+citations_str+' support **all** information in the sentence?*'
                    else:
                        coverage_text = '*1. Do the sources of '+citations_str+' together support **all** information in the sentence?*'
                    # Show the coverage question and multiple choice
                    cov_result = placeholders_cov[i].radio(
                                label=coverage_text,
                                options=["Yes", "No"],
                                index=None,
                                key=str(i)+'coverage',
                                args=(i,'cov',))
                    cov_pressed = placeholders_cov_button[i].button('Continue task', key='cov_continue_press_button_sentence'+str(i))
                
                # Proceed to coverage & then the next sentence
                eval_next_sentence(cov_pressed, cov_result, citations_dict, i, save_time, col2_container, sources_placeholder, highlighted_cited_sources)
                    