import streamlit as st
import pandas as pd
import ssl
from streamlit_gsheets import GSheetsConnection
from supabase import create_client, Client
import random

ssl._create_default_https_context = ssl._create_stdlib_context
st.set_page_config(initial_sidebar_state="collapsed",
                   page_title="Username Form", 
                   page_icon=":mag_right:")

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

# Connect to supabase
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"] # TODO add to .streamlit/secrets.toml
    key = st.secrets["SUPABASE_KEY"] # TODO add to .streamlit/secrets.toml
    return create_client(url, key)

db_conn = init_connection()
st.session_state["db_conn"] = db_conn
## Page configs
st.title("Citation Quality Evaluation")

st.session_state["username"] = st.text_input(
        "Please enter your username",
        key="get_username",
        max_chars=20,
    )
st.session_state["hit_specific_id"] = st.query_params['hit_specific_id']
st.session_state["hit_finished"] = False

if (st.session_state["username"]):
    st.session_state["total_tasks"] = 5 # The number of tasks shown per HIT. E.g. 5 to display one of each OP 
    
    # Set streamlit variables to the corresponding Supabase table names
    if (int(st.session_state["hit_specific_id"]) == 1):
        # Initiate an annotation session without tracking the number of sessions completed by the annotator with an MTurk qual
        conn = st.connection("gsheets_example", type=GSheetsConnection) # TODO add to .streamlit/secrets.toml
        st.session_state['annotations_db'] = 'annotations'
        instances_to_annotate = 'instances_to_annotate' 
        st.session_state['annotator_db_str'] = 'annotators'
    else:
        # Initiate an annotation session that tracks the number of sessions completed by the annotator with an MTurk qual
        # See done.py
        conn = st.connection("mturk_gsheets_example", type=GSheetsConnection) # TODO add to .streamlit/secrets.toml
        st.session_state['annotations_db'] = 'mturk_annotations'
        instances_to_annotate = 'mturk_instances_to_annotate' 
        st.session_state['annotator_db_str'] = 'mturk_annotators'
        st.session_state['NUM_TRIALS_QUAL_ID'] = '' # TODO add the MTurk qualification ID associated with the annotator's session count
    
    # get data
    df = conn.read()

    # get annotator's history
    annotator_rows = db_conn.table(st.session_state['annotator_db_str']).select("*").execute()
    touched_response_ids = None
    promised_query_ids = []
    promised_ops = []
    for row in annotator_rows.data:
        if (row['annotator_id']==st.session_state["username"]):
            # get annotator's annotation history if they have annotated before
            touched_response_ids = row['annotated_query_ids']
            promised_query_ids = row['promised_query_ids']
            promised_ops = row['promised_ops']
    if (touched_response_ids is None):
        st.switch_page('pages/unknown_user.py')
    st.session_state['touched_response_ids'] = touched_response_ids
    
    # get all instances that still require annotation and meet the annotator's constraints,
    # i.e. no annotator may evaluate more than one of the multiple generations for a query
    remaining_response_ids = pd.DataFrame(db_conn.table(instances_to_annotate).select("*").execute().data)
    if (len(remaining_response_ids) == 0):
        st.switch_page('pages/no_more.py')
    remaining_response_ids = remaining_response_ids.sort_values(by='query_id', ascending=True)
    viable_response_ids = remaining_response_ids[~remaining_response_ids['query_id'].isin(touched_response_ids)]
    
    # select the desired number of viable instances that are each a different generation type
    i = 0
    n_hit = 0
    hit_op_ls = []
    hit_id_ls = []
    tt = viable_response_ids['query_id'].tolist()
    # continue until there are no more responses to annotate or all OPs are collected 
    while ((i < len(viable_response_ids)) & (n_hit < st.session_state["total_tasks"])):
        instance = viable_response_ids.iloc[i]
        remaining_ops = instance['ops']
        # shuffle to avoid undesired ordering patterns
        remaining_ops_shuffled_copy = random.sample(remaining_ops.copy(), len(remaining_ops))
        for op in remaining_ops_shuffled_copy:
            # if this op is not yet in the hit, add it
            if (op not in hit_op_ls):
            # if (True):
                hit_op_ls.append(op)
                query_id = int(instance['query_id'])
                hit_id_ls.append(query_id)
                n_hit += 1
                # remove op from instances_to_annotate
                if (len(remaining_ops) == 1):
                    # remove row from instances_to_annotate
                    db_conn.table(instances_to_annotate).delete().eq('query_id', query_id).execute()
                else:
                    remaining_ops.remove(op) 
                    db_conn.table(instances_to_annotate).update({'ops': remaining_ops}).eq('query_id', query_id).execute()
                break
        i+=1
    # record the generation type (hit_op_ls) for the id of each query (hit_id_ls) chosen for the annotation session
    st.session_state["hit_ops"] = hit_op_ls
    st.session_state["hit_response_ids"] = hit_id_ls

    # form the dataframe of instance info for this hit
    rows_to_annotate = []
    for query_id, op in zip(st.session_state["hit_response_ids"], st.session_state["hit_ops"]):
        rows_to_annotate.append(df[(df['ID']==query_id)&(df['op']==op)])
    if (len(rows_to_annotate)==0):
        st.switch_page('pages/no_more.py')
        
    hit_df = pd.concat(rows_to_annotate, ignore_index=True) 
    if ((len(hit_df)==0) or (st.session_state["total_tasks"]==0)):
        st.switch_page('pages/no_more.py')

    # handle the cases where there are less than the desired number of instances still requiring annotation
    promised_query_ids.append(st.session_state["hit_response_ids"]+[-1]*(st.session_state["total_tasks"]-len(st.session_state["hit_response_ids"])))
    promised_ops.append(st.session_state["hit_ops"]+["Null"]*(st.session_state["total_tasks"]-len(st.session_state["hit_response_ids"])))
    db_conn.table(st.session_state['annotator_db_str']).update({'promised_query_ids': promised_query_ids,  'promised_ops':promised_ops}).eq('annotator_id', st.session_state["username"]).execute()
    st.session_state["total_tasks"] = min(st.session_state["total_tasks"], len(hit_df))
    st.session_state["hit_df"] = hit_df
    st.session_state["task_n"] = 0

    # progress to the annotation task
    st.switch_page('pages/response_level.py')
  




