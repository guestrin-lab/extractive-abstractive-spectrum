import streamlit as st
import random
import string
import boto3
import pandas as pd 

st.set_page_config(initial_sidebar_state="collapsed")

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

if (int(st.session_state["hit_specific_id"]) != 1): # if this is an MTurk annotator session that must be tracked:
    # create mturk client
    mturk = boto3.client('mturk',
                        aws_access_key_id = st.secrets["aws_access_key_id"], # TODO add to .streamlit/secrets.toml
                        aws_secret_access_key = st.secrets["aws_secret_access_key"], # TODO add to .streamlit/secrets.toml
                        region_name='us-east-1',
                        endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
                        )
    # get new number of completed HITs for this group
    num_worker_annotations = len(st.session_state["db_conn"].table(st.session_state['annotations_db']).select("query_id").eq("annotator_id", st.session_state["username"]).execute().data)//st.session_state["total_tasks"]
    
    # update the worker's qualification
    worker_id = st.session_state["username"]
    
    # If you are testing mturk mode (i.e. accessing the webapp with url parameter != 1)
    response = mturk.associate_qualification_with_worker(
                QualificationTypeId=st.session_state['NUM_TRIALS_QUAL_ID'],
                WorkerId=worker_id,
                IntegerValue=num_worker_annotations
            )

# Obtain the completion code associated with the webapp url parameter
# HITs posted on MTurk set the webapp url parameter and also have access to the hit_completion_codes supabase below;
# this allows for unique complete codes for each HIT
completion_code = st.session_state["db_conn"].table("hit_completion_codes").select("completion_code").eq("hit_specific_id", st.session_state["hit_specific_id"]).execute().data[0]['completion_code']

if (st.session_state["hit_finished"]):
    st.markdown('''# Done! Thank you so much! :raised_hands:''')
    st.markdown('''Please enter the one-time-use completion code below on the MTurk HIT webpage for compensation.''')
    st.markdown(completion_code)
else:
    st.markdown('''# Incomplete''')