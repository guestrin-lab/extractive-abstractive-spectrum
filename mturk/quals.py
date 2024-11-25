import boto3
from supabase import create_client
import pandas as pd
import argparse
from utils import init_db_connection, init_mturk_connection

SANDBOX = False

mturk = init_mturk_connection(sandbox=SANDBOX)
db_conn = init_db_connection()

# If an annotator has this qualification, they passed the screener and were admitted into the study
REAL_INCLUDED_QUAL_ID = '' # TODO make a qualification (can do this on the MTurk requestor website)

# These qualifications count how many annotation sessions the annotator completed for the respective query distribution
REAL_NUM_NQ_QUAL_ID = '' # TODO 
REAL_NUM_ELI3_QUAL_ID = '' # TODO 
REAL_NUM_MH_QUAL_ID = '' # TODO 
REAL_NUM_EMR_QUAL_ID = '' # TODO 
