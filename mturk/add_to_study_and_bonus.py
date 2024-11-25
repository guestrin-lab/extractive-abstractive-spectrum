import boto3
from supabase import create_client
import pandas as pd
import argparse
from quals import *
from utils import init_db_connection, init_mturk_connection
import argparse

mturk = init_mturk_connection(sandbox=False)
db_conn = init_db_connection()

def add_worker_to_study(worker_id):
    qual_id = REAL_INCLUDED_QUAL_ID
    response = mturk.associate_qualification_with_worker(
            QualificationTypeId=qual_id,
            WorkerId=worker_id,
            IntegerValue=0
        )

def give_worker_zero_trials_for_all(worker_id):
    num_task_qual_id_ls = [REAL_NUM_NQ_QUAL_ID, REAL_NUM_ELI3_QUAL_ID, REAL_NUM_MH_QUAL_ID, REAL_NUM_EMR_QUAL_ID]
    for num_task_qual_id in num_task_qual_id_ls:
        response = mturk.associate_qualification_with_worker(
                    QualificationTypeId=num_task_qual_id,
                    WorkerId=worker_id,
                    IntegerValue=0
                )

def give_all_workers_n_trials_for_NQ(n):
    worker_ids = pd.DataFrame(db_conn.table('mturk_annotators').select("*").execute().data)
    worker_ids = worker_ids['annotator_id'].tolist()
    for worker_id in worker_ids:
        response = mturk.associate_qualification_with_worker(
                        QualificationTypeId=REAL_NUM_NQ_QUAL_ID,
                        WorkerId=worker_id,
                        IntegerValue=n
                    )
    print('Done!')

def give_worker_zero_trials_for_NQ(worker_id):
    response = mturk.associate_qualification_with_worker(
            QualificationTypeId=REAL_NUM_NQ_QUAL_ID,
            WorkerId=worker_id,
            IntegerValue=0
        )

def get_touched_query_ids(worker_id, db_name):
    # Obtain the previously touched query IDs
    annotator_rows = db_conn.table(db_name).select("*").execute()
    for row in annotator_rows.data:
        if (row['annotator_id']==worker_id):
            # get annotator's annotation history if they've annotated before
            return row['annotated_query_ids']
    return []

def sign_up_worker_to_study(username):
    worker_id = username.replace('_Trial', '')

    # Give workers the necessary qualifications for all tasks
    add_worker_to_study(worker_id)
    give_worker_zero_trials_for_all(worker_id)

    # Add to supabase tables for all tasks
    db_conn = init_db_connection()
    
    db_conn.table("mturk_qualified_nq_annotators").insert({'annotator_id': worker_id, 'annotated_query_ids': []}).execute()
    db_conn.table("mturk_qualified_eli3_annotators").insert({'annotator_id': worker_id, 'annotated_query_ids': []}).execute()
    db_conn.table("mturk_qualified_mh_annotators").insert({'annotator_id': worker_id, 'annotated_query_ids': []}).execute()
    db_conn.table("mturk_qualified_emr_annotators").insert({'annotator_id': worker_id, 'annotated_query_ids': []}).execute()
    
def get_assignment_id(worker_id):
    # Create a reusable Paginator
    hit_paginator = mturk.get_paginator('list_hits_for_qualification_type')
    # Create a PageIterator from the Paginator
    hit_page_iterator = hit_paginator.paginate(QualificationTypeId=REAL_INCLUDED_QUAL_ID)

    for page in hit_page_iterator:
        hit_responses = page['HITs']
        for hit_r in hit_responses:
            hit_id = hit_r['HITId']
            num_completed = hit_r['NumberOfAssignmentsCompleted']
            if (num_completed > 0):
                # Create a reusable Paginator
                asmt_paginator = mturk.get_paginator('list_assignments_for_hit')
                # Create a PageIterator from the Paginator
                asmt_page_iterator = asmt_paginator.paginate(HITId=hit_id, AssignmentStatuses=['Submitted', 'Approved', 'Rejected'])
                for asmt_page in asmt_page_iterator:
                    asmt_responses = asmt_page['Assignments']
                    for asmt_r in asmt_responses:
                        curr_worker_id = asmt_r['WorkerId']
                        if (curr_worker_id == worker_id):
                            return asmt_r['AssignmentId']
    print('Came up with nothing for the assignment_id :(')
    return None

def give_bonus(worker_id):
    assignment_id = get_assignment_id(worker_id)
    response = mturk.send_bonus(
                                WorkerId=worker_id,
                                BonusAmount='2', 
                                AssignmentId=assignment_id,
                                Reason='bonus',
                                UniqueRequestToken='bonus_'+worker_id 
                            )
    print(response)
    
def main(args):
    sign_up_worker_to_study(args.worker_id, True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", help="MTurk WorkerID", type=str)
    args = parser.parse_args()
    main(args)
