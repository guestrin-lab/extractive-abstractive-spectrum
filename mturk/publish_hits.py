import boto3
from supabase import create_client
import pandas as pd
import random
import string
import argparse
from utils import init_db_connection, init_mturk_connection
from quals import *

SANDBOX = False

def generate_completion_code():
    alphanumeric = string.ascii_letters + string.digits
    return ''.join(random.choice(alphanumeric) for _ in range(20))

def main(args):
    db_conn = init_db_connection()
    mturk = init_mturk_connection(sandbox=SANDBOX)
    print("I have $" + mturk.get_account_balance()['AvailableBalance'] + " in my account")

    compensation = '2.55'
    max_num_hits = 10
    description = 'Determine whether answers to questions are fluent, useful, and supported by accompanying sources.'
    quals_list_for_hit = []

    # get the correct qualifier IDs for this task
    if (args.task == 'NQ'):
        title = 'Citation Evaluation Study: NQ'
        is_included_qual_id = REAL_INCLUDED_QUAL_ID
        num_hits_compeleted_qual_id = REAL_NUM_NQ_QUAL_ID
        question = open(file='nq_question.xml',mode='r').read()
    # if (args.task == 'MH'):
    #     title = 'Citation Evaluation Study: MH'
    #     is_included_qual_id = REAL_INCLUDED_QUAL_ID
    #     num_hits_compeleted_qual_id = REAL_NUM_MH_QUAL_ID
    #     question = open(file='mh_question.xml',mode='r').read()

    # if (args.task == 'ELI3'):
    #     title = 'Citation Evaluation Study: Explain to a third-grader'
    #     is_included_qual_id = REAL_INCLUDED_QUAL_ID
    #     num_hits_compeleted_qual_id = REAL_NUM_ELI3_QUAL_ID
    #     question = open(file='eli3_question.xml',mode='r').read()

    # if (args.task == 'MASH'):
    #     title = 'Citation Evaluation Study: Medical Advice Questions'
    #     is_included_qual_id = REAL_INCLUDED_QUAL_ID
    #     num_hits_compeleted_qual_id = REAL_NUM_EMR_QUAL_ID
    #     question = open(file='emr_question.xml',mode='r').read()
    else:
        print('Please specify a task')
        exit()

    for i in range(args.n):
        # generate completion code and add it to the database for this hit
        prev_values = pd.DataFrame(db_conn.table("hit_completion_codes").select('hit_specific_id').execute().data)['hit_specific_id']
        prev_max = max(prev_values)
        hit_specific_id = prev_max+1
        completion_code = generate_completion_code() + str(hit_specific_id)
        db_conn.table("hit_completion_codes").insert({'hit_specific_id': hit_specific_id, 'completion_code':completion_code}).execute()

        # use the question format to make a new hit
        curr_question = question.replace('!!!hit_specific_id!!!', str(hit_specific_id))
        new_hit = mturk.create_hit(
            Title = title,
            Description = description,
            Keywords = 'text, labeling, reading, learning, QA, citations',
            Reward = compensation, 
            MaxAssignments = 1,
            LifetimeInSeconds = 86400,
            AssignmentDurationInSeconds = 3600,
            AutoApprovalDelayInSeconds = 2*86400,
            Question = curr_question,
            AssignmentReviewPolicy={
                'PolicyName':'ScoreMyKnownAnswers/2011-09-01',
                'Parameters':[
                    {'Key':'AnswerKey', 'MapEntries':[
                        {'Key': 'completion_code', 
                        'Values':[completion_code]
                        }]},
                    {'Key': 'ApproveIfKnownAnswerScoreIsAtLeast', 'Values':['1']}, 
                    {'Key': 'RejectIfKnownAnswerScoreIsLessThan', 'Values':['0']},
                    {'Key': 'RejectReason', 
                    'Values':['Sorry, the completion code is incorrect.']},
                ]
            },
            QualificationRequirements=quals_list_for_hit+[
                                        { # Number of HITs approved
                                            'QualificationTypeId': '00000000000000000040',
                                            'Comparator': 'GreaterThanOrEqualTo',
                                            'IntegerValues': [135], #TODO 0 if in SB
                                            'ActionsGuarded': 'PreviewAndAccept'
                                        },
                                        { # Percentage of HITs approved
                                            'QualificationTypeId': '000000000000000000L0',
                                            'Comparator': 'GreaterThanOrEqualTo',
                                            'IntegerValues': [96], 
                                            'ActionsGuarded': 'PreviewAndAccept'
                                        },
                                        {
                                            'QualificationTypeId': '00000000000000000071',
                                            'Comparator': 'In',
                                            'LocaleValues': [
                                            { 'Country': "US" },
                                            ],
                                            'ActionsGuarded': 'PreviewAndAccept'
                                        },
                                        {
                                            'QualificationTypeId': is_included_qual_id,
                                            'Comparator': 'Exists',
                                            'ActionsGuarded': 'DiscoverPreviewAndAccept'
                                        },
                                        {
                                            'QualificationTypeId': num_hits_compeleted_qual_id,
                                            'Comparator': 'LessThan',
                                            'IntegerValues': [max_num_hits],
                                            'ActionsGuarded': 'DiscoverPreviewAndAccept' 
                                        },
                                    ]
        )

    # print("A new HIT has been created. You can preview it here:")
    print("https://worker.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
    print("HITID = " + new_hit['HIT']['HITId'] + " (Use to Get Results)")
    print("I have $" + mturk.get_account_balance()['AvailableBalance'] + " in my account")
    # Remember to modify the URL above when you're publishing
    # HITs to the live marketplace.
    # Use: https://worker.mturk.com/mturk/preview?groupId=

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("--n", help="Number of HITs to publish", type=int)
    parser.add_argument("--task", help="MTurk batch", type=str) # Trial, NQ
    args = parser.parse_args()
    main(args)

# python publish_hits.py --task Trial --n 1