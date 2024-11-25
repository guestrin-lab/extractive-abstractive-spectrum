import boto3
from supabase import create_client


SB_MTURK_ENDPOINT = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
REAL_MTURK_ENDPOINT = 'https://mturk-requester.us-east-1.amazonaws.com'

def init_db_connection():
    url = os.environ['SUPABASE_URL'] # TODO
    key = os.environ['SUPABASE_KEY'] # TODO
    return create_client(url, key)

def init_mturk_connection(sandbox):
    mturk_endpoint = SB_MTURK_ENDPOINT if sandbox else REAL_MTURK_ENDPOINT
    mturk = boto3.client('mturk',
    aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID'] # TODO
    aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY'] # TODO
    region_name='us-east-1',
    endpoint_url = mturk_endpoint
    )
    return mturk

