"""
Add entries to instances_to_annotate database
"""
import argparse
import numpy as np
import pandas as pd
from supabase import create_client
import os

def init_connection():
    url = os.environ['SUPABASE_URL'] # TODO
    key = os.environ['SUPABASE_KEY'] # TODO
    return create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

def load_ops(args):
    db_conn = init_connection()
    df = pd.read_csv(args.filename)
    ids = np.unique(df['ID'])
    ops = ["Snippet", "Quoted", "Paraphrased", "Entailed", "Abstractive"]
    print(ids)
    for x in ids:
        db_conn.table(args.db).insert({
                        "query_id": int(x), 
                        "ops": ops,
                    }).execute() 
        
def load_baselines(args):
    db_conn = init_connection()
    df = pd.read_csv(args.filename)
    ids = np.unique(df['ID'])
    print(ids)
    for x in ids:
        ops = df[df['ID']==x]['op'].to_list()
        db_conn.table(args.db).insert({
                        "query_id": int(x), 
                        "ops": ops,
                    }).execute() 

def load_all(args):
    db_conn = init_connection()
    df = pd.read_csv(args.filename)
    ids = np.unique(df['ID'])
    print(ids)
    for x in ids:
        ops = df[df['ID']==x]['op'].to_list()
        db_conn.table(args.db).insert({
                        "query_id": int(x), 
                        "ops": ops,
                    }).execute() 

def main(args):
    if (args.original_ops):
        load_ops(args)
    elif (args.all):
        load_all(args)
    else:
        load_baselines(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default=None, type=str)
    parser.add_argument('--db', default=None, type=str)
    parser.add_argument('--original_ops', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    main(args)