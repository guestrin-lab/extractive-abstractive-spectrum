from naturalQuestions import NaturalQuestions
from wikiMultiHopQA import WikiMultiHopQA
from mashQA import MashQA
from instructions import nq_baseline_instruction_str, mh_baseline_instruction_str, mash_baseline_instruction_str
import argparse

def main(args):
    if ((args.data == 'nq') or (args.data == 'eli5_nq')):
        baseline_instruction_str = nq_baseline_instruction_str
        data = NaturalQuestions(seed=0)
    elif (args.data == 'multihop'):
        baseline_instruction_str = mh_baseline_instruction_str
        data = WikiMultiHopQA(seed=0)
    elif (args.data == 'mash'):
        baseline_instruction_str = mash_baseline_instruction_str
        data = MashQA(seed=0)
    else:
        print('Need to implement baseline_instruction_str for dataset!')
        return
    
    for i in range(args.start_n, args.start_n+args.n):
        question = data[i]['question']
        if (args.data == 'eli5_nq'):
            question = 'Explain to a third-grader: '+question
        print()
        print(i)
        print()
        print(baseline_instruction_str+question)
        print()
        breakpoint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_n', type=int, default=0)
    parser.add_argument('--n', type=int, default=60)
    parser.add_argument('--data', type=str, required=True) # 'nq' or 'multihop' or 'eli5_nq' or 'mash'
    args = parser.parse_args()
    main(args)