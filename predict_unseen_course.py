import pandas as pd 
from rank_bm25 import BM25Okapi
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="./")
    parser.add_argument("--output_path", type=Path)
    args = parser.parse_args()
    return args

def main(args):
    df_course = pd.read_csv(args.data_dir / 'course_clean.csv')
    df_course['parser_'] = df_course['parser_'].apply(lambda x: ' '.join([ i for i in x.split(' ') if len(i) > 1]))
    df_test_unseen = pd.read_csv(args.data_dir / 'user_clean.csv')
    tokenized_corpus = [doc.split(" ") for doc in df_course['parser_']]
    bm25 = BM25Okapi(tokenized_corpus)

    for ratio in [0]:
        Answer = []
        for idx,rows in df_test_unseen.iterrows():
            query = rows['parser_']
            tokenized_query = query.split(" ")
            tokenized_query = [ i for i in tokenized_query if ((i != '') and (len(i) > 1))]
            # print(tokenized_query)
            doc_scores = bm25.get_scores(tokenized_query)
            doc_scores = softmax(doc_scores)
            doc_scores = np.round(doc_scores,2)
            doc_scores_index = [i[0] for i in sorted(enumerate(doc_scores), key=lambda k: k[1], reverse=True)]
            sort_container = []
            for index in doc_scores_index:
                if doc_scores[index] > ratio:
                    sort_container.append(index)
                else:
                    break
            if len(sort_container) == 0:
                sort_container.append(doc_scores_index[0])
            answer = []
            for pos in sort_container:
                cid = df_course.at[pos,'course_id']
                answer.append(cid)
            Answer.append(' '.join(answer))

    df_test_unseen['course_id'] = Answer 
    df_output = df_test_unseen[['user_id','course_id']]
    df_output.to_csv(args.output_path,index=False,encoding='utf-8-sig')

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
    
