import pandas as pd 
from ckip_transformers.nlp import CkipWordSegmenter
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="./")
    args = parser.parse_args()
    return args

def main(args):
    df_course = pd.read_csv(args.data_dir / 'courses.csv')
    df_course_item = pd.read_csv(args.data_dir / 'course_chapter_items.csv')
    col = ['course_name','teacher_intro', 'groups', 'sub_groups', 'topics',
        'course_published_at_local', 'description', 'will_learn',
        'required_tools', 'recommended_background', 'target_group']
    item = {}
    for k,v in df_course_item[['course_id','chapter_item_name']].iterrows():
        idx = v['course_id']
        content = v['chapter_item_name']
        if item.get(idx,-1) == -1:
            item[idx] = [content]
        else:
            item[idx].append(content)
    for k,v in df_course.iterrows():
        idx = v['course_id']
        if item.get(idx,-1) == -1:
            item[idx] = []
        for colu in col:
            item[idx].append(str(v[colu]))
    for k,v in item.items():
        txt = ''
        for i in v:
            if str(i) == 'nan': continue
            txt = txt + ' ' + str(i)
        item[k] = txt
    df_t = pd.DataFrame({'course_id':item.keys(),'content':item.values()})
    ws_driver  = CkipWordSegmenter(model="bert-base")
    ws_driver = CkipWordSegmenter(device=0) # -1: CPU ; 0: GPU
    text = df_t.content.values.tolist()
    # Enable sentence segmentation
    ws = ws_driver(text, batch_size=64, max_length=128)
    Container = []
    for i in ws:
        Container.append(' '.join([ wd for wd in re.sub('[^\u4e00-\u9fa5 ]+', '', ' '.join(i)).split(' ') if wd != '' ]))
    df_t['parser_'] = Container
    df_t[['course_id','parser_']].to_csv(args.data_dir / 'course_clean.csv',index=False,encoding='utf-8-sig')

    df_user = pd.read_csv(args.data_dir /  'users.csv')
    df_user_test = pd.read_csv(args.data_dir /  'test_unseen.csv')

    USERS = {}
    for k,v in df_user.iterrows():
        user = v['user_id']
        occupation = v['occupation_titles']
        inter = v['interests']
        react = v['recreation_names']
        content = f"{occupation} {inter} {react}"
        USERS[user] = content

    USERS_TEST = {}
    for k,v in df_user_test.iterrows():
        user = v['user_id']
        USERS_TEST[user] = USERS[user]


    df_u = pd.DataFrame({'user_id':USERS_TEST.keys(),'interest':USERS_TEST.values()})
    df_u['interest'] = df_u['interest'].apply(lambda x: x.replace('nan','').replace('_',' ').replace(',',' ').replace('、',' ').replace('與',' ').replace('及',' '))
    Container = []
    for i in df_u.interest.values:
        temp = []
        text = [ wd for wd in i.split(' ') if wd !='']
        for wdd in text:
            if len(wdd) % 2 == 0:
                for L in range(len(wdd)//2):
                    temp.append(wdd[L*2:(L+1)*2])
            else:
                temp.append(wdd)
        temp = [ wddd for wddd in temp if wddd != '']
        Container.append(' '.join(temp))
    df_u['parser_'] = Container
    df_u[['user_id','parser_']].to_csv(args.data_dir / 'user_clean.csv',index=False,encoding='utf-8-sig')


if __name__ == '__main__':
    args = parse_args()
    main(args)

