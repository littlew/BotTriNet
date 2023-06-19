import pandas as pd
from sklearn.model_selection import train_test_split


def read_dataset(files, path = 'dataset/cresci'):
    df_all = pd.DataFrame()
    for file in files:
        this_path = path + '/' + file
        print(this_path)
        df = pd.read_csv(this_path)
        df['text'] = df['text'].apply(lambda x: "" if type(x) == float else x)
        df = df[['text', 'created_at', 'user_id']]
        if 'genuine' in file: df['label'] = 0
        else: df['label'] = 1
        df_all = pd.concat([df_all, df])

    df_all = df_all.dropna()
    df_all['user_id'] = df_all['user_id'].astype('int')
    df_all['user_id'] = df_all['user_id'].astype('str')
    return df_all

def split_dataset(df, test_size = 0.3):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, test_df

# 从train_df的text中，按照label=1的text占据ratio比例，重新采样
def sample_corpus(df_train, pos_ratio=0.5):
    df_0 = df_train[df_train['label'] == 0]
    df_1 = df_train[df_train['label'] == 1]
    cnt_0 = len(df_0)
    cnt_1 = len(df_1)
    neg_ratio = 1 - pos_ratio
    if cnt_1 / cnt_0 <= pos_ratio / neg_ratio: # 正样本更少
        sample_1 = cnt_1
        sample_0 = int(neg_ratio / pos_ratio * sample_1)
    else: # 负样本更少
        sample_0 = cnt_0
        sample_1 = int(pos_ratio / neg_ratio * sample_0)

    # 从df中抽取sample_0个neg, sample_1个pos
    df_0 = df_0.sample(n=sample_0)
    df_1 = df_1.sample(n=sample_1)

    return df_0, df_1
