import numpy as np
def get_setence_Xy(df, emb_name):
    X = [list(x) for x in df[emb_name].values]
    y = df['label'].values
    return X, y


def get_account_Xy(df, emb_name):
    X, y = [], []
    df_groups = df.groupby('user_id')
    for key, this_df in df_groups:
        embeddings = [list(x) for x in this_df[emb_name].values]
        embedding = np.mean(embeddings, axis=0)
        X.append(list(embedding))
        y.append(this_df['label'].values[0])
    return X, y