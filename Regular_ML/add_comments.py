import pandas as pd
import numpy as np

labels = pd.read_csv("data/users_labels.csv")

comments = pd.read_json("babynamesDB_comments.json")
comments = comments[["author", "text"]]

combined_df = comments.merge(labels, how='left', left_on='author' , right_on='_id')

combined_df = combined_df.drop('author', axis='columns')

grouped_df = combined_df.groupby('_id').agg({'text': lambda x: list(x), 'labels': 'first'}).reset_index()

df_with_comments = grouped_df.merge(labels, on='_id', how='right')
df_with_comments.drop( columns='labels_x')
df_with_comments.fillna('')

df_with_posts = pd.read_csv("data/users_grouped_posts.csv")

#join the two dataframes

df_with_posts_and_comments = df_with_posts.merge(df_with_comments, on='_id', how='left')

df_with_posts_and_comments.to_csv("data/users_grouped_posts_and_comments.csv", index=False)