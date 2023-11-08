import pandas as pd
import re

df = pd.read_csv('data/users_grouped_posts_and_comments.csv')

# remove \\n, \' and http links
df['text_x'] = df['text_x'].apply(lambda x: re.sub(r'\\n', ' ', x)) 
df['text_x'] = df['text_x'].apply(lambda x: re.sub(r'\'', '', x))
df['text_x'] = df['text_x'].apply(lambda x: re.sub(r'http\S+', '', x))

df['text_y'] = df['text_y'].apply(lambda x: re.sub(r'\\n', ' ', x)) 
df['text_y'] = df['text_y'].apply(lambda x: re.sub(r'\'', '', x))
df['text_y'] = df['text_y'].apply(lambda x: re.sub(r'http\S+', '', x))

# cast each entry to a list and unnest to a list of words and remove non alphanumeric characters
df['text_x'] = df['text_x'].apply(lambda x: x.split())
df['text_x'] = df['text_x'].apply(lambda x: [re.sub(r'\W+', '', word) for word in x])

df['text_y'] = df['text_y'].apply(lambda x: x.split())
df['text_y'] = df['text_y'].apply(lambda x: [re.sub(r'\W+', '', word) for word in x])

# replace nan with empty string
df.fillna('', inplace=True)

# join text_x and text_y
df['text'] = df['text_x'] + df['text_y']

#rename labels_x to labels, cast to int and replace nan with -1
df.rename(columns={'labels_x': 'labels'}, inplace=True)
df['labels'] = df['labels'].astype(int)
df['labels'].fillna(-1, inplace=True)

users_text = df[['_id', 'text', 'labels']]
users_text.to_csv('data/users_text.csv', index=False)

# create a list with all the words and their counts
words = []
for text in df['text'].values():
    words.extend(text)

from collections import Counter

word_counts = Counter(words)

tokens = word_counts.keys()

counts = word_counts.values()

relative_frequencies = [word_counts[token]/len(words) for token in tokens]

num_unique_words = len(word_counts)

zipfian_df = pd.DataFrame({'types': tokens, 'counts' : counts, 'probs': relative_frequencies, 'totalunique' : num_unique_words})

zipfian_df.to_csv('data/full_zipfian_df.csv', index=False)

# enumerate the labels and create zipfian dataframes for each label
labels = df['labels'].unique()

for label in labels:
    df_label = df[df['labels'] == label]
    words = []
    for text in df_label['text'].values():
        words.extend(text)
    word_counts = Counter(words)
    tokens = word_counts.keys()
    counts = word_counts.values()
    relative_frequencies = [word_counts[token]/len(words) for token in tokens]
    num_unique_words = len(word_counts)
    zipfian_df = pd.DataFrame({'types': tokens, 'counts' : counts, 'probs': relative_frequencies, 'totalunique' : num_unique_words})
    zipfian_df.to_csv('data/zipfian_df_' + str(label) + '.csv', index=False)