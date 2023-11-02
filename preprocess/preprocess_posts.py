import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the data
posts = pd.read_json("babynamesDB_posts.json")
posts = posts[["author", "text"]]

# Group by author and aggregate text
temp = posts.groupby('author')['text'].apply(list)

# Flatten the list of texts and keep track of authors
texts = []
authors = []
for author, texts_list in temp.iteritems():
    texts.extend(texts_list)
    authors.extend([author] * len(texts_list))

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the texts
encoded_text = model.encode(texts, show_progress_bar=True)

# Combine authors and encoded texts into a DataFrame
encoded_df = pd.DataFrame(encoded_text)
encoded_df['author'] = authors

# Save the result
encoded_df.to_csv("data/encoded_text.csv", index=False)
