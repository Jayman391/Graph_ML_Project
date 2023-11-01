import pandas as pd
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

# 1. Load all data upfront
groups = pd.read_json("babynamesDB_groups.json")
# filter out groups with less than 3 users
groups = groups[groups["num_users_stored"] > 3]
group_ids = groups["_id"].to_list()

print("group data successfully loaded")

users = pd.read_json("babynamesDB_users.json")
users = users[["_id", "num_comments_stored", "groups", "num_posts_stored"]]
# unnest groups
users = users.explode("groups")
users = users[users["groups"].isin(group_ids)]
# 1 hot encode groups
users = pd.concat([users, pd.get_dummies(users["groups"], dtype=float)], axis=1)
# join data on user id
users = users.groupby("_id").sum()
# users = users.drop(columns=['groups'])

print("user data successfully loaded")

posts = pd.read_json("babynamesDB_posts.json")

posts = posts[["author", "text"]]

temp = posts.groupby('author')['text'].apply(list)

#add posts to users by post author and user username
users = users.join(temp, on='_id')

# replace NaN with empty list
users['text'] = users['text'].fillna('')

print('posts successfully added to users')

# 7. Encode Text Data
model = SentenceTransformer('all-MiniLM-L6-v2')

# Assuming that the text data is a list of strings for each user
user_text_embeddings = users['text'].apply(lambda texts: model.encode(texts))


print('posts successfully encoded')
# add embeddings to users with each embedding as a column

# 7.1. Create a DataFrame from the embeddings
user_text_embeddings = pd.DataFrame(user_text_embeddings.tolist())

# 7.2. Join the embeddings to the users DataFrame
users = users.join(user_text_embeddings)

print(f'Users embedding shape: {users.shape}')

# The users_scaled DataFrame now contains the scaled numerical data and the encoded text data

# Saving the scaled data
users.to_csv('data/users_scaled.csv')
