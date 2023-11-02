import pandas as pd
from sklearn.preprocessing import StandardScaler

# load in data
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


#load in post embeddings

posts = pd.read_csv("data/encoded_text.csv")

# group by author and aggregate embeddings
posts = posts.groupby("author").mean()

# add empty embeddings for users with no posts in the posts df
users = pd.merge(users, posts, left_on='_id', right_index=True, how='left')
users.fillna(0, inplace=True)

print("post embeddings successfully loaded")

# write out data
users.to_csv("data/users_embeddings.csv")
