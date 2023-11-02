import pandas as pd

def composite(s, d):
  return (1/d)*s

best = 0
best_df = None
best_params = None


for i in [3,4,5]:
  for j in range(3,11):
    for dim_red in ['pca', 'umap']:
      for clust in ['kmeans', 'hdbscan']:
        df = pd.read_csv(f'data/{i}_{dim_red}_{j}_{clust}.csv')

        s = df['silhouette'][0]
        d = df['db'][0]
        c = df['ch'][0]

        comp = composite(s, d)
        print(f'{i}_{dim_red}_{j}_{clust}: {comp}')

        if comp > best:
          best = comp
          best_df = df
          best_params = [i, dim_red, j, clust]

print(best)
print(best_params)
best_df.to_csv(f'data/best.csv')