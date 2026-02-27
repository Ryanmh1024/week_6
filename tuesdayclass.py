#%%
import pandas as pd
import numpy as np
import sklearn as sk
#%%

# Loading Data
# Merging Data
# sklearn four steps (review documentation)
# lambda functions

#%%
# load the data, column titles in salary in first row
salary_data = pd.read_csv('2025_salaries.csv', header=1, encoding="latin-1")
stats = pd.read_csv('nba_2025.txt', sep=",", encoding="latin-1")

#%%
# need to merge the data by player name
df_merged = pd.merge(salary_data, stats, on="Player")

# %%
# checking how many duplicates there are
# df[df.function] - df.function will create a True and False matrix and passing that
# to df will only return the True values of df from df.function
duplicates = df_merged[df_merged.duplicated(subset="Player", keep=False)]
# %%
# Sklearn four steps
# 1. Create an instance of the model - Example: mymodel = KMeans(n_clusters=3)
# 2. Fit the model to the data example - Example: mymodel.fit(X)
# 3. Make predictions using the model - Example: predictions = mymodel.predict(X)
# 4. Evaluate the model's performance - Example: score = mymodel.score(X)

# for kmeans you don't need to predict, you can just use the labels_ attribute
# to get the cluster assignments for each data point after fitting the model.

# if did it right, low salary should be bottom right and high salary should be top right
# and salary should be a heatmap -- however the x and y values of the axis need to be
# features of the dataset that hopefully show the salary correctly. (goal is positive correlation)

# Possible x and y features: Games played, points per game, total_rebounds

# Data points should have shapes for each cluster and then color of points should be the salary
#%%
# To solve the problem of duplicate player names:
# Could write a function that looks at duplicates and keeps the row
# for the player with the highest number of games played.