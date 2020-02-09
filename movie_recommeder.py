import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Helper functions
def get_title_from_index(index):
    return df[df.index == index]['title'].values[0]
def get_index_from_title(title):
    return df[df.title == title]['index'].values[0]

# 1. Read csv file
df = pd.read_csv('movie_dataset.csv')

# 2. Select features
features = ['keywords', 'cast', 'genres', 'director']

# 3. Create a column in df which combines all selected features
def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']

# 4. Clean and preprocess the data
for feature in features:
    df[feature] = df[feature].fillna('')

# 5. Get a new combined column
df['combnined_features'] = df.apply(combine_features, axis=1)

# 6. Create a count matrix for new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combnined_features'])

# 7. Compute the cosine similarity based on the count matrix
similarity_scores = cosine_similarity(count_matrix)
# print(similarity_scores)

# 8. Movie user likes: Get title of the movie
movie_user_likes = 'Avatar'

# 9. Get index of the movie the user likes
movie_index = get_index_from_title(movie_user_likes)

# 10. Get a list of similar movies in descending order of similarity score
movie_row = similarity_scores[movie_index]
similar_movies = list(enumerate(movie_row)) # (index, similarity score)
sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)[1:]
# print(sorted_similar_movies)

# 11. Print titles of first 5 movies
print('\nYou liked: ' + movie_user_likes)
print('\nTop 5 recommended movies for you: \n')
i = 0
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i = i + 1
    if i > 5:
        break
