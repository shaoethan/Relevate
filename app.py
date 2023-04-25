from flask import Flask, render_template
# Import the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Define the route for the list of movies
@app.route('/movies')
def movies():
    # Load the dataset
    df = pd.read_csv('movies.csv')
    
    # Define the CountVectorizer object
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['description'])
    
    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(count_matrix)
    
    # Get the index of the movie that the user likes
    movie_index = df[df['title'] == 'The Dark Knight'].index[0]
    
    # Get the list of similar movies and their indices
    sim_movies = list(enumerate(cosine_sim[movie_index]))
    sim_movies = sorted(sim_movies, key=lambda x: x[1], reverse=True)
    sim_movies = sim_movies[1:11]
    movie_indices = [i[0] for i in sim_movies]
    
    # Get the titles of the similar movies
    similar_movies = df.iloc[movie_indices]['title'].tolist()
    
    # Render the HTML template and pass the list of similar movies to it
    return render_template('movies.html', movies=similar_movies)

if __name__ == '__main__':
    app.run(debug=True)