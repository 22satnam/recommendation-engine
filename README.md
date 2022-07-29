
"# recommendation-engine" 
=======
The details of the movies(title, genre, runtime, rating, poster, etc) are fetched using an API by TMDB and using the IMDB id of the movie in the API, We did web scraping to get the reviews given by the user in the IMDB site using beautifulsoup4 and performed sentiment analysis on those reviews.


How to run the project?

1. Clone or download this repository to your local machine.
2. Download all data sets given below(for proper work flow).
4. Install all the libraries mentioned in the [requirements.txt](https://github.com/22satnam/recommendation-engine) file with the command `pip install -r requirements.txt`
5. Get your API key from https://www.themoviedb.org/.
6. Open your terminal/command prompt from your project directory and run the file `application.py` by executing the command `python application.py`.
7. Go to your browser and type `http://127.0.0.1:5000/` in the address bar.


How Cosine Similarity works?
Cosine similarity is a metric used to measure how similar the documents are irrespective of their size. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance (due to the size of the document), chances are they may still be oriented closer together. The smaller the angle, higher the cosine similarity.

How Content based Recommendation Engine works?

During recommendation, the similarity metrics (We will talk about it in a bit) are calculated from the item’s feature vectors and the user’s preferred feature vectors from his/her previous records. Then, the top few are recommended.
Content-based filtering does not require other users' data during recommendations to one user.
https://towardsdatascience.com/introduction-to-recommender-systems-1-971bd274f421


Sources of the datasets 

1. [IMDB 5000 Movie Dataset](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset)
2. [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset)

