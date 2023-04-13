# Implementing Recommender Systems - Lab

## Introduction

In this lab, you'll practice creating a recommender system model using `surprise`. You'll also get the chance to create a more complete recommender system pipeline to obtain the top recommendations for a specific user.


## Objectives

In this lab you will: 

- Use surprise's built-in reader class to process data to work with recommender algorithms 
- Obtain a prediction for a specific user for a particular item 
- Introduce a new user with rating to a rating matrix and make recommendations for them 
- Create a function that will return the top n recommendations for a user 


For this lab, we will be using the famous 1M movie dataset. It contains a collection of user ratings for many different movies. In the last lesson, you were exposed to working with `surprise` datasets. In this lab, you will also go through the process of reading in a dataset into the `surprise` dataset format. To begin with, load the dataset into a Pandas DataFrame. Determine which columns are necessary for your recommendation system and drop any extraneous ones.

To complete this lab, you will need to install the library **Surprise**. 

**Uncomment and run the cell below only once to install Surprise to learn-env.**


```python
# !pip install surprise
```


```python
import pandas as pd
df = pd.read_csv('./ml-latest-small/ratings.csv')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100836 entries, 0 to 100835
    Data columns (total 4 columns):
     #   Column     Non-Null Count   Dtype  
    ---  ------     --------------   -----  
     0   userId     100836 non-null  int64  
     1   movieId    100836 non-null  int64  
     2   rating     100836 non-null  float64
     3   timestamp  100836 non-null  int64  
    dtypes: float64(1), int64(3)
    memory usage: 3.1 MB



```python
# Drop unnecessary columns
new_df = df.drop(columns='timestamp')
```

It's now time to transform the dataset into something compatible with `surprise`. In order to do this, you're going to need `Reader` and `Dataset` classes. There's a method in `Dataset` specifically for loading dataframes.


```python
from surprise import Reader, Dataset
reader = Reader()
data = Dataset.load_from_df(new_df,reader)
```

Let's look at how many users and items we have in our dataset. If using neighborhood-based methods, this will help us determine whether or not we should perform user-user or item-item similarity


```python
dataset = data.build_full_trainset()
print('Number of users: ', dataset.n_users, '\n')
print('Number of items: ', dataset.n_items)
```

    Number of users:  610 
    
    Number of items:  9724


## Determine the best model 

Now, compare the different models and see which ones perform best. For consistency sake, use RMSE to evaluate models. Remember to cross-validate! Can you get a model with a higher average RMSE on test data than 0.869?


```python
# importing relevant libraries
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import SVD
from surprise.prediction_algorithms import KNNWithMeans, KNNBasic, KNNBaseline
from surprise.model_selection import GridSearchCV
import numpy as np
```


```python
## Perform a gridsearch with SVD
# ⏰ This cell may take several minutes to run
params = {'n_factors': [20, 50, 100],
         'reg_all': [0.02, 0.05, 0.1]}
g_s_svd = GridSearchCV(SVD,param_grid=params,n_jobs=-1)
g_s_svd.fit(data)
```


```python
print(g_s_svd.best_score)
print(g_s_svd.best_params)
```

    {'rmse': 0.868102914598724, 'mae': 0.6676775766541562}
    {'rmse': {'n_factors': 100, 'reg_all': 0.05}, 'mae': {'n_factors': 100, 'reg_all': 0.05}}



```python
# cross validating with KNNBasic
knn_basic = KNNBasic(sim_options={'name':'pearson', 'user_based':True})
cv_knn_basic = cross_validate(knn_basic, data, n_jobs=-1)
```


```python
for i in cv_knn_basic.items():
    print(i)
print('-----------------------')
print(np.mean(cv_knn_basic['test_rmse']))
```

    ('test_rmse', array([0.97369003, 0.97375821, 0.9745871 , 0.96542379, 0.97288311]))
    ('test_mae', array([0.74898092, 0.753142  , 0.75212946, 0.74723532, 0.7495044 ]))
    ('fit_time', (0.060803890228271484, 0.06330013275146484, 0.0663609504699707, 0.060571908950805664, 0.05236673355102539))
    ('test_time', (0.44602489471435547, 0.4508967399597168, 0.4301769733428955, 0.44738316535949707, 0.42812514305114746))
    -----------------------
    0.9720684474044512



```python
# cross validating with KNNBaseline
knn_baseline = KNNBaseline(sim_options={'name':'pearson', 'user_based':True})
cv_knn_baseline = cross_validate(knn_baseline,data)
```

    Estimating biases using als...
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Estimating biases using als...
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Estimating biases using als...
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Estimating biases using als...
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Estimating biases using als...
    Computing the pearson similarity matrix...
    Done computing similarity matrix.



```python
for i in cv_knn_baseline.items():
    print(i)

np.mean(cv_knn_baseline['test_rmse'])
```

    ('test_rmse', array([0.87594063, 0.8744408 , 0.88114275, 0.87173134, 0.88099762]))
    ('test_mae', array([0.67057651, 0.66737735, 0.67067535, 0.6677891 , 0.67128841]))
    ('fit_time', (0.09939002990722656, 0.10296893119812012, 0.1037588119506836, 0.10332417488098145, 0.11193490028381348))
    ('test_time', (0.5622682571411133, 0.5735628604888916, 0.5679421424865723, 0.6012997627258301, 0.6014230251312256))





    0.876850626575931



Based off these outputs, it seems like the best performing model is the SVD model with `n_factors = 50` and a regularization rate of 0.05. Use that model or if you found one that performs better, feel free to use that to make some predictions.

## Making Recommendations

It's important that the output for the recommendation is interpretable to people. Rather than returning the `movie_id` values, it would be far more valuable to return the actual title of the movie. As a first step, let's read in the movies to a dataframe and take a peek at what information we have about them.


```python
df_movies = pd.read_csv('./ml-latest-small/movies.csv')
```


```python
df_movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



## Making simple predictions
Just as a reminder, let's look at how you make a prediction for an individual user and item. First, we'll fit the SVD model we had from before.


```python
svd = SVD(n_factors= 50, reg_all=0.05)
svd.fit(dataset)
```




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x1463ef3a0>




```python
svd.predict(2, 4)
```




    Prediction(uid=2, iid=4, r_ui=None, est=3.1441023814846094, details={'was_impossible': False})



This prediction value is a tuple and each of the values within it can be accessed by way of indexing. Now let's put our knowledge of recommendation systems to do something interesting: making predictions for a new user!

## Obtaining User Ratings 

It's great that we have working models and everything, but wouldn't it be nice to get to recommendations specifically tailored to your preferences? That's what we'll be doing now. The first step is to create a function that allows us to pick randomly selected movies. The function should present users with a movie and ask them to rate it. If they have not seen the movie, they should be able to skip rating it. 

The function `movie_rater()` should take as parameters: 

* `movie_df`: DataFrame - a dataframe containing the movie ids, name of movie, and genres
* `num`: int - number of ratings
* `genre`: string - a specific genre from which to draw movies

The function returns:
* rating_list : list - a collection of dictionaries in the format of {'userId': int , 'movieId': int , 'rating': float}

#### This function is optional, but fun :) 


```python
def movie_rater(movie_df,num, genre=None):
    userID = 1000
    rating_list = []
    while num > 0:
        if genre:
            movie = movie_df[movie_df['genres'].str.contains(genre)].sample(1)
        else:
            movie = movie_df.sample(1)
        print(movie)
        rating = input('How do you rate this movie on a scale of 1-5, press n if you have not seen :\n')
        if rating == 'n':
            continue
        else:
            rating_one_movie = {'userId':userID,'movieId':movie['movieId'].values[0],'rating':rating}
            rating_list.append(rating_one_movie) 
            num -= 1
    return rating_list
```


```python
user_rating = movie_rater(df_movies, 4, 'Comedy')
```

          movieId          title                    genres
    1551     2088  Popeye (1980)  Adventure|Comedy|Musical
    How do you rate this movie on a scale of 1-5, press n if you have not seen :
    1
          movieId                        title          genres
    3535     4831  Can't Stop the Music (1980)  Comedy|Musical
    How do you rate this movie on a scale of 1-5, press n if you have not seen :
    2
          movieId                      title                  genres
    3480     4749  3 Ninjas Kick Back (1994)  Action|Children|Comedy
    How do you rate this movie on a scale of 1-5, press n if you have not seen :
    3
          movieId                  title                  genres
    4923     7380  Ella Enchanted (2004)  Comedy|Fantasy|Romance
    How do you rate this movie on a scale of 1-5, press n if you have not seen :
    4


If you're struggling to come up with the above function, you can use this list of user ratings to complete the next segment


```python
user_rating
```




    [{'userId': 1000, 'movieId': 2088, 'rating': '1'},
     {'userId': 1000, 'movieId': 4831, 'rating': '2'},
     {'userId': 1000, 'movieId': 4749, 'rating': '3'},
     {'userId': 1000, 'movieId': 7380, 'rating': '4'}]



### Making Predictions With the New Ratings
Now that you have new ratings, you can use them to make predictions for this new user. The proper way this should work is:

* add the new ratings to the original ratings DataFrame, read into a `surprise` dataset 
* train a model using the new combined DataFrame
* make predictions for the user
* order those predictions from highest rated to lowest rated
* return the top n recommendations with the text of the actual movie (rather than just the index number) 


```python
## add the new ratings to the original ratings DataFrame
user_ratings = pd.DataFrame(user_rating)
new_ratings_df = pd.concat([new_df, user_ratings], axis=0)
new_data = Dataset.load_from_df(new_ratings_df,reader)
```


```python
# train a model using the new combined DataFrame
svd_ = SVD(n_factors= 50, reg_all=0.05)
svd_.fit(new_data.build_full_trainset())
```




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x1463ef820>




```python
# make predictions for the user
# you'll probably want to create a list of tuples in the format (movie_id, predicted_score)
list_of_movies = []
for m_id in new_df['movieId'].unique():
    list_of_movies.append( (m_id,svd_.predict(1000,m_id)[3]))
```


```python
# order the predictions from highest to lowest rated
ranked_movies = sorted(list_of_movies, key=lambda x:x[1], reverse=True)
```

 For the final component of this challenge, it could be useful to create a function `recommended_movies()` that takes in the parameters:
* `user_ratings`: list - list of tuples formulated as (user_id, movie_id) (should be in order of best to worst for this individual)
* `movie_title_df`: DataFrame 
* `n`: int - number of recommended movies 

The function should use a `for` loop to print out each recommended *n* movies in order from best to worst


```python
# return the top n recommendations using the 
def recommended_movies(user_ratings,movie_title_df,n):
        for idx, rec in enumerate(user_ratings):
            title = movie_title_df.loc[movie_title_df['movieId'] == int(rec[0])]['title']
            print('Recommendation # ', idx+1, ': ', title, '\n')
            n-= 1
            if n == 0:
                break
            
recommended_movies(ranked_movies,df_movies,5)
```

    Recommendation #  1 :  277    Shawshank Redemption, The (1994)
    Name: title, dtype: object 
    
    Recommendation #  2 :  602    Dr. Strangelove or: How I Learned to Stop Worr...
    Name: title, dtype: object 
    
    Recommendation #  3 :  686    Rear Window (1954)
    Name: title, dtype: object 
    
    Recommendation #  4 :  585    Wallace & Gromit: The Best of Aardman Animatio...
    Name: title, dtype: object 
    
    Recommendation #  5 :  659    Godfather, The (1972)
    Name: title, dtype: object 
    


## Level Up (Optional)

* Try and chain all of the steps together into one function that asks users for ratings for a certain number of movies, then all of the above steps are performed to return the top $n$ recommendations
* Make a recommender system that only returns items that come from a specified genre

## Summary

In this lab, you got the chance to implement a collaborative filtering model as well as retrieve recommendations from that model. You also got the opportunity to add your own recommendations to the system to get new recommendations for yourself! Next, you will learn how to use Spark to make recommender systems.
