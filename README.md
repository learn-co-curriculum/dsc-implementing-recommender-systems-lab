
# Implementing Recommender Systems - Lab

## Introduction

In this lab, you'll practice creating a recommender system model using surprise. You'll also get the chance to create a more complete recommender system pipeline to obtain the top recommendations for a specific user.


## Objectives
You will be able to:
* Fit a recommender system model to a set of data
* Create a function that will return the top recommendations for a user
* Introduce a new user to a rating matrix and make recommendations for them

For this lab, we will be using the famous 1M movie dataset. It contains a collection of user ratings for many different movies. In the last  lesson, you got exposed to working with Surprise datasets. In this lab, you will also go through the process of reading in a dataset into the Surprise dataset format. To begin with, load the dataset into a pandas dataframe. Determine which columns are necessary for your recommendation system and drop any extraneous ones.


```python
import pandas as pd
df = pd.read_csv('./ml-latest-small/ratings.csv')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100836 entries, 0 to 100835
    Data columns (total 4 columns):
    userId       100836 non-null int64
    movieId      100836 non-null int64
    rating       100836 non-null float64
    timestamp    100836 non-null int64
    dtypes: float64(1), int64(3)
    memory usage: 3.1 MB



```python
#drop unnecessary columns
new_df=None
```

It's now time to transform the dataset into something compatible with Surprise. In order to do this, you're going to need `Reader` and `Dataset` classes. There's a method in `Dataset` specifically for loading dataframes.


```python
from surprise import Reader, Dataset
# read in values as Surprise dataset 


```

Let's look at how many users and items we have in our dataset. If using neighborhood-based methods, this will help us determine whether or not we should perform user-user or item-item similarity


```python
dataset = data.build_full_trainset()
print('Number of users: ',dataset.n_users,'\n')
print('Number of items: ',dataset.n_items)
```

    Number of users:  610 
    
    Number of items:  9724


## Determine the Best Model
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

```


```python
# print out optimal parameters for SVD after GridSearch
```

    {'rmse': 0.8689250510051669, 'mae': 0.6679404366294037}
    {'rmse': {'n_factors': 50, 'reg_all': 0.05}, 'mae': {'n_factors': 100, 'reg_all': 0.05}}



```python
# cross validating with KNNBasic

```


```python
# print out the average RMSE score for the test set
```

    ('test_rmse', array([0.97646619, 0.97270627, 0.97874535, 0.97029184, 0.96776748]))
    ('test_mae', array([0.75444119, 0.75251222, 0.7531242 , 0.74938542, 0.75152129]))
    ('fit_time', (0.46678805351257324, 0.54010009765625, 0.7059998512268066, 0.5852491855621338, 1.0139541625976562))
    ('test_time', (2.308177947998047, 2.4834508895874023, 2.6563329696655273, 2.652374029159546, 1.2219891548156738))
    -----------------------
    0.9731954260849399



```python
# cross validating with KNNBaseline

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
# print out the average score for the test set

```

    ('test_rmse', array([0.87268017, 0.88765352, 0.87311917, 0.88706914, 0.87043399]))
    ('test_mae', array([0.66796685, 0.676203  , 0.66790869, 0.67904038, 0.66459155]))
    ('fit_time', (0.6972200870513916, 0.7296440601348877, 0.5842609405517578, 0.609612226486206, 0.61130690574646))
    ('test_time', (1.5466029644012451, 1.567044973373413, 1.6441452503204346, 1.5709199905395508, 1.6216418743133545))





    0.8781911983703239



Based off these outputs, it seems like the best performing model is the SVD model with n_factors = 50 and a regularization rate of 0.05. Let's use that model to make some predictions. Use that model or if you found one that performs better, feel free to use that.

## Making Recommendations

This next section is going to involve making recommendations, and it's important that the output for the recommendation is interpretable to people. Rather than returning the movie_id values, it would be far more valuable to return the actual title of the movie. As a first step, let's read in the movies to a dataframe and take a peek at what information we have about them.


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




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x11952ab38>




```python
svd.predict(2,4)
```




    Prediction(uid=2, iid=4, r_ui=None, est=3.0129484092252135, details={'was_impossible': False})



This prediction value is a tuple and each of the values within it can be accessed by way of indexing. Now let's put all of our knowledge of recommendation systems to do something interesting: making predictions for a new user!

## Obtaining User Ratings 

It's great that we have working models and everything, but wouldn't it be nice to get to recommendations specifically tailored to your preferences? That's what we'll be doing now. The first step to go let's create a function that allows students to pick randomly selected movies. The function should present users with a movie and ask them to rate it. If they have not seen the movie, they should be able to skip rating it. 

The function `movie_rater` should take as parameters:
* movie_df : DataFrame - a dataframe containing the movie ids, name of movie, and genres
* num : int - number of ratings
* genre : string - a specific genre from which to draw movies

The function returns:
* rating_list : list - a collection of dictionaries in the format of {'userId': int  , 'movieId': int  ,'rating': float  }

#### This function is optional, but fun :) 


```python
def movie_rater(movie_df,num, genre=None):
    pass
        
```


```python
# try out the new function here!
```

          movieId                   title          genres
    6579    55245  Good Luck Chuck (2007)  Comedy|Romance
    How do you rate this movie on a scale of 1-5, press n if you have not seen :
    5
          movieId                       title          genres
    1873     2491  Simply Irresistible (1999)  Comedy|Romance
    How do you rate this movie on a scale of 1-5, press n if you have not seen :
    4
          movieId                  title  genres
    3459     4718  American Pie 2 (2001)  Comedy
    How do you rate this movie on a scale of 1-5, press n if you have not seen :
    4
          movieId             title                   genres
    4160     5990  Pinocchio (2002)  Children|Comedy|Fantasy
    How do you rate this movie on a scale of 1-5, press n if you have not seen :
    3


If you're struggling to come up with the above function, you can use this list of user ratings to complete the next segment


```python
user_rating
```




    [{'userId': 1000, 'movieId': 55245, 'rating': '5'},
     {'userId': 1000, 'movieId': 2491, 'rating': '4'},
     {'userId': 1000, 'movieId': 4718, 'rating': '4'},
     {'userId': 1000, 'movieId': 5990, 'rating': '3'}]



### Making Predictions With the New Ratings
Now that you have new ratings, you can use them to make predictions for this new user. The proper way this should work is:

* add the new ratings to the original ratings DataFrame, read into a Surprise dataset
* train a model using the new combined DataFrame
* make predictions for the user
* order those predictions from highest rated to lowest rated
* return the top n recommendations with the text of the actual movie (rather than just the index number)


```python
## add the new ratings to the original ratings DataFrame

```


```python
# train a model using the new combined DataFrame

```




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x11daeb898>




```python
# make predictions for the user
# you'll probably want to create a list of tuples in the format (movie_id, predicted_score)

```


```python
# order the predictions from highest to lowest rated

ranked_movies = None
```

 For the final component of this challenge, it could be useful to create a function `recommended_movies` that takes in the parameters:
* `user_ratings` : list - list of tuples formulated as (user_id, movie_id) (should be in order of best to worst for this individual)
* `movie_title_df` : DataFrame 
* `n` : int- number of recommended movies 

The function should use a for loop to print out each recommended *n* movies in order from best to worst


```python
# return the top n recommendations using the 
def recommended_movies(user_ratings,movie_title_df,n):
        pass
            
recommended_movies(ranked_movies,df_movies,5)
```

    Recommendation #  1 :  277    Shawshank Redemption, The (1994)
    Name: title, dtype: object 
    
    Recommendation #  2 :  680    Philadelphia Story, The (1940)
    Name: title, dtype: object 
    
    Recommendation #  3 :  686    Rear Window (1954)
    Name: title, dtype: object 
    
    Recommendation #  4 :  602    Dr. Strangelove or: How I Learned to Stop Worr...
    Name: title, dtype: object 
    
    Recommendation #  5 :  926    Amadeus (1984)
    Name: title, dtype: object 
    


## Level Up

* Try and chain all of the steps together into one function that asks users for ratings for a certain number of movies, then all of the above steps are performed to return the top n recommendations
* Make a recommender system that only returns items that come from a specified genre

## Summary

In this lab, you got the change to implement a collaborative filtering model as well as retrieve recommendations from that model. You also got the opportunity to add your own recommendations to the system to get new recommendations for yourself! Next, you will get exposed to using spark to make recommender systems.
