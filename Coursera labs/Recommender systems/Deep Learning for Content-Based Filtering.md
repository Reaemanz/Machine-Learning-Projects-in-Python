# <img align="left" src="./images/film_strip_vertical.png"     style=" width:40px;  " > Practice lab: Deep Learning for Content-Based Filtering

In this exercise, you will implement content-based filtering using a neural network to build a recommender system for movies. 


# Outline
- [ 1 - Packages ](#1)
- [ 2 - Movie ratings dataset ](#2)
- [ 3 - Content-based filtering with a neural network](#3)
  - [ 3.1 Training Data](#3.1)
  - [ 3.2 Preparing the training data](#3.2)
- [ 4 - Neural Network for content-based filtering](#4)
  - [ Exercise 1](#ex01)
- [ 5 - Predictions](#5)
  - [ 5.1 - Predictions for a new user](#5.1)
  - [ 5.2 - Predictions for an existing user.](#5.2)
  - [ 5.3 - Finding Similar Items](#5.3)
    - [ Exercise 2](#ex02)
- [ 6 - Congratulations! ](#6)


_**NOTE:** To prevent errors from the autograder, you are not allowed to edit or delete non-graded cells in this lab. Please also refrain from adding any new cells. 
**Once you have passed this assignment** and want to experiment with any of the non-graded code, you may follow the instructions at the bottom of this notebook._

<a name="1"></a>
## 1 - Packages <img align="left" src="./images/movie_camera.png"     style=" width:40px;  ">
We will use familiar packages, NumPy, TensorFlow and helpful routines from [scikit-learn](https://scikit-learn.org/stable/). We will also use [tabulate](https://pypi.org/project/tabulate/) to neatly print tables and [Pandas](https://pandas.pydata.org/) to organize tabular data.


```python
import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
from recsysNN_utils import *
pd.set_option("display.precision", 1)
```

<a name="2"></a>
## 2 - Movie ratings dataset <img align="left" src="./images/film_rating.png" style=" width:40px;" >
The data set is derived from the [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/latest/) dataset. 

[F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>]

The original dataset has roughly 9000 movies rated by 600 users with ratings on a scale of 0.5 to 5 in 0.5 step increments. The dataset has been reduced in size to focus on movies from the years since 2000 and popular genres. The reduced dataset has $n_u = 397$ users, $n_m= 847$ movies and 25521 ratings. For each movie, the dataset provides a movie title, release date, and one or more genres. For example "Toy Story 3" was released in 2010 and has several genres: "Adventure|Animation|Children|Comedy|Fantasy". This dataset contains little information about users other than their ratings. This dataset is used to create training vectors for the neural networks described below. 
Let's learn a bit more about this data set. The table below shows the top 10 movies ranked by the number of ratings. These movies also happen to have high average ratings. How many of these movies have you watched? 


```python
top10_df = pd.read_csv("./data/content_top10_df.csv")
bygenre_df = pd.read_csv("./data/content_bygenre_df.csv")
top10_df
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
      <th>movie id</th>
      <th>num ratings</th>
      <th>ave rating</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4993</td>
      <td>198</td>
      <td>4.1</td>
      <td>Lord of the Rings: The Fellowship of the Ring,...</td>
      <td>Adventure|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5952</td>
      <td>188</td>
      <td>4.0</td>
      <td>Lord of the Rings: The Two Towers, The</td>
      <td>Adventure|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7153</td>
      <td>185</td>
      <td>4.1</td>
      <td>Lord of the Rings: The Return of the King, The</td>
      <td>Action|Adventure|Drama|Fantasy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4306</td>
      <td>170</td>
      <td>3.9</td>
      <td>Shrek</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy|Ro...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>58559</td>
      <td>149</td>
      <td>4.2</td>
      <td>Dark Knight, The</td>
      <td>Action|Crime|Drama</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6539</td>
      <td>149</td>
      <td>3.8</td>
      <td>Pirates of the Caribbean: The Curse of the Bla...</td>
      <td>Action|Adventure|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>6</th>
      <td>79132</td>
      <td>143</td>
      <td>4.1</td>
      <td>Inception</td>
      <td>Action|Crime|Drama|Mystery|Sci-Fi|Thriller</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6377</td>
      <td>141</td>
      <td>4.0</td>
      <td>Finding Nemo</td>
      <td>Adventure|Animation|Children|Comedy</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4886</td>
      <td>132</td>
      <td>3.9</td>
      <td>Monsters, Inc.</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7361</td>
      <td>131</td>
      <td>4.2</td>
      <td>Eternal Sunshine of the Spotless Mind</td>
      <td>Drama|Romance|Sci-Fi</td>
    </tr>
  </tbody>
</table>
</div>



The next table shows information sorted by genre. The number of ratings per genre vary substantially. Note that a movie may have multiple genre's so the sum of the ratings below is larger than the number of original ratings.


```python
bygenre_df
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
      <th>genre</th>
      <th>num movies</th>
      <th>ave rating/genre</th>
      <th>ratings per genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Action</td>
      <td>321</td>
      <td>3.4</td>
      <td>10377</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adventure</td>
      <td>234</td>
      <td>3.4</td>
      <td>8785</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Animation</td>
      <td>76</td>
      <td>3.6</td>
      <td>2588</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Children</td>
      <td>69</td>
      <td>3.4</td>
      <td>2472</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Comedy</td>
      <td>326</td>
      <td>3.4</td>
      <td>8911</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Crime</td>
      <td>139</td>
      <td>3.5</td>
      <td>4671</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Documentary</td>
      <td>13</td>
      <td>3.8</td>
      <td>280</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Drama</td>
      <td>342</td>
      <td>3.6</td>
      <td>10201</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Fantasy</td>
      <td>124</td>
      <td>3.4</td>
      <td>4468</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Horror</td>
      <td>56</td>
      <td>3.2</td>
      <td>1345</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Mystery</td>
      <td>68</td>
      <td>3.6</td>
      <td>2497</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Romance</td>
      <td>151</td>
      <td>3.4</td>
      <td>4468</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Sci-Fi</td>
      <td>174</td>
      <td>3.4</td>
      <td>5894</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Thriller</td>
      <td>245</td>
      <td>3.4</td>
      <td>7659</td>
    </tr>
  </tbody>
</table>
</div>



<a name="3"></a>
## 3 - Content-based filtering with a neural network

In the collaborative filtering lab, you generated two vectors, a user vector and an item/movie vector whose dot product would predict a rating. The vectors were derived solely from the ratings.   

Content-based filtering also generates a user and movie feature vector but recognizes there may be other information available about the user and/or movie that may improve the prediction. The additional information is provided to a neural network which then generates the user and movie vector as shown below.
<figure>
    <center> <img src="./images/RecSysNN.png"   style="width:500px;height:280px;" ></center>
</figure>

<a name="3.1"></a>
### 3.1 Training Data
The movie content provided to the network is a combination of the original data and some 'engineered features'. Recall the feature engineering discussion and lab from Course 1, Week 2, lab 4. The original features are the year the movie was released and the movie's genre's presented as a one-hot vector. There are 14 genres. The engineered feature is an average rating derived from the user ratings. 

The user content is composed of engineered features. A per genre average rating is computed per user. Additionally, a user id, rating count and rating average are available but not included in the training or prediction content. They are carried with the data set because they are useful in interpreting data.

The training set consists of all the ratings made by the users in the data set. Some ratings are repeated to boost the number of training examples of underrepresented genre's. The training set is split into two arrays with the same number of entries, a user array and a movie/item array.  

Below, let's load and display some of the data.


```python
# Load Data, set configuration variables
item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

num_user_features = user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
num_item_features = item_train.shape[1] - 1  # remove movie id at train time
uvs = 3  # user genre vector start
ivs = 3  # item genre vector start
u_s = 3  # start of columns to use in training, user
i_s = 1  # start of columns to use in training, items
print(f"Number of training vectors: {len(item_train)}")
```

    Number of training vectors: 50884


Let's look at the first few entries in the user training array.


```python
pprint_train(user_train, user_features, uvs,  u_s, maxcount=5)
```




<table>
<thead>
<tr><th style="text-align: center;"> [user id] </th><th style="text-align: center;"> [rating count] </th><th style="text-align: center;"> [rating ave] </th><th style="text-align: center;"> Act ion </th><th style="text-align: center;"> Adve nture </th><th style="text-align: center;"> Anim ation </th><th style="text-align: center;"> Chil dren </th><th style="text-align: center;"> Com edy </th><th style="text-align: center;"> Crime </th><th style="text-align: center;"> Docum entary </th><th style="text-align: center;"> Drama </th><th style="text-align: center;"> Fan tasy </th><th style="text-align: center;"> Hor ror </th><th style="text-align: center;"> Mys tery </th><th style="text-align: center;"> Rom ance </th><th style="text-align: center;"> Sci -Fi </th><th style="text-align: center;"> Thri ller </th></tr>
</thead>
<tbody>
<tr><td style="text-align: center;">     2     </td><td style="text-align: center;">       22       </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">    4.2     </td><td style="text-align: center;">    0.0     </td><td style="text-align: center;">    0.0    </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">  4.1  </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">  4.0  </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.0   </td><td style="text-align: center;">   4.0    </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.9   </td><td style="text-align: center;">    3.9    </td></tr>
<tr><td style="text-align: center;">     2     </td><td style="text-align: center;">       22       </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">    4.2     </td><td style="text-align: center;">    0.0     </td><td style="text-align: center;">    0.0    </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">  4.1  </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">  4.0  </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.0   </td><td style="text-align: center;">   4.0    </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.9   </td><td style="text-align: center;">    3.9    </td></tr>
<tr><td style="text-align: center;">     2     </td><td style="text-align: center;">       22       </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">    4.2     </td><td style="text-align: center;">    0.0     </td><td style="text-align: center;">    0.0    </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">  4.1  </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">  4.0  </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.0   </td><td style="text-align: center;">   4.0    </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.9   </td><td style="text-align: center;">    3.9    </td></tr>
<tr><td style="text-align: center;">     2     </td><td style="text-align: center;">       22       </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">    4.2     </td><td style="text-align: center;">    0.0     </td><td style="text-align: center;">    0.0    </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">  4.1  </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">  4.0  </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.0   </td><td style="text-align: center;">   4.0    </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.9   </td><td style="text-align: center;">    3.9    </td></tr>
<tr><td style="text-align: center;">     2     </td><td style="text-align: center;">       22       </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">    4.2     </td><td style="text-align: center;">    0.0     </td><td style="text-align: center;">    0.0    </td><td style="text-align: center;">   4.0   </td><td style="text-align: center;">  4.1  </td><td style="text-align: center;">     4.0      </td><td style="text-align: center;">  4.0  </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.0   </td><td style="text-align: center;">   4.0    </td><td style="text-align: center;">   0.0    </td><td style="text-align: center;">   3.9   </td><td style="text-align: center;">    3.9    </td></tr>
</tbody>
</table>



Some of the user and item/movie features are not used in training. In the table above, the features in brackets "[]" such as the "user id", "rating count" and "rating ave" are not included when the model is trained and used.
Above you can see the per genre rating average for user 2. Zero entries are genre's which the user had not rated. The user vector is the same for all the movies rated by a user.  
Let's look at the first few entries of the movie/item array.


```python
pprint_train(item_train, item_features, ivs, i_s, maxcount=5, user=False)
```




<table>
<thead>
<tr><th style="text-align: center;"> [movie id] </th><th style="text-align: center;"> year </th><th style="text-align: center;"> ave rating </th><th style="text-align: center;"> Act ion </th><th style="text-align: center;"> Adve nture </th><th style="text-align: center;"> Anim ation </th><th style="text-align: center;"> Chil dren </th><th style="text-align: center;"> Com edy </th><th style="text-align: center;"> Crime </th><th style="text-align: center;"> Docum entary </th><th style="text-align: center;"> Drama </th><th style="text-align: center;"> Fan tasy </th><th style="text-align: center;"> Hor ror </th><th style="text-align: center;"> Mys tery </th><th style="text-align: center;"> Rom ance </th><th style="text-align: center;"> Sci -Fi </th><th style="text-align: center;"> Thri ller </th></tr>
</thead>
<tbody>
<tr><td style="text-align: center;">    6874    </td><td style="text-align: center;"> 2003 </td><td style="text-align: center;">    4.0     </td><td style="text-align: center;">    1    </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">   1   </td><td style="text-align: center;">      0       </td><td style="text-align: center;">   0   </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">     1     </td></tr>
<tr><td style="text-align: center;">    8798    </td><td style="text-align: center;"> 2004 </td><td style="text-align: center;">    3.8     </td><td style="text-align: center;">    1    </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">   1   </td><td style="text-align: center;">      0       </td><td style="text-align: center;">   1   </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">     1     </td></tr>
<tr><td style="text-align: center;">   46970    </td><td style="text-align: center;"> 2006 </td><td style="text-align: center;">    3.2     </td><td style="text-align: center;">    1    </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0     </td><td style="text-align: center;">    1    </td><td style="text-align: center;">   0   </td><td style="text-align: center;">      0       </td><td style="text-align: center;">   0   </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">     0     </td></tr>
<tr><td style="text-align: center;">   48516    </td><td style="text-align: center;"> 2006 </td><td style="text-align: center;">    4.3     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">   1   </td><td style="text-align: center;">      0       </td><td style="text-align: center;">   1   </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">     1     </td></tr>
<tr><td style="text-align: center;">   58559    </td><td style="text-align: center;"> 2008 </td><td style="text-align: center;">    4.2     </td><td style="text-align: center;">    1    </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0      </td><td style="text-align: center;">     0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">   1   </td><td style="text-align: center;">      0       </td><td style="text-align: center;">   1   </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0     </td><td style="text-align: center;">    0    </td><td style="text-align: center;">     0     </td></tr>
</tbody>
</table>



Above, the movie array contains the year the film was released, the average rating and an indicator for each potential genre. The indicator is one for each genre that applies to the movie. The movie id is not used in training but is useful when interpreting the data.


```python
print(f"y_train[:5]: {y_train[:5]}")
```

    y_train[:5]: [4.  3.5 4.  4.  4.5]


The target, y, is the movie rating given by the user. 

Above, we can see that movie 6874 is an Action/Crime/Thriller movie released in 2003. User 2 rates action movies as 3.9 on average. MovieLens users gave the movie an average rating of 4. 'y' is 4 indicating user 2 rated movie 6874 as a 4 as well. A single training example consists of a row from both the user and item arrays and a rating from y_train.

<a name="3.2"></a>
### 3.2 Preparing the training data
Recall in Course 1, Week 2, you explored feature scaling as a means of improving convergence. We'll scale the input features using the [scikit learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). This was used in Course 1, Week 2, Lab 5.  Below, the inverse_transform is also shown to produce the original inputs. We'll scale the target ratings using a Min Max Scaler which scales the target to be between -1 and 1. [scikit learn MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)


```python
# scale training data
item_train_unscaled = item_train
user_train_unscaled = user_train
y_train_unscaled    = y_train

scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train = scalerItem.transform(item_train)

scalerUser = StandardScaler()
scalerUser.fit(user_train)
user_train = scalerUser.transform(user_train)

scalerTarget = MinMaxScaler((-1, 1))
scalerTarget.fit(y_train.reshape(-1, 1))
y_train = scalerTarget.transform(y_train.reshape(-1, 1))
#ynorm_test = scalerTarget.transform(y_test.reshape(-1, 1))

print(np.allclose(item_train_unscaled, scalerItem.inverse_transform(item_train)))
print(np.allclose(user_train_unscaled, scalerUser.inverse_transform(user_train)))
```

    True
    True


To allow us to evaluate the results, we will split the data into training and test sets as was discussed in Course 2, Week 3. Here we will use [sklean train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) to split and shuffle the data. Note that setting the initial random state to the same value ensures item, user, and y are shuffled identically.


```python
item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test       = train_test_split(y_train,    train_size=0.80, shuffle=True, random_state=1)
print(f"movie/item training data shape: {item_train.shape}")
print(f"movie/item test data shape: {item_test.shape}")
```

    movie/item training data shape: (40707, 17)
    movie/item test data shape: (10177, 17)


The scaled, shuffled data now has a mean of zero.


```python
pprint_train(user_train, user_features, uvs, u_s, maxcount=5)
```

<a name="4"></a>
## 4 - Neural Network for content-based filtering
Now, let's construct a neural network as described in the figure above. It will have two networks that are combined by a dot product. You will construct the two networks. In this example, they will be identical. Note that these networks do not need to be the same. If the user content was substantially larger than the movie content, you might elect to increase the complexity of the user network relative to the movie network. In this case, the content is similar, so the networks are the same.

<a name="ex01"></a>
### Exercise 1

- Use a Keras sequential model
    - The first layer is a dense layer with 256 units and a relu activation.
    - The second layer is a dense layer with 128 units and a relu activation.
    - The third layer is a dense layer with `num_outputs` units and a linear or no activation.   
    
The remainder of the network will be provided. The provided code does not use the Keras sequential model but instead uses the Keras [functional api](https://keras.io/guides/functional_api/). This format allows for more flexibility in how components are interconnected.



```python
# GRADED_CELL
# UNQ_C1

num_outputs = 32
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
    ### START CODE HERE ###     
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_outputs, activation='linear')
  
  
    ### END CODE HERE ###  
])

item_NN = tf.keras.models.Sequential([
    ### START CODE HERE ###     
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_outputs, activation='linear')
  
  
    ### END CODE HERE ###  
])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = tf.keras.Model([input_user, input_item], output)

model.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 14)]         0                                            
    __________________________________________________________________________________________________
    input_2 (InputLayer)            [(None, 16)]         0                                            
    __________________________________________________________________________________________________
    sequential (Sequential)         (None, 32)           40864       input_1[0][0]                    
    __________________________________________________________________________________________________
    sequential_1 (Sequential)       (None, 32)           41376       input_2[0][0]                    
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize/Square [(None, 32)]         0           sequential[0][0]                 
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize_1/Squa [(None, 32)]         0           sequential_1[0][0]               
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize/Sum (T [(None, 1)]          0           tf_op_layer_l2_normalize/Square[0
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize_1/Sum  [(None, 1)]          0           tf_op_layer_l2_normalize_1/Square
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize/Maximu [(None, 1)]          0           tf_op_layer_l2_normalize/Sum[0][0
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize_1/Maxi [(None, 1)]          0           tf_op_layer_l2_normalize_1/Sum[0]
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize/Rsqrt  [(None, 1)]          0           tf_op_layer_l2_normalize/Maximum[
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize_1/Rsqr [(None, 1)]          0           tf_op_layer_l2_normalize_1/Maximu
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize (Tenso [(None, 32)]         0           sequential[0][0]                 
                                                                     tf_op_layer_l2_normalize/Rsqrt[0]
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize_1 (Ten [(None, 32)]         0           sequential_1[0][0]               
                                                                     tf_op_layer_l2_normalize_1/Rsqrt[
    __________________________________________________________________________________________________
    dot (Dot)                       (None, 1)            0           tf_op_layer_l2_normalize[0][0]   
                                                                     tf_op_layer_l2_normalize_1[0][0] 
    ==================================================================================================
    Total params: 82,240
    Trainable params: 82,240
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
# Public tests
from public_tests import *
test_tower(user_NN)
test_tower(item_NN)
```

    [92mAll tests passed!
    [92mAll tests passed!


<details>
  <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
    
  You can create a dense layer with a relu activation as shown.
    
```python     
user_NN = tf.keras.models.Sequential([
    ### START CODE HERE ###     
  tf.keras.layers.Dense(256, activation='relu'),

    
    ### END CODE HERE ###  
])

item_NN = tf.keras.models.Sequential([
    ### START CODE HERE ###     
  tf.keras.layers.Dense(256, activation='relu'),

    
    ### END CODE HERE ###  
])
```    
<details>
    <summary><font size="2" color="darkblue"><b> Click for solution</b></font></summary>
    
```python 
user_NN = tf.keras.models.Sequential([
    ### START CODE HERE ###     
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_outputs),
    ### END CODE HERE ###  
])

item_NN = tf.keras.models.Sequential([
    ### START CODE HERE ###     
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_outputs),
    ### END CODE HERE ###  
])
```
</details>
</details>

    


We will use a mean squared error loss and an Adam optimizer.


```python
tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
              loss=cost_fn)
```


```python
tf.random.set_seed(1)
model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)
```

    Train on 40707 samples
    Epoch 1/30
    40707/40707 [==============================] - 5s 122us/sample - loss: 0.0707
    Epoch 2/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0704
    Epoch 3/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0697
    Epoch 4/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0694
    Epoch 5/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0689
    Epoch 6/30
    40707/40707 [==============================] - 5s 122us/sample - loss: 0.0686
    Epoch 7/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0683
    Epoch 8/30
    40707/40707 [==============================] - 5s 121us/sample - loss: 0.0679
    Epoch 9/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0676
    Epoch 10/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0671
    Epoch 11/30
    40707/40707 [==============================] - 5s 119us/sample - loss: 0.0669
    Epoch 12/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0666
    Epoch 13/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0662
    Epoch 14/30
    40707/40707 [==============================] - 5s 122us/sample - loss: 0.0662
    Epoch 15/30
    40707/40707 [==============================] - 5s 121us/sample - loss: 0.0659
    Epoch 16/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0653
    Epoch 17/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0653
    Epoch 18/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0650
    Epoch 19/30
    40707/40707 [==============================] - 5s 122us/sample - loss: 0.0647
    Epoch 20/30
    40707/40707 [==============================] - 5s 121us/sample - loss: 0.0645
    Epoch 21/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0642
    Epoch 22/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0640
    Epoch 23/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0638
    Epoch 24/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0635
    Epoch 25/30
    40707/40707 [==============================] - 5s 120us/sample - loss: 0.0635
    Epoch 26/30
    40707/40707 [==============================] - 5s 119us/sample - loss: 0.0632
    Epoch 27/30
    40707/40707 [==============================] - 5s 122us/sample - loss: 0.0629
    Epoch 28/30
    40707/40707 [==============================] - 5s 125us/sample - loss: 0.0628
    Epoch 29/30
    40707/40707 [==============================] - 5s 123us/sample - loss: 0.0628
    Epoch 30/30
    40707/40707 [==============================] - 5s 123us/sample - loss: 0.0626





    <tensorflow.python.keras.callbacks.History at 0x7fd3c0f2c250>



Evaluate the model to determine loss on the test data. 


```python
model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)
```

    10177/10177 [==============================] - 0s 38us/sample - loss: 0.0780





    0.07799451819504714



It is comparable to the training loss indicating the model has not substantially overfit the training data.

<a name="5"></a>
## 5 - Predictions
Below, you'll use your model to make predictions in a number of circumstances. 
<a name="5.1"></a>
### 5.1 - Predictions for a new user
First, we'll create a new user and have the model suggest movies for that user. After you have tried this on the example user content, feel free to change the user content to match your own preferences and see what the model suggests. Note that ratings are between 0.5 and 5.0, inclusive, in half-step increments.


```python
new_user_id = 5000
new_rating_ave = 0.0
new_action = 0.0
new_adventure = 5.0
new_animation = 0.0
new_childrens = 0.0
new_comedy = 0.0
new_crime = 0.0
new_documentary = 0.0
new_drama = 0.0
new_fantasy = 5.0
new_horror = 0.0
new_mystery = 0.0
new_romance = 0.0
new_scifi = 0.0
new_thriller = 0.0
new_rating_count = 3

user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                      new_action, new_adventure, new_animation, new_childrens,
                      new_comedy, new_crime, new_documentary,
                      new_drama, new_fantasy, new_horror, new_mystery,
                      new_romance, new_scifi, new_thriller]])
```

The new user enjoys movies from the adventure, fantasy genres. Let's find the top-rated movies for the new user.  
Below, we'll use a set of movie/item vectors, `item_vecs` that have a vector for each movie in the training/test set. This is matched with the new user vector above and the scaled vectors are used to predict ratings for all the movies.


```python
# generate and replicate the user vector to match the number movies in the data set.
user_vecs = gen_user_vecs(user_vec,len(item_vecs))

# scale our user and item vectors
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# make a prediction
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y prediction 
y_pu = scalerTarget.inverse_transform(y_p)

# sort the results, highest prediction first
sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
sorted_ypu   = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  #using unscaled vectors for display

print_pred_movies(sorted_ypu, sorted_items, movie_dict, maxcount = 10)
```




<table>
<thead>
<tr><th style="text-align: right;">  y_p</th><th style="text-align: right;">  movie id</th><th style="text-align: right;">  rating ave</th><th>title                                              </th><th>genres                           </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">  4.5</td><td style="text-align: right;">     98809</td><td style="text-align: right;">         3.8</td><td>Hobbit: An Unexpected Journey, The (2012)          </td><td>Adventure|Fantasy                </td></tr>
<tr><td style="text-align: right;">  4.4</td><td style="text-align: right;">     40815</td><td style="text-align: right;">         3.8</td><td>Harry Potter and the Goblet of Fire (2005)         </td><td>Adventure|Fantasy|Thriller       </td></tr>
<tr><td style="text-align: right;">  4.4</td><td style="text-align: right;">      8368</td><td style="text-align: right;">         3.9</td><td>Harry Potter and the Prisoner of Azkaban (2004)    </td><td>Adventure|Fantasy                </td></tr>
<tr><td style="text-align: right;">  4.4</td><td style="text-align: right;">     54001</td><td style="text-align: right;">         3.9</td><td>Harry Potter and the Order of the Phoenix (2007)   </td><td>Adventure|Drama|Fantasy          </td></tr>
<tr><td style="text-align: right;">  4.3</td><td style="text-align: right;">     59387</td><td style="text-align: right;">         4  </td><td>Fall, The (2006)                                   </td><td>Adventure|Drama|Fantasy          </td></tr>
<tr><td style="text-align: right;">  4.3</td><td style="text-align: right;">     87232</td><td style="text-align: right;">         3.8</td><td>X-Men: First Class (2011)                          </td><td>Action|Adventure|Sci-Fi|Thriller </td></tr>
<tr><td style="text-align: right;">  4.3</td><td style="text-align: right;">     69844</td><td style="text-align: right;">         3.9</td><td>Harry Potter and the Half-Blood Prince (2009)      </td><td>Adventure|Fantasy|Mystery|Romance</td></tr>
<tr><td style="text-align: right;">  4.3</td><td style="text-align: right;">      5816</td><td style="text-align: right;">         3.6</td><td>Harry Potter and the Chamber of Secrets (2002)     </td><td>Adventure|Fantasy                </td></tr>
<tr><td style="text-align: right;">  4.2</td><td style="text-align: right;">      5952</td><td style="text-align: right;">         4  </td><td>Lord of the Rings: The Two Towers, The (2002)      </td><td>Adventure|Fantasy                </td></tr>
<tr><td style="text-align: right;">  4.2</td><td style="text-align: right;">     81834</td><td style="text-align: right;">         4  </td><td>Harry Potter and the Deathly Hallows: Part 1 (2010)</td><td>Action|Adventure|Fantasy         </td></tr>
</tbody>
</table>



<a name="5.2"></a>
### 5.2 - Predictions for an existing user.
Let's look at the predictions for "user 2", one of the users in the data set. We can compare the predicted ratings with the model's ratings.


```python
uid = 2 
# form a set of user vectors. This is the same vector, transformed and repeated.
user_vecs, y_vecs = get_user_vecs(uid, user_train_unscaled, item_vecs, user_to_genre)

# scale our user and item vectors
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# make a prediction
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y prediction 
y_pu = scalerTarget.inverse_transform(y_p)

# sort the results, highest prediction first
sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
sorted_ypu   = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  #using unscaled vectors for display
sorted_user  = user_vecs[sorted_index]
sorted_y     = y_vecs[sorted_index]

#print sorted predictions for movies rated by the user
print_existing_user(sorted_ypu, sorted_y.reshape(-1,1), sorted_user, sorted_items, ivs, uvs, movie_dict, maxcount = 50)
```




<table>
<thead>
<tr><th style="text-align: right;">  y_p</th><th style="text-align: right;">  y</th><th style="text-align: right;">  user</th><th>user genre ave           </th><th style="text-align: right;">  movie rating ave</th><th style="text-align: right;">  movie id</th><th>title                                             </th><th>genres                                    </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">  4.5</td><td style="text-align: right;">5.0</td><td style="text-align: right;">     2</td><td>[4.0]                    </td><td style="text-align: right;">               4.3</td><td style="text-align: right;">     80906</td><td>Inside Job (2010)                                 </td><td>Documentary                               </td></tr>
<tr><td style="text-align: right;">  4.3</td><td style="text-align: right;">3.5</td><td style="text-align: right;">     2</td><td>[4.0,4.1,4.0,3.9]        </td><td style="text-align: right;">               3.8</td><td style="text-align: right;">      8798</td><td>Collateral (2004)                                 </td><td>Action|Crime|Drama|Thriller               </td></tr>
<tr><td style="text-align: right;">  4.2</td><td style="text-align: right;">3.5</td><td style="text-align: right;">     2</td><td>[4.0,3.9,3.9]            </td><td style="text-align: right;">               3.9</td><td style="text-align: right;">    115713</td><td>Ex Machina (2015)                                 </td><td>Drama|Sci-Fi|Thriller                     </td></tr>
<tr><td style="text-align: right;">  4.1</td><td style="text-align: right;">4.5</td><td style="text-align: right;">     2</td><td>[4.0,4.0]                </td><td style="text-align: right;">               4.1</td><td style="text-align: right;">     68157</td><td>Inglourious Basterds (2009)                       </td><td>Action|Drama                              </td></tr>
<tr><td style="text-align: right;">  4.0</td><td style="text-align: right;">4.0</td><td style="text-align: right;">     2</td><td>[4.1,4.0,3.9]            </td><td style="text-align: right;">               4.3</td><td style="text-align: right;">     48516</td><td>Departed, The (2006)                              </td><td>Crime|Drama|Thriller                      </td></tr>
<tr><td style="text-align: right;">  4.0</td><td style="text-align: right;">3.5</td><td style="text-align: right;">     2</td><td>[4.0,4.0]                </td><td style="text-align: right;">               3.9</td><td style="text-align: right;">     99114</td><td>Django Unchained (2012)                           </td><td>Action|Drama                              </td></tr>
<tr><td style="text-align: right;">  4.0</td><td style="text-align: right;">4.0</td><td style="text-align: right;">     2</td><td>[4.0,4.1,3.9]            </td><td style="text-align: right;">               4.0</td><td style="text-align: right;">      6874</td><td>Kill Bill: Vol. 1 (2003)                          </td><td>Action|Crime|Thriller                     </td></tr>
<tr><td style="text-align: right;">  3.9</td><td style="text-align: right;">4.0</td><td style="text-align: right;">     2</td><td>[4.0,4.1,4.0,4.0,3.9,3.9]</td><td style="text-align: right;">               4.1</td><td style="text-align: right;">     79132</td><td>Inception (2010)                                  </td><td>Action|Crime|Drama|Mystery|Sci-Fi|Thriller</td></tr>
<tr><td style="text-align: right;">  3.9</td><td style="text-align: right;">4.5</td><td style="text-align: right;">     2</td><td>[4.0,4.1,4.0]            </td><td style="text-align: right;">               4.2</td><td style="text-align: right;">     58559</td><td>Dark Knight, The (2008)                           </td><td>Action|Crime|Drama                        </td></tr>
<tr><td style="text-align: right;">  3.8</td><td style="text-align: right;">3.5</td><td style="text-align: right;">     2</td><td>[4.0,4.2,4.1]            </td><td style="text-align: right;">               4.0</td><td style="text-align: right;">     91529</td><td>Dark Knight Rises, The (2012)                     </td><td>Action|Adventure|Crime                    </td></tr>
<tr><td style="text-align: right;">  3.8</td><td style="text-align: right;">5.0</td><td style="text-align: right;">     2</td><td>[4.0,4.2,3.9,3.9]        </td><td style="text-align: right;">               3.8</td><td style="text-align: right;">    122882</td><td>Mad Max: Fury Road (2015)                         </td><td>Action|Adventure|Sci-Fi|Thriller          </td></tr>
<tr><td style="text-align: right;">  3.8</td><td style="text-align: right;">4.0</td><td style="text-align: right;">     2</td><td>[4.0]                    </td><td style="text-align: right;">               4.0</td><td style="text-align: right;">    112552</td><td>Whiplash (2014)                                   </td><td>Drama                                     </td></tr>
<tr><td style="text-align: right;">  3.8</td><td style="text-align: right;">5.0</td><td style="text-align: right;">     2</td><td>[4.0]                    </td><td style="text-align: right;">               3.6</td><td style="text-align: right;">     60756</td><td>Step Brothers (2008)                              </td><td>Comedy                                    </td></tr>
<tr><td style="text-align: right;">  3.8</td><td style="text-align: right;">5.0</td><td style="text-align: right;">     2</td><td>[4.0]                    </td><td style="text-align: right;">               3.7</td><td style="text-align: right;">     89774</td><td>Warrior (2011)                                    </td><td>Drama                                     </td></tr>
<tr><td style="text-align: right;">  3.8</td><td style="text-align: right;">5.0</td><td style="text-align: right;">     2</td><td>[4.0,4.1,4.0]            </td><td style="text-align: right;">               3.9</td><td style="text-align: right;">    106782</td><td>Wolf of Wall Street, The (2013)                   </td><td>Comedy|Crime|Drama                        </td></tr>
<tr><td style="text-align: right;">  3.7</td><td style="text-align: right;">3.0</td><td style="text-align: right;">     2</td><td>[3.9]                    </td><td style="text-align: right;">               4.0</td><td style="text-align: right;">    109487</td><td>Interstellar (2014)                               </td><td>Sci-Fi                                    </td></tr>
<tr><td style="text-align: right;">  3.6</td><td style="text-align: right;">4.0</td><td style="text-align: right;">     2</td><td>[4.0,4.0,3.9]            </td><td style="text-align: right;">               4.0</td><td style="text-align: right;">     74458</td><td>Shutter Island (2010)                             </td><td>Drama|Mystery|Thriller                    </td></tr>
<tr><td style="text-align: right;">  3.6</td><td style="text-align: right;">3.0</td><td style="text-align: right;">     2</td><td>[4.0,4.0,3.0]            </td><td style="text-align: right;">               3.9</td><td style="text-align: right;">     71535</td><td>Zombieland (2009)                                 </td><td>Action|Comedy|Horror                      </td></tr>
<tr><td style="text-align: right;">  3.5</td><td style="text-align: right;">4.5</td><td style="text-align: right;">     2</td><td>[4.1,4.0,3.9]            </td><td style="text-align: right;">               4.0</td><td style="text-align: right;">     80489</td><td>Town, The (2010)                                  </td><td>Crime|Drama|Thriller                      </td></tr>
<tr><td style="text-align: right;">  3.2</td><td style="text-align: right;">3.0</td><td style="text-align: right;">     2</td><td>[4.0,4.0]                </td><td style="text-align: right;">               4.0</td><td style="text-align: right;">     77455</td><td>Exit Through the Gift Shop (2010)                 </td><td>Comedy|Documentary                        </td></tr>
<tr><td style="text-align: right;">  3.2</td><td style="text-align: right;">2.5</td><td style="text-align: right;">     2</td><td>[4.0,3.9]                </td><td style="text-align: right;">               3.5</td><td style="text-align: right;">     91658</td><td>Girl with the Dragon Tattoo, The (2011)           </td><td>Drama|Thriller                            </td></tr>
<tr><td style="text-align: right;">  2.8</td><td style="text-align: right;">4.0</td><td style="text-align: right;">     2</td><td>[4.0,4.0]                </td><td style="text-align: right;">               3.2</td><td style="text-align: right;">     46970</td><td>Talladega Nights: The Ballad of Ricky Bobby (2006)</td><td>Action|Comedy                             </td></tr>
</tbody>
</table>



The model prediction is generally within 1 of the actual rating though it is not a very accurate predictor of how a user rates specific movies. This is especially true if the user rating is significantly different than the user's genre average. You can vary the user id above to try different users. Not all user id's were used in the training set.

<a name="5.3"></a>
### 5.3 - Finding Similar Items
The neural network above produces two feature vectors, a user feature vector $v_u$, and a movie feature vector, $v_m$. These are 32 entry vectors whose values are difficult to interpret. However, similar items will have similar vectors. This information can be used to make recommendations. For example, if a user has rated "Toy Story 3" highly, one could recommend similar movies by selecting movies with similar movie feature vectors.

A similarity measure is the squared distance between the two vectors $ \mathbf{v_m^{(k)}}$ and $\mathbf{v_m^{(i)}}$ :
$$\left\Vert \mathbf{v_m^{(k)}} - \mathbf{v_m^{(i)}}  \right\Vert^2 = \sum_{l=1}^{n}(v_{m_l}^{(k)} - v_{m_l}^{(i)})^2\tag{1}$$

<a name="ex02"></a>
### Exercise 2

Write a function to compute the square distance.


```python
# GRADED_FUNCTION: sq_dist
# UNQ_C2
def sq_dist(a,b):
    """
    Returns the squared distance between two vectors
    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features
    Returns:
      d (float) : distance
    """
    ### START CODE HERE ###     
    d = np.sum((a - b) ** 2)
    ### END CODE HERE ###     
    return d
```


```python
a1 = np.array([1.0, 2.0, 3.0]); b1 = np.array([1.0, 2.0, 3.0])
a2 = np.array([1.1, 2.1, 3.1]); b2 = np.array([1.0, 2.0, 3.0])
a3 = np.array([0, 1, 0]);       b3 = np.array([1, 0, 0])
print(f"squared distance between a1 and b1: {sq_dist(a1, b1):0.3f}")
print(f"squared distance between a2 and b2: {sq_dist(a2, b2):0.3f}")
print(f"squared distance between a3 and b3: {sq_dist(a3, b3):0.3f}")
```

    squared distance between a1 and b1: 0.000
    squared distance between a2 and b2: 0.030
    squared distance between a3 and b3: 2.000


**Expected Output**:

squared distance between a1 and b1: 0.000    
squared distance between a2 and b2: 0.030   
squared distance between a3 and b3: 2.000


```python
# Public tests
test_sq_dist(sq_dist)
```

    [92mAll tests passed!


<details>
  <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
    
  While a summation is often an indication a for loop should be used, here the subtraction can be element-wise in one statement. Further, you can utilized np.square to square, element-wise, the result of the subtraction. np.sum can be used to sum the squared elements.
    
</details>

    


A matrix of distances between movies can be computed once when the model is trained and then reused for new recommendations without retraining. The first step, once a model is trained, is to obtain the movie feature vector, $v_m$, for each of the movies. To do this, we will use the trained `item_NN` and build a small model to allow us to run the movie vectors through it to generate $v_m$.


```python
input_item_m = tf.keras.layers.Input(shape=(num_item_features))    # input layer
vm_m = item_NN(input_item_m)                                       # use the trained item_NN
vm_m = tf.linalg.l2_normalize(vm_m, axis=1)                        # incorporate normalization as was done in the original model
model_m = tf.keras.Model(input_item_m, vm_m)                                
model_m.summary()
```

    Model: "model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_3 (InputLayer)            [(None, 16)]         0                                            
    __________________________________________________________________________________________________
    sequential_1 (Sequential)       (None, 32)           41376       input_3[0][0]                    
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize_2/Squa [(None, 32)]         0           sequential_1[1][0]               
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize_2/Sum  [(None, 1)]          0           tf_op_layer_l2_normalize_2/Square
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize_2/Maxi [(None, 1)]          0           tf_op_layer_l2_normalize_2/Sum[0]
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize_2/Rsqr [(None, 1)]          0           tf_op_layer_l2_normalize_2/Maximu
    __________________________________________________________________________________________________
    tf_op_layer_l2_normalize_2 (Ten [(None, 32)]         0           sequential_1[1][0]               
                                                                     tf_op_layer_l2_normalize_2/Rsqrt[
    ==================================================================================================
    Total params: 41,376
    Trainable params: 41,376
    Non-trainable params: 0
    __________________________________________________________________________________________________


Once you have a movie model, you can create a set of movie feature vectors by using the model to predict using a set of item/movie vectors as input. `item_vecs` is a set of all of the movie vectors. It must be scaled to use with the trained model. The result of the prediction is a 32 entry feature vector for each movie.


```python
scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:,i_s:])
print(f"size of all predicted movie feature vectors: {vms.shape}")
```

    size of all predicted movie feature vectors: (847, 32)


Let's now compute a matrix of the squared distance between each movie feature vector and all other movie feature vectors:
<figure>
    <left> <img src="./images/distmatrix.PNG"   style="width:400px;height:225px;" ></center>
</figure>

We can then find the closest movie by finding the minimum along each row. We will make use of [numpy masked arrays](https://numpy.org/doc/1.21/user/tutorial-ma.html) to avoid selecting the same movie. The masked values along the diagonal won't be included in the computation.


```python
count = 50  # number of movies to display
dim = len(vms)
dist = np.zeros((dim,dim))

for i in range(dim):
    for j in range(dim):
        dist[i,j] = sq_dist(vms[i, :], vms[j, :])
        
m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0]))  # mask the diagonal

disp = [["movie1", "genres", "movie2", "genres"]]
for i in range(count):
    min_idx = np.argmin(m_dist[i])
    movie1_id = int(item_vecs[i,0])
    movie2_id = int(item_vecs[min_idx,0])
    disp.append( [movie_dict[movie1_id]['title'], movie_dict[movie1_id]['genres'],
                  movie_dict[movie2_id]['title'], movie_dict[movie1_id]['genres']]
               )
table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
table
```




<table>
<thead>
<tr><th>movie1                                  </th><th>genres                                             </th><th>movie2                                                 </th><th>genres                                             </th></tr>
</thead>
<tbody>
<tr><td>Save the Last Dance (2001)              </td><td>Drama|Romance                                      </td><td>Mona Lisa Smile (2003)                                 </td><td>Drama|Romance                                      </td></tr>
<tr><td>Wedding Planner, The (2001)             </td><td>Comedy|Romance                                     </td><td>Mr. Deeds (2002)                                       </td><td>Comedy|Romance                                     </td></tr>
<tr><td>Hannibal (2001)                         </td><td>Horror|Thriller                                    </td><td>Final Destination 2 (2003)                             </td><td>Horror|Thriller                                    </td></tr>
<tr><td>Saving Silverman (Evil Woman) (2001)    </td><td>Comedy|Romance                                     </td><td>Sweetest Thing, The (2002)                             </td><td>Comedy|Romance                                     </td></tr>
<tr><td>Down to Earth (2001)                    </td><td>Comedy|Fantasy|Romance                             </td><td>America&#x27;s Sweethearts (2001)                           </td><td>Comedy|Fantasy|Romance                             </td></tr>
<tr><td>Mexican, The (2001)                     </td><td>Action|Comedy                                      </td><td>Rush Hour 2 (2001)                                     </td><td>Action|Comedy                                      </td></tr>
<tr><td>15 Minutes (2001)                       </td><td>Thriller                                           </td><td>Panic Room (2002)                                      </td><td>Thriller                                           </td></tr>
<tr><td>Enemy at the Gates (2001)               </td><td>Drama                                              </td><td>Bourne Supremacy, The (2004)                           </td><td>Drama                                              </td></tr>
<tr><td>Heartbreakers (2001)                    </td><td>Comedy|Crime|Romance                               </td><td>Fun with Dick and Jane (2005)                          </td><td>Comedy|Crime|Romance                               </td></tr>
<tr><td>Spy Kids (2001)                         </td><td>Action|Adventure|Children|Comedy                   </td><td>Scooby-Doo (2002)                                      </td><td>Action|Adventure|Children|Comedy                   </td></tr>
<tr><td>Along Came a Spider (2001)              </td><td>Action|Crime|Mystery|Thriller                      </td><td>Insomnia (2002)                                        </td><td>Action|Crime|Mystery|Thriller                      </td></tr>
<tr><td>Blow (2001)                             </td><td>Crime|Drama                                        </td><td>Man Who Wasn&#x27;t There, The (2001)                       </td><td>Crime|Drama                                        </td></tr>
<tr><td>Bridget Jones&#x27;s Diary (2001)            </td><td>Comedy|Drama|Romance                               </td><td>Punch-Drunk Love (2002)                                </td><td>Comedy|Drama|Romance                               </td></tr>
<tr><td>Joe Dirt (2001)                         </td><td>Adventure|Comedy|Mystery|Romance                   </td><td>Alexander (2004)                                       </td><td>Adventure|Comedy|Mystery|Romance                   </td></tr>
<tr><td>Crocodile Dundee in Los Angeles (2001)  </td><td>Comedy|Drama                                       </td><td>Legally Blonde 2: Red, White &amp; Blonde (2003)           </td><td>Comedy|Drama                                       </td></tr>
<tr><td>Mummy Returns, The (2001)               </td><td>Action|Adventure|Comedy|Thriller                   </td><td>Reign of Fire (2002)                                   </td><td>Action|Adventure|Comedy|Thriller                   </td></tr>
<tr><td>Knight&#x27;s Tale, A (2001)                 </td><td>Action|Comedy|Romance                              </td><td>My Big Fat Greek Wedding (2002)                        </td><td>Action|Comedy|Romance                              </td></tr>
<tr><td>Shrek (2001)                            </td><td>Adventure|Animation|Children|Comedy|Fantasy|Romance</td><td>Tangled (2010)                                         </td><td>Adventure|Animation|Children|Comedy|Fantasy|Romance</td></tr>
<tr><td>Moulin Rouge (2001)                     </td><td>Drama|Romance                                      </td><td>Hours, The (2002)                                      </td><td>Drama|Romance                                      </td></tr>
<tr><td>Pearl Harbor (2001)                     </td><td>Action|Drama|Romance                               </td><td>Fun with Dick and Jane (2005)                          </td><td>Action|Drama|Romance                               </td></tr>
<tr><td>Animal, The (2001)                      </td><td>Comedy                                             </td><td>Dumb and Dumberer: When Harry Met Lloyd (2003)         </td><td>Comedy                                             </td></tr>
<tr><td>Evolution (2001)                        </td><td>Comedy|Sci-Fi                                      </td><td>Behind Enemy Lines (2001)                              </td><td>Comedy|Sci-Fi                                      </td></tr>
<tr><td>Swordfish (2001)                        </td><td>Action|Crime|Drama                                 </td><td>Bourne Legacy, The (2012)                              </td><td>Action|Crime|Drama                                 </td></tr>
<tr><td>Atlantis: The Lost Empire (2001)        </td><td>Adventure|Animation|Children|Fantasy               </td><td>Meet the Robinsons (2007)                              </td><td>Adventure|Animation|Children|Fantasy               </td></tr>
<tr><td>Lara Croft: Tomb Raider (2001)          </td><td>Action|Adventure                                   </td><td>Paycheck (2003)                                        </td><td>Action|Adventure                                   </td></tr>
<tr><td>Dr. Dolittle 2 (2001)                   </td><td>Comedy                                             </td><td>Legally Blonde 2: Red, White &amp; Blonde (2003)           </td><td>Comedy                                             </td></tr>
<tr><td>Fast and the Furious, The (2001)        </td><td>Action|Crime|Thriller                              </td><td>xXx (2002)                                             </td><td>Action|Crime|Thriller                              </td></tr>
<tr><td>A.I. Artificial Intelligence (2001)     </td><td>Adventure|Drama|Sci-Fi                             </td><td>Bad Santa (2003)                                       </td><td>Adventure|Drama|Sci-Fi                             </td></tr>
<tr><td>Cats &amp; Dogs (2001)                      </td><td>Children|Comedy                                    </td><td>Stepford Wives, The (2004)                             </td><td>Children|Comedy                                    </td></tr>
<tr><td>Scary Movie 2 (2001)                    </td><td>Comedy                                             </td><td>Orange County (2002)                                   </td><td>Comedy                                             </td></tr>
<tr><td>Final Fantasy: The Spirits Within (2001)</td><td>Adventure|Animation|Fantasy|Sci-Fi                 </td><td>Enemy at the Gates (2001)                              </td><td>Adventure|Animation|Fantasy|Sci-Fi                 </td></tr>
<tr><td>Legally Blonde (2001)                   </td><td>Comedy|Romance                                     </td><td>Serendipity (2001)                                     </td><td>Comedy|Romance                                     </td></tr>
<tr><td>Score, The (2001)                       </td><td>Action|Drama                                       </td><td>Bourne Ultimatum, The (2007)                           </td><td>Action|Drama                                       </td></tr>
<tr><td>Jurassic Park III (2001)                </td><td>Action|Adventure|Sci-Fi|Thriller                   </td><td>Time Machine, The (2002)                               </td><td>Action|Adventure|Sci-Fi|Thriller                   </td></tr>
<tr><td>America&#x27;s Sweethearts (2001)            </td><td>Comedy|Romance                                     </td><td>Maid in Manhattan (2002)                               </td><td>Comedy|Romance                                     </td></tr>
<tr><td>Ghost World (2001)                      </td><td>Comedy|Drama                                       </td><td>Royal Tenenbaums, The (2001)                           </td><td>Comedy|Drama                                       </td></tr>
<tr><td>Planet of the Apes (2001)               </td><td>Action|Adventure|Drama|Sci-Fi                      </td><td>Day After Tomorrow, The (2004)                         </td><td>Action|Adventure|Drama|Sci-Fi                      </td></tr>
<tr><td>Princess Diaries, The (2001)            </td><td>Children|Comedy|Romance                            </td><td>Bend It Like Beckham (2002)                            </td><td>Children|Comedy|Romance                            </td></tr>
<tr><td>Rush Hour 2 (2001)                      </td><td>Action|Comedy                                      </td><td>Mexican, The (2001)                                    </td><td>Action|Comedy                                      </td></tr>
<tr><td>American Pie 2 (2001)                   </td><td>Comedy                                             </td><td>Rat Race (2001)                                        </td><td>Comedy                                             </td></tr>
<tr><td>Others, The (2001)                      </td><td>Drama|Horror|Mystery|Thriller                      </td><td>Dogville (2003)                                        </td><td>Drama|Horror|Mystery|Thriller                      </td></tr>
<tr><td>Rat Race (2001)                         </td><td>Comedy                                             </td><td>American Pie 2 (2001)                                  </td><td>Comedy                                             </td></tr>
<tr><td>Jay and Silent Bob Strike Back (2001)   </td><td>Adventure|Comedy                                   </td><td>EuroTrip (2004)                                        </td><td>Adventure|Comedy                                   </td></tr>
<tr><td>Training Day (2001)                     </td><td>Crime|Drama|Thriller                               </td><td>Frailty (2001)                                         </td><td>Crime|Drama|Thriller                               </td></tr>
<tr><td>Zoolander (2001)                        </td><td>Comedy                                             </td><td>Old School (2003)                                      </td><td>Comedy                                             </td></tr>
<tr><td>Serendipity (2001)                      </td><td>Comedy|Romance                                     </td><td>Legally Blonde (2001)                                  </td><td>Comedy|Romance                                     </td></tr>
<tr><td>Mulholland Drive (2001)                 </td><td>Crime|Drama|Mystery|Thriller                       </td><td>Prisoners (2013)                                       </td><td>Crime|Drama|Mystery|Thriller                       </td></tr>
<tr><td>From Hell (2001)                        </td><td>Crime|Horror|Mystery|Thriller                      </td><td>Identity (2003)                                        </td><td>Crime|Horror|Mystery|Thriller                      </td></tr>
<tr><td>Waking Life (2001)                      </td><td>Animation|Drama|Fantasy                            </td><td>Birdman: Or (The Unexpected Virtue of Ignorance) (2014)</td><td>Animation|Drama|Fantasy                            </td></tr>
<tr><td>K-PAX (2001)                            </td><td>Drama|Fantasy|Mystery|Sci-Fi                       </td><td>Gosford Park (2001)                                    </td><td>Drama|Fantasy|Mystery|Sci-Fi                       </td></tr>
</tbody>
</table>



The results show the model will generally suggest a movie with similar genre's.

<a name="6"></a>
## 6 - Congratulations! <img align="left" src="./images/film_award.png" style=" width:40px;">
You have completed a content-based recommender system.    

This structure is the basis of many commercial recommender systems. The user content can be greatly expanded to incorporate more information about the user if it is available.  Items are not limited to movies. This can be used to recommend any item, books, cars or items that are similar to an item in your 'shopping cart'.

<details>
  <summary><font size="2" color="darkgreen"><b>Please click here if you want to experiment with any of the non-graded code.</b></font></summary>
    <p><i><b>Important Note: Please only do this when you've already passed the assignment to avoid problems with the autograder.</b></i>
    <ol>
        <li> On the notebook’s menu, click “View” > “Cell Toolbar” > “Edit Metadata”</li>
        <li> Hit the “Edit Metadata” button next to the code cell which you want to lock/unlock</li>
        <li> Set the attribute value for “editable” to:
            <ul>
                <li> “true” if you want to unlock it </li>
                <li> “false” if you want to lock it </li>
            </ul>
        </li>
        <li> On the notebook’s menu, click “View” > “Cell Toolbar” > “None” </li>
    </ol>
    <p> Here's a short demo of how to do the steps above: 
        <br>
        <img src="https://drive.google.com/uc?export=view&id=14Xy_Mb17CZVgzVAgq7NCjMVBvSae3xO1" align="center" alt="unlock_cells.gif">
</details>
