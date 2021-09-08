# Movie Recommendation Engine
This project seeks to build and implement a Movie Recommendation Engine from the Movie Lens 1M data set.

## Problem Statement
Recommendation systems are crucial to the services we use on the internet every day. I wanted to create a system of my own to create a system to find solutions to the problems these systems run into today.

## Summary of Project
The project was primarily about developing a good model and implementing the model on a Streamlit app. The data itself was relatively straight forward, it consisted of one-million user reviews of movies, 6400 unique movies, and 3883 unique users. The ratings are on a scale of 1-5 with 5 meaning the user enjoyed the movie. The main issue with the data was that not every movie was accounted for as a rating. To compensate for this I created a fake user to apply the average rating to the movies that had not been represented. This allowed the model to work more comprehensively with the data, but ultimately was not absolutely necessary. The issue it rose  was during the joining of our data frames, another solution would be to use a different type of join.
Modeling was done in two-parts. First a content based model was created by judging the cosine similarity of genres, after turning genres into Tf-IDF vectors. This could be considered overkill but it did give our content model more robust capabilities because if we added new movies of genres that haven't yet been accounted for, our model would not break. There was another 'soft' model that was applied, which was a system facing content filter. It involved ranking every movie in order of popularity as to do two things, 1) it prevented from only feeding popular movies to the user, 2) it broke up the size of arrays we would feed to our collaborative model. 
Second, a collaborative model was implemented to predict user ratings of movies. There were a couple of ways to fundamentally slice this. Since ratings are either 1, 2, 3, 4, or 5 we could treat this problem as a classification problem. The issue with implementing a classification model is that it in some ways fundamentally misses the actual goal of the collaborative model. The purpose of the model is to sort movies for a user in the order that they are most likely to enjoy them. One problem with our classification would be that most movies are rated as 4, or 5. Meaning that a user is likely to like the movies they have seen. If a classification system was implemented we would need to balance classes, which means the amount of fake data we created would be pretty massive. This is not impossible but it opens a can of worms when you don't need to. Implementing the model as a regression will not only solve our problem, but also be the path of least resistance. In a future implementation, I would consider implementing a soft-max model.
The model I finally went with was a artificial neural network and a Keras matrix-factorizer. The matrix factorizer was simple and easy to run not on a TPU. This is what was ultimately implemented in the app because of the speed it could run on a CPU, and its modest results. The Artificial Neural Network took the idea of matrix factorization, but instead of using a dot-product would create a long vector based on all the inputs, and their biases, then fed that to a Neural network that would predict the rating by squishing the final output to a sigmoid function then use a Lambda Layer to multiply the final result by 5.
Implementation was done simply by creating a user signup screen, where we would have them select 3 genres they enjoyed. They would then be directed to a content filter that is trying to generate ratings from them by filtering them movies similar to the genres they chose. Finally we would use the collaborative model to filter them movies they are likely to enjoy from multiple ends of the popularity filter.

## Datasets Used
- `/main/combined.csv` combined data set of all ratings, users, and movies
- `/main/processed_movies.csv` all movies
- `/main/processed_ratings.csv` all ratings
- `/main/processed_users.csv` all users

## Data
|Feature|Type|Dataset|Description|
|---|---|---|---|
|movie_id|int|`combined.csv`|Movie Identification Number|
|title|string|`combined.csv`|Title of the movie|
|genres|string|`combined.csv`|Genre of the movie|
|user_id|int|`combined.csv`|User Identification Number|
|rating|int|`combined.csv`|Rating on a scale of 1-5 of how much the user liked the movie|
|timestamp|int|`combined.csv`|Timestamp of rating|
|gender|string|`combined.csv`|Gender of user M or F|
|age|int|`combined.csv`|Numeric ID of Age category of user|
|occupation|int|`combined.csv`|Identification of Occupation Category|
|zip|int|`combined.csv`|postal code of user|

## Conclusion
I believe that it is possible to implement a hybrid recommendation system on Streamlit, although a different app platform would be much better. Most of the issues I ran into had to deal with the shear size of the data, and the lack of tensor processing capacity a cpu from 5 years ago has. 




