# Movie Recommender System
A project for a lecture in "Decision Support Systems". Written in Python.

This Recommender system consists of:
- Random Predictor: Predicts random ratings for movies. Used as a base predictor to compare
- Views Predictor: Recommends movies that are watched the most. The safest option and not personalized.
- Averag ePredictor: Recommends movies using the adjusted average formula. Unpersonalized.
- ItemBased Predictor: Recommends movies using the item-based collaborative filtering. Calculates similarites between movies and using the weighted average combined with the user's ratings, recommends movies the user would enjoy. Personalized.
- SlopeOne Predictor: Uses the Slope One technique. Predicts the rating the user would give a certain movie by combining his ratings for other movies with deviations between movies (how much better/worse the movie is rated on average compared to another movie)

The Recommend system comes with an evaluation for predictors.
It uses:
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- Precision: How many movies has the user watched out of those that were recommended to him
- Recall: How many movies were recommended to him that were relevant and were watched (how many movies we missed and the user watched something else, or in other words: does the user watch only recommended movies, which is good for us)
- F Score: Combines Precision and Recall to give an overall score.


