# Customer-Purchase-Predictor
This model is based on Decision Tree Classifier and uses the data based on age and salary of the people to predict how likely they are to buy a particular product.

The maximum depth of the tree is kept to be 3, as there are less number of features, and this will also decrease the computation time, making the model overall more efficient.

Here, I use a confusion matrix to compute the accuracy with which it was able to predit the data, along with its seaborn heatmap for better visualisation and understanding.

Although we can plot the tree using Scikit-Learn, I have used GraphViz here for better visualisation of the Decision Tree.
