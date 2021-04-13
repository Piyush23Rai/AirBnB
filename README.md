
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*. 

## Project Motivation<a name="motivation"></a>

For this project, I was interestested in using AirBnB open data for two US cities Boston and Seattle to better understand:

1. How does Listings price relate to the month of the year?
2. How do Amenities relate to the Prices of the listings?
3. What are the top 5 amenities widely present across listings in both the cities?
4. Do the same set of top amenities also have the potential to increase the prices of any listing?
5. What is the best property in terms of Prices for the hosts?
6. Does same property will get me a higher price if put out on rent for a week or month?

Also, I have built a price model where in I have selected features by experimenting with different methods.


## File Descriptions <a name="files"></a>

There are 4 notebooks available here to showcase work related to the above questions.  Each of the notebooks is exploratory in searching through the data pertaining to the questions showcased by the notebook title.  Markdown cells were used to assist in walking through the thought process for individual steps.  

There is an additional `.py` file that runs the necessary code to obtain the final model used to predict price and contains other helper functions uses while wrangling and cleaning data.

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://winxzenpiyush23.medium.com/whats-the-idea-of-one-perfect-stay-a38a0974c6eb).

The Price model however does not have a decent accuracy. Partly due to the fact, that I have neglected text based columns which require Sentiment based Analysis.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to AirBnb for the data.  You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/airbnb/seattle/data) and [here](https://www.kaggle.com/airbnb/boston).

Also, while building the model, the methods I have used for Feature Extraction were inspired from [here](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b) and [here](https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf)
Otherwise, feel free to use the code here as you would like! 

