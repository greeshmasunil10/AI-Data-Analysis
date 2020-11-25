# Hacker News Post Classification system

## Introduction
A program that classifies Hacker News data into corresponding post types.
* Used the HN posts from 2018 and 2019 (2,709,143 posts) as training/ testing set.
* Developed a Naive Bayes with ~92% accuracy in prediction results.

## Screenshots


## Running instructions:-

To run the code:
* Run 'runthis.py'
* Enter the name of input csv file
   Please note that the input file MUST be located in 'Resources' folder
* It is recommened to run input with  size~100 for least waiting time
* Enter choice according to the console 
* Check the console for testing results
* Check the "Output" folder for textfile results lie model,scores etc.*
* Check "Output\frequency_filter_output" folder contains output for Infrequent word filtering (Experiment 4)
* Check "Output\smooth_filter_output" folder contains output for smoothing output (Experiment 5)


## Tools and Technology
* [nltk](https://www.nltk.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)


