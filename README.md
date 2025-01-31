# Hacker News Post Classification system

## Introduction
Built an application that classifies Hacker News data into corresponding post types.
* Used the HN posts from 2018 and 2019 (2,709,143 posts) as training/ testing set.
* Developed a Naive Bayes with ~92% accuracy in prediction results.

## Tools and Technology
* [nltk](https://www.nltk.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)

## Screenshots
* ![](Screenshots/Capture2.PNG)
* ![](Screenshots/Capture3.PNG)
* ![](Screenshots/Capture4.PNG)

## Results
![](Screenshots/Capture1.PNG)
![](Screenshots/Capture5.PNG)
![](Screenshots/Capture6.PNG)

## Running instructions:-

To run the code:
* Run 'runthis.py'
* Enter the name of input csv file
   Please note that the input file MUST be located in 'Resources' folder
* Enter choice according to the console 
* Follow prompts
* Check the "Output" folder for textfile results lie model,scores etc.*
* Check "Output\frequency_filter_output" folder contains output for Infrequent word filtering (Experiment 4)
* Check "Output\smooth_filter_output" folder contains output for smoothing output (Experiment 5)

## Future Scope
* Classify other websites
* Take user defined classifications
