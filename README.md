# Assignment 4 - Using finetuned transformers via HuggingFace

The assignment focuses on feature extraction using Emotion Classification. The classification will be done on the ```fake_or_real_news``` dataset, and the objective is to present the results of the emotion classification, that is the distribution of different emotions across fake and real news headlines, in a meaningful way using tables and visualisations, where possible similarities and differences will be discussed further below. 

## Tasks
The tasks for this assignment are to:

-	Create a pipeline for Emotion Classification on every headline in the data using ```HuggingFace``` 
-	Create both meaningful and readable tables and visualisations for the distribution of emotions across:
o	All of the news headlines in the data
o	Only the real news headlines in the data
o	Only the fake news headlines in the data
-	Discuss and compare possible key differences found between the distributions

## Repository content
The GitHub repository consists of four folders; The ```src``` folder, which contains the Python script for the Emotion Classification and the creation of tables and visualisations, the ```out``` folder, which contains the tables of emotion distributions across the headlines, the ```models``` folder, which contains the visualisations of the emotion distribution across the headlines, and the ```in``` folder containing the provided corpus used in the assignment. 
Furthermore, the repository contains a ```ReadMe.md``` file, as well as files for ```setup.sh``` and ```requirements.txt```. 

## Data
The corpus used for this assignment is a single CSV file, which contains the text from several news articles, their headlines, and whether this article is real or fake. This assignment takes use of the headlines and the label pertaining hereto.  To download the data, please following this link:

https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news

To access and prepare the data for use in the script, please; Create a user on the website, download the data, and import the data into the ```in``` folder in the repository. The script will call the data set as “fake_or_real_news”, and I would recommend changing the name to this before proceeding with running the script.

## Methods
The following is a description of parts of my code where further explanation of my decisions on arguments and functions may be needed.

To be able to classify and extract emotions based on headlines in the dataset, and also separate these by whether they are real or fake, I first read in the data and loaded the classifier. When loading the emotion classifier, please know that the model “j-hartmann/emotion-english-distilroberta-base” was chosen due to previous experience with it in the course, along with it being recommended by the course instructor. Furthermore is the argument “return_all_scores” in the classifier set to “False” as this ensures, that the model only returns the most likely predicted emotion for each headline, which is assumed to be the correct label.

After creating a Pandas series for each of the three “categories” (all headlines, only real headlines, and only fake headlines), which allows me to divide each emotion into its own row with its own count, I am able to create tables for each of them and save these as an output. 

To create visualisations for each of the categories and their emotion distribution, I make use of matplotlib and arguments pertaining hereto, and in order to make these I first create an empty dictionary to contain the different emotions, and then use this to count instances across the different headline categories. 

## Usage
### Prerequisites and packages
To be able to reproduce and run this code, make sure to have Bash and Python3 installed on whichever device it will be run on. Please be aware, that the published script was made and run on a MacBook Pro from 2017 with the MacOS Ventura package, and that all Python code was run successfully on version 3.11.1.

The repository will need to be cloned to your device. Before running the code, please make sure that your Bash terminal is running from the repository; After, please run the following from the command line to install and update necessary packages:

    bash setup.sh


### Running the script
My system requires me to type “python3” in the beginning of my commands, and the following is therefor based on this. To run the script from the command line please be aware of your specific system, and whether it is necessary to type “python3”, “python”, or something else in front of the commands. Now run:

    python3 src/emotion_clf.py

This will active the script. When running, it will go through each of the functions in the order written in my main function. That is:

-	Processing and classifying the data for each of the three categories. The function will find the data in the ```in``` folder, where it is already placed, but the step on where and how to download it yourself is described further up in the ReadMe. 
-	Generate crosstab tables and then saving these as CSV files to the ```out``` folder.
-	Create a dictionary for all emotions in the classifier, which is then used in the bar chart for each category to showcase the distribution across emotions.
-	Saving the visualisations to the ```models``` folder. 

## Results
From both the tables and bar charts I am able to see some key differences in the distribution of emotions across the real and fake headlines. Across all headlines the most dominating emotion is “Neutral”, which can both be observed from the bar chart as well as its actual count in the table. As it is assumed that the given emotions label for each headline is the correct one, this could be interpreted as the model assigning this emotion to headlines not fitting to any of the other emotions in the model. 

The other most noticeable difference is that the fake news headlines has higher numbers in 4 of the seven categories, and is almost equal in another one. It could be seen as if the fake headlines have been made with the aim of trying to invoke emotions, whereas the real headlines more often try to remain neutral in their wording. 
