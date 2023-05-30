# Assignment 4 - Using finetuned transformers via HuggingFace
This assignment focuses on feature extraction using Emotion Classification. The classification will be done on the ```fake_or_real_news``` dataset, and the objective is to present the results of the emotion classification, that is the distribution of different emotions across fake and real news headlines, in a meaningful way using tables and visualisations, where possible similarities and differences will be discussed further below. 

## Tasks
The tasks for this assignment are to:
-	Create a pipeline for Emotion Classification on every headline in the data using ```HuggingFace```.
-	Create both meaningful and readable tables and visualisations for the distribution of emotions across:
o	All of the news headlines in the data.
o	Only the real news headlines in the data.
o	Only the fake news headlines in the data.
-	Discuss and compare possible key differences found between the distributions.

## Repository content
The GitHub repository contains four folders, namely the ```in``` folder, where the dataset for this assignment can be found, the ```models``` folder, which contains visualisations of the emotion distribution across the headlines as histograms, the ```out``` folder, which contains the tables of emotion distributions across the headlines, and the ```src``` folder, which contains the Python script for the emotion classifier. Additionally, the repository has a ```ReadMe.md``` file, as well ```setup.sh``` and ```requirements.txt``` files.

### Repository structure
| Column | Description|
|--------|:-----------|
| ```in``` | Folder containing the ```fake_or_real_news``` dataset |
| ```models``` | Folder containing histograms of emotions |
| ```out``` | Folder containing the emotions crosstab tables |
| ```src```  | Folder containing Python script for emotion classification |

## Data
The corpus used for this assignment is a single CSV file, which contains the text from 6,335 news articles, their headlines, and the label to either classify the article as real or fake. To download the data, please follow this link:

https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news

To access and prepare the data for use in the script, please; Create a user on the website, download the data, and move it to the ```in``` folder in the repository. The script will call the data set ```fake_or_real_news```, and I would recommend changing the name to this before proceeding with running the script.
As this dataset was provided by the course instructor for this assignment, I have chosen to leave it in the repository, and the above guide to downloading and preparing the data for use is included for reproducibility purposes.  

## Methods
The following is a description of parts of my code where additional explanation of my decisions on arguments and functions may be needed than what is otherwise provided in the code. 

To be able to classify and extract emotions based on headlines in the dataset, and also separate these by whether they are real or fake, I first read the data and load the classifier. When loading the emotion classifier, please know that the model “j-hartmann/emotion-english-distilroberta-base” was chosen due to previous experience with it in the course, along with it being recommended by the course instructor. Furthermore is the argument “return_all_scores” in the classifier set to “False” as this ensures that the model only returns the most likely predicted emotion for each headline, which is assumed to be the correct label.

After creating a Pandas series for each of the three “categories” (all headlines, only real headlines, and only fake headlines), which allows me to divide each emotion into separate rows with separate counts, the script then generates tables for each of them and save these as an output. 

To create visualisations for each of the categories and their emotion distribution, I make use of matplotlib and arguments pertaining hereto, and to make these I first create an empty dictionary to contain the different emotions, and then use this to count instances across the different headline categories. 

## Usage
### Prerequisites and packages
To be able to reproduce and run this code, make sure to have Bash and Python3 installed on whichever device it will be run on. Please be aware that the published script was made and run on a MacBook Pro from 2017 with the MacOS Ventura package, and that all Python code was run successfully on version 3.11.1.

The repository will need to be cloned to your device. Before running the code, please make sure that your Bash terminal is running from the repository; After, please run the following from the command line to install and update the necessary packages:

    bash setup.sh

### Running the script
My system requires me to type “python3” at the beginning of my commands, and the following is therefore based on this. To run the script from the command line please be aware of your specific system, and whether it is necessary to type “python3”, “python”, or something else in front of the commands. Now run:

	python3 src/emotion_clf.py

This will activate the script. When running, it will go through each of the functions in the order written in my main function. That is:
-	Processing and classifying the data for each of the three categories. 
-	Generating crosstab tables and then saving these as CSV files to the ```out``` folder.
-	Creating a dictionary for all emotions in the classifier, which is then used in the histograms for each category to showcase the distribution across emotions.
-	Saving the visualisations to the ```models``` folder. 

## Results
Both the tables and histograms show differences in their distribution of emotions across the real and fake headlines. From the tables containing the number of real and fake headlines, I can calculate the total in each category, which is 3,171 and 3,164 respectively. I see this as a near-perfect 50/50 distribution, which makes it easier to compare the results. 

Across all headlines, the most dominating emotion is “Neutral”, which can both be observed from the histogram as well as its actual count in the table. As it is assumed that the given emotions label for each headline is the correct one, this could be interpreted as the model assigning this emotion to headlines not fitting any of the other emotions in the model. The other most noticeable difference is that the fake news headlines have higher numbers in 4 of the seven categories, and is almost equal in another one when compared to the real news headline. It could be interpreted as if the fake headlines have been made to try to invoke emotions, whereas the real headlines more often try to remain neutral in their wording, as the real news headlines have 1,649 counts of neutral headlines, whereas the fake news headlines only have 1,531 instances. 

## References
Jochen Hartmann, "Emotion English DistilRoBERTa-base". https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022.
