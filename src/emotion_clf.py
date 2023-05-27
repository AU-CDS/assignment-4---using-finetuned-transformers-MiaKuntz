# Author: Mia Kuntz
# Date hand-in: 31/5 - 2023

# Description: This script classifies the headlines in the fake_or_real_news.csv file.
# The script creates histograms for the distribution of emotions across all headlines, real headlines, and fake headlines. 
# The histograms are saved in the models folder. 
# The script also saves crosstab tables for the distribution of emotions across all headlines, real headlines, and fake headlines in the out folder.

# importing pipeline
from transformers import pipeline
# importing operating system
import os
# importing pandas
import pandas as pd
# importing plotting tool
import matplotlib.pyplot as plt

# defining function for processing data
def clf_data(): 
    # defining path to data file
    data_file = os.path.join("in/fake_or_real_news.csv")
    # reading data file
    news_df = pd.read_csv(data_file)
    # selecting only the headlines
    headlines = news_df["title"]
    # defining pipeline for emotion classification
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        # getting the top score for each headline
                        return_all_scores=False)
    # classifying emotions for each headline
    emotions = [result["label"] for result in classifier(headlines.tolist())]
    # creating pandas series for emotion distribution across all headlines
    all_emotions = pd.Series(emotions, index=news_df.index, name="Emotion")
    # creating pandas series for emotion distribution across only real headlines
    real_mask = news_df["label"]=="REAL"
    real_emotions = all_emotions.loc[real_mask].reset_index(drop=True).rename("Emotion")
    # creating pandas series for emotion distribution across only fake headlines
    fake_mask = news_df["label"]=="FAKE"
    fake_emotions = all_emotions.loc[fake_mask].reset_index(drop=True).rename("Emotion")
    return all_emotions, real_emotions, fake_emotions

# defining function for plotting emotions
def plot_emotions(input_emotions):
    # creating dictionary for counting instances of each emotion
    emotions_dict = {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "neutral": 0, "sadness": 0, "surprise": 0}
    # counting instances of each emotion
    for emotion in input_emotions:
        # adding 1 to the value of the emotion key
        emotions_dict[emotion] += 1
    # setting values variable
    values = list(emotions_dict.values())
    # setting names variable
    names = list(emotions_dict.keys())
    # setting arguments for bar plot
    plt.bar(names, values)
    # setting x axis label
    plt.xlabel("Emotion")
    # setting y axis label
    plt.ylabel("Count")

# defining main function
def main():
    # processing data
    all_emotions, real_emotions, fake_emotions = clf_data()
    # creating crosstab tables for each emotions distribution
    all_headlines_table = pd.crosstab(index=all_emotions, columns="Count")
    real_headlines_table = pd.crosstab(index=real_emotions, columns="Count")
    fake_headlines_table = pd.crosstab(index=fake_emotions, columns="Count")
    # saving tables to csv files
    all_headlines_table.to_csv("out/all_headlines_table.csv")
    real_headlines_table.to_csv("out/real_headlines_table.csv")
    fake_headlines_table.to_csv("out/fake_headlines_table.csv")
    # plotting histogram for all headlines
    plot_emotions(all_emotions)
    plt.title("Distribution of emotions for all headlines")
    plt.savefig("models/all_headlines_bars.png")
    plt.clf() # clearing figure
    # plotting histogram for real headlines
    plot_emotions(real_emotions)
    plt.title("Distribution of emotions for real headlines")
    plt.savefig("models/real_headlines_bars.png")
    plt.clf() # clearing figure
    # plotting histogram for fake headlines
    plot_emotions(fake_emotions)
    plt.title("Distribution of emotions for fake headlines")
    plt.savefig("models/fake_headlines_bars.png")

if __name__=="__main__":
    main()

# Command line argument:
# python3 emotion_clf.py