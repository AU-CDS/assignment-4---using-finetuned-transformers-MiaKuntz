# importing pipeline
from transformers import pipeline
# data processing tools
import os
import pandas as pd
# importing plotting tool
import matplotlib.pyplot as plt

# defining function for classifying data
def clf_data(): 
    # creating filepath
    data_file = os.path.join("in/fake_or_real_news.csv")
    # reading in news data
    news_df = pd.read_csv(data_file)
    # choosing only "title" column
    headlines = news_df["title"]
    # using the model emotion english for text classification pipeline
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        # getting the top score for each headline
                        return_all_scores=False)
    # classifying all headlines
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

def plot_emotions(input_emotions):
# plot for headline emotion distribution
    # creating dictionary
    emotions_dict = {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "neutral": 0, "sadness": 0, "surprise": 0}
    # counting instances across headline categories
    for emotion in input_emotions:
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
    # processing
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
