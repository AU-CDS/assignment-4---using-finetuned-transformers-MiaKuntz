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
    # choosing random subset of data
    rd_news_df = news_df.sample(n=100)
    # choosing only "title" column
    titles = rd_news_df["title"]
    # using the model emotion english for text classification pipeline
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        # getting the top score for each headline
                        return_all_scores=False)
    # classifying all titles
    emotions = [result['label'] for result in classifier(titles.tolist())]
    # creating pandas series for emotion distribution across all articles
    all_emotions = pd.Series(emotions, index=rd_news_df.index, name="Emotion")
    # creating pandas series for emotion distribution across only real articles
    real_mask = rd_news_df["label"]=="REAL"
    real_emotions = all_emotions.loc[real_mask].reset_index(drop=True).rename("Emotion")
    # creating pandas series for emotion distribution across only fake articles
    fake_mask = rd_news_df["label"]=="FAKE"
    fake_emotions = all_emotions.loc[fake_mask].reset_index(drop=True).rename("Emotion")
    return all_emotions, real_emotions, fake_emotions

# defining main function
def main():
    # processing
    all_emotions, real_emotions, fake_emotions = clf_data()
    # creating crosstab tables for each emotion distribution
    all_titles_table = pd.crosstab(index=all_emotions, columns="Count")
    real_titles_table = pd.crosstab(index=real_emotions, columns="Count")
    fake_titles_table = pd.crosstab(index=fake_emotions, columns="Count")
    # saving tables to csv files
    all_titles_table.to_csv("out/all_titles_table.csv")
    real_titles_table.to_csv("out/real_titles_table.csv")
    fake_titles_table.to_csv("out/fake_titles_table.csv")
    # plotting histogram for all articles
    plt.hist(all_emotions)
    plt.title("Distribution of emotions for all articles")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.savefig("models/all_emotions_histogram.png")
    plt.clf() # clearing figure
    # plotting histogram for real articles
    plt.hist(real_emotions)
    plt.title("Distribution of emotions for real articles")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.savefig("models/real_emotions_histogram.png")
    plt.clf() # clearing figure
    # plotting histogram for fake articles
    plt.hist(fake_emotions)
    plt.title("Distribution of emotions for fake articles")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.savefig("models/fake_emotions_histogram.png")
    plt.clf() # clearing figure

if __name__=="__main__":
    main()
