# importing pipeline
from transformers import pipeline
# data processing tools
import os
import pandas as pd

# defining function for classifying data
def clf_data(): 
    # creating filepath
    data_file = os.path.join("in/fake_or_real_news.csv")
    # reading in news data
    news_df = pd.read_csv(data_file)
    # choosing randon subset of data
    rd_news_df = news_df.sample(n=100)
    # choosing only "title" column
    titles = rd_news_df["title"]
    # choosing only real news titles
    real_titles = rd_news_df.loc[rd_news_df["label"]=="REAL", "title"]
    # choosing only fake news titles
    fake_titles = rd_news_df.loc[rd_news_df["label"]=="FAKE", "title"]
    # using the model emotion english for text classification pipeline
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        # getting the top score for each headline
                        return_all_scores=False)
    # classifying all titles (real and fake)
    emotion_all = []
    for headline in titles:
        emotion_all.append(classifier(headline))
    # classifying only real titles
    emotion_real = []
    for headline in real_titles:
        emotion_real.append(classifier(headline))
    # classifying only fake titles
    emotion_fake = []
    for headline in fake_titles:
        emotion_fake.append(classifier(headline))
    return emotion_all, emotion_real, emotion_fake

# defining main function
def main():
    # processing
    emotion_all, emotion_real, emotion_fake = clf_data()
    print(emotion_all, emotion_real, emotion_fake)

if __name__=="__main__":
    main()  

# i need to filter in the output of the classifier, so that the table is "pretty"
# it is a list with a dictionary
# do this after the emotion classification
# for output in results:
    # label = output[0]["label"]
