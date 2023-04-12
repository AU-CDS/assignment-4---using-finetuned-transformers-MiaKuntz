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
    # choosing only "title" column
    titles = news_df["title"]
    # using the model emotion english... for text classification pipeline
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        # getting the top score for each headline
                        return_all_scores=False)
    for headline in titles:
        emotion_clf = classifier(headline)
    return emotion_clf

# defining main function
def main():
    # processing
    emotion_clf = clf_data

if __name__=="__main__":
    main()  

# i need to filter in the output of the classifier, so that the table is "pretty"
# it is a list with a dictionary
# do this after the emotion classification
# for output in results:
    # label = output[0]["label"]
