from summarizer import Summarizer
import os
import pandas as pd

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/BDA_Project"
news_data = "news_data/us_equities_news_dataset.csv"
summarized_data = "news_data/summarized_data.csv"
news_data = os.path.join(root_dir, news_data)
summarized_data = os.path.join(root_dir, summarized_data)


def apply_summarizer(input_string):
    model = Summarizer()
    sentence_list = model(input_string, min_length=60)
    summarized_content = ''.join(sentence_list)
    return summarized_content


def process_dataframe():
    news_df = pd.read_csv(news_data)
    news_df["summary"] = news_df["content"].apply(apply_summarizer)
    news_df.to_csv(summarized_data, header=True, index=False)


if __name__ == "__main__":
    process_dataframe()
