import json
from data_process.news_data_analysis import load_dataframe
import pandas as pd
import os

from utils.file_utils import read_json

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/BDA_Project"
mapping_dict = os.path.join(root_dir, "Sentiment_Financial_Data/label_mapping_dict.json")
news_data_path = os.path.join(root_dir, "news_data/news_with_summary.csv")
news_classify_path = os.path.join(root_dir, "news_data/news_classification.csv")
classification_processed = os.path.join(root_dir, "news_data/processed_sentiments.csv")
merged_final = os.path.join(root_dir, "news_data/merged_final_news.csv")

mapping_dict = read_json(mapping_dict)


def final_review_companies():
    with open('Review.json', 'r') as f:
        companies_dict = json.load(f)
        f.close()
    return companies_dict


def extract_news_company_data():
    companies = final_review_companies()
    news_data = load_dataframe()
    list_companies = list(set(companies.keys()))
    df_dict = {}
    for each_company in list_companies:
        print(f'processing {each_company}')
        df_dict[each_company] = news_data[news_data.ticker == each_company].reset_index()
    final_df = pd.concat(list(df_dict.values())).drop(columns=['index'])
    final_df.to_csv("final_news_data_processed.csv", index=False, header=True)


def prepare_news_data():
    news_data = pd.read_csv(news_data_path)
    classification_df = news_data[['id', 'summary']]
    classification_df.rename(columns={'summary': 'desc', 'id': 'u_id'}, inplace=True)
    classification_df.to_csv(news_classify_path, header=True, index=False)


def merge_final_output():
    summary_df = pd.read_csv(news_data_path)
    summary_df = summary_df.loc[:, ~summary_df.columns.str.contains('^Unnamed')]
    sentiments_df = pd.read_csv(classification_processed)
    summary_df = summary_df.merge(sentiments_df, on=['id'])
    summary_df['classification'] = summary_df['classification'].astype("int64")
    summary_df['classification'] = summary_df['classification'].apply(lambda x: mapping_dict[str(x)])
    summary_df.to_csv(merged_final, index=False, header=True)


if __name__ == '__main__':
    merge_final_output()
