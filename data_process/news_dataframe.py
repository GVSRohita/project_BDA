import json
from data_process.news_data_analysis import load_dataframe
import pandas as pd


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


if __name__ == '__main__':
    # extract_company_data()
    final_dct = pd.read_csv('final_news_data_processed.csv')
    print('processed')
