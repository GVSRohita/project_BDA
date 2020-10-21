import pandas as pd
import os


def load_dataframe():
    root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/BDA_Project/"
    news_data = os.path.join(root_dir, "news_data/us_equities_news_dataset.csv")
    news_data = pd.read_csv(news_data)
    return news_data


def get_list_news():
    news_data = load_dataframe()
    news_data.ticker.unique().tofile("companies.csv", sep=',')
    return list(news_data.ticker.unique())


def get_news_article(name_firm):
    news_data = load_dataframe()
    firm_level = news_data[news_data['ticker'] == name_firm].reset_index()
    return firm_level.content[0]


if __name__ == '__main__':
    content = get_news_article('NIO')
    print('finished execution')
