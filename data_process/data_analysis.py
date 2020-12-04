import os
import pandas as pd
import matplotlib.pyplot as plt

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/BDA_Project"
news_data = "news_data/merged_final_news.csv"
stock_data = "stocks_data/final_stock_consolidated.csv"
sentiment_data = "Sentiment_Financial_Data/all-data.csv"

sentiment_data = os.path.join(root_dir, sentiment_data)

news_data = os.path.join(root_dir, news_data)
stock_data = os.path.join(root_dir, stock_data)
plots_path = os.path.join(root_dir, 'plots')

news_data_df = pd.read_csv(news_data)
# news_data_df.groupby(by=["ticker"])["id"].count().to_csv("groupby")
stock_data_df = pd.read_csv(stock_data)
# stock_data_df.groupby(by=["stock_ticker"])["Open"].count().to_csv("groupby")

sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}


def apply_date(input):
    list_input = input.split("-")
    total = 0
    for each in list_input:
        total += int(each)
    return total


def process_data():
    counter = 0

    for each in list(news_data_df.ticker.unique()):
        filtered_stocks = stock_data_df[stock_data_df.stock_ticker == each]
        filtered_opinions = news_data_df[news_data_df.ticker == each]
        filtered_opinions.classification = filtered_opinions.classification.apply(lambda x: sentiment_map[x])
        filtered_opinions['sequence'] = filtered_opinions.release_date.apply(apply_date)
        filtered_stocks['sequence'] = filtered_stocks.Date.apply(apply_date)
        merged = pd.merge(filtered_opinions, filtered_stocks, how='inner', on=['sequence', 'sequence'])
        merged.drop_duplicates(subset=['sequence'], inplace=True)
        merged.sort_values(by=['sequence'], inplace=True)
        fig, axs = plt.subplots(2)
        fig.suptitle(f'{each} - Report Stock Data VS New Sentiments')
        axs[0].plot(merged['sequence'], merged['Open'])
        axs[1].plot(merged['sequence'], merged['classification'])
        each_plot_path = os.path.join(plots_path, each + ".png")
        plt.savefig(each_plot_path)
        counter += 1
        if counter >= 20:
            break


def analyze_data_balance():
    sentiment_df = pd.read_csv(sentiment_data, encoding="ISO-8859-1", header=None)
    sentiment_df.rename(columns={0: 'label', 1: 'desc'}, inplace=True)
    grouped_df = sentiment_df.groupby(by=["label"])["desc"].count()
    grouped_df.plot.bar(rot=0)
    plt.savefig("imbalanced.png")
    plt.show()


if __name__ == '__main__':
    analyze_data_balance()
