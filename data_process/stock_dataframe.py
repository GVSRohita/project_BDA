from data_process.news_dataframe import final_review_companies
import os
import json
import pandas as pd

from utils.file_utils import write_json_dict

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/BDA_Project/"
stocks_data = os.path.join(root_dir, "stocks_data/Stocks")


def extract_stocks_related_files():
    company_file = {}
    companies = final_review_companies()
    list_companies = list(companies.values())
    for each_file in os.listdir(stocks_data):
        firm_name = each_file.replace(".us.txt", "").upper()
        if firm_name in list_companies:
            company_file[firm_name] = each_file
    write_json_dict(company_file, 'company_to_file.json')


def create_dataframe_stocks():
    stocks_df_dict = {}
    with open('company_to_file.json', 'r') as f:
        company_dict = json.load(f)
    for each in sorted(list(company_dict.keys())):
        print(f'Currently processing{each}')
        file_name = company_dict[each]
        file_name = os.path.join(stocks_data, file_name)
        company_stock_df = pd.read_csv(file_name)
        company_stock_df['stock_ticker'] = each
        stocks_df_dict[each] = company_stock_df
        print(f'Currently processing{each}' + str(company_stock_df.shape))
    final_stock_data = pd.concat(list(stocks_df_dict.values()))
    final_stock_data.to_csv('final_stock_consolidate.csv', header=True, index=False)


if __name__ == '__main__':
    create_dataframe_stocks()
    print("Processed!")
