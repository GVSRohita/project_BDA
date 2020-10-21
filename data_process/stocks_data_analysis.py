import os
from data_process.news_data_analysis import get_list_news, get_news_article
from utils.file_utils import write_json_dict

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/BDA_Project/"
etf_data = os.path.join(root_dir, "stocks_data/ETFs")
stocks_data = os.path.join(root_dir, "stocks_data/Stocks")

etf_firms = os.listdir(etf_data)
stocks_firms = os.listdir(stocks_data)

etf_firms = list(set(etf_firms))
stocks_firms = list(set(stocks_firms))
etf_firms = [each.replace(".us.txt", "").upper() for each in etf_firms]
stocks_firms = [each.replace(".us.txt", "").upper() for each in stocks_firms]
news_firms = get_list_news()

compare_dict = {}

for each_firm in news_firms:
    for each in stocks_firms:
        if len(each_firm) > 1 and len(each) > 1:
            first_char = each_firm[0]
            # if each[0] == first_char:
            if each == each_firm:
                print(f'{each_firm}__{each}')
                # news_content = get_news_article(each_firm)
                compare_dict[each_firm] = each
write_json_dict(compare_dict, 'Review.json')
