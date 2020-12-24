from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import os
import pandas as pd

under_sample = RandomUnderSampler(sampling_strategy='majority')
over_sample = RandomOverSampler(sampling_strategy='majority')


def func_under_sample():
    root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/Saria"
    data = "twitter_29_liwc_label.csv"
    data = os.path.join(root_dir, data)
    df_data = pd.read_csv(data)
    list_val = ["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]
    y_target = {}
    for each in list_val:
        y_target[each] = df_data[each]
    x_data = df_data.drop(columns=list_val)

    for each in list(y_target.keys()):
        print(f"processing {each}")
        x_under, y_under = under_sample.fit_resample(x_data, y_target[each])
        x_under[each] = y_under
        x_under.to_csv("twitter_uc_{}.csv".format(each), header=True, index=False)
        for inner_each in list(x_under[each].unique()):
            each_df = x_under[x_under[each] == inner_each]
            print(f"class - {inner_each} - shape {each_df.shape}")

    # x_under, y_under = under_sample.fit_resample(df_data['STATUS'].values.reshape(-1, 1),
    #                                              df_data['cOPN'].values.reshape(-1, 1))
    # under_sample_df = pd.DataFrame({"STATUS": x_under.flatten(), "cOPN": y_under.flatten()})
    # under_sample_df.to_csv("twitter_uc_cOPN.csv", header=True, index=False)
    # print('finished')
    # x_data = df_data.loc[:, df_data.columns != 'cOPN']
    # y_data = df_data['cOPN']
    # x_under, y_under = under_sample.fit_resample(x_data, y_data)
    # x_under['cOPN'] = y_under
    # x_under.to_csv("twitter_uc_cOPN.csv", header=True, index=False)
    print("finished")


def read_file():
    data = pd.read_csv('twitter_uc_cOPN.csv')
    print("finished")


if __name__ == '__main__':
    func_under_sample()
