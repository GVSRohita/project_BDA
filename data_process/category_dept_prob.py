import os
import pandas as pd

from utils.file_utils import write_json_dict, read_json
import numpy as np

root_dir = "/home/charan/DATA/311_Data/"
dept_path = os.path.join(root_dir, "Department/department.csv")
prob_path = os.path.join(root_dir, "Problem/category.csv")
df_path = os.path.join(root_dir, "311_VIZ_DESCRIPTION.csv")
dept_json = os.path.join(root_dir, 'Department/parent_map.json')
prob_json = os.path.join(root_dir, 'Problem/parent_map.json')
dept_class_json = os.path.join(root_dir, 'Department/class.json')
prob_class_json = os.path.join(root_dir, 'Problem/class.json')
dept_parent = read_json(dept_json)
prob_parent = read_json(prob_json)
parent_df_path = os.path.join(root_dir, "311_VIZ_DESCRIPTION_PARENT.csv")
dept_class_dict = read_json(dept_class_json)
prob_class_dict = read_json(prob_class_json)
class_df_path = os.path.join(root_dir, "311_VIZ_DESCRIPTION_PARENT_CLASS.csv")


def class_json(input, output):
    class_dict = {}
    counter = 0
    for key in list(input.keys()):
        class_dict[str(counter)] = key
        class_dict[key] = counter
        counter += 1
    write_json_dict(class_dict, output)


def prepare_dept_data():
    print("processing")
    dept_df = pd.read_csv(dept_path)
    dept_dict = {}
    for index, row in dept_df.iterrows():
        if not row['Group_DEPT'] in dept_dict:
            dept_dict[row['Group_DEPT']] = []
        dept_dict[row['Group_DEPT']].append(row['DEPT'])
    write_json_dict(dept_dict, dept_json)


def prepare_prob_data():
    print("processing")
    prob_df = pd.read_csv(prob_path)
    dept_dict = {}
    for index, row in prob_df.iterrows():
        if not row['group_category'] in dept_dict:
            dept_dict[row['group_category']] = []
        dept_dict[row['group_category']].append(row['category'])
    write_json_dict(dept_dict, prob_json)


def apply_dept_parent(input):
    for each in list(dept_parent.keys()):
        if any(input in dept for dept in dept_parent[each]):
            return each
    return np.nan


def apply_prob_parent(input):
    for each in list(prob_parent.keys()):
        if any(input in category for category in prob_parent[each]):
            return each
    return np.nan


def parent_mapping():
    data_311 = pd.read_csv(df_path)
    data_311["PARENT_DEPT"] = data_311.DEPARTMENT.apply(apply_dept_parent)
    data_311["PARENT_CATEGORY"] = data_311.CATEGORY.apply(apply_prob_parent)
    data_311.to_csv(parent_df_path, index=False)


def class_mapping():
    class_json(dept_parent, dept_class_json)
    class_json(prob_parent, prob_class_json)


def df_class_map():
    parent_df = pd.read_csv(parent_df_path)
    parent_df["DEPT_CLASS"] = parent_df["PARENT_DEPT"].apply(lambda x: dept_class_dict[x])
    parent_df["PROB_CLASS"] = parent_df["PARENT_CATEGORY"].apply(lambda x: prob_class_dict[x])
    parent_df.to_csv(class_df_path, index=False)


if __name__ == '__main__':
    final_df = pd.read_csv(class_df_path)
    print("finished")
