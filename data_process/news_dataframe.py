import json


def final_review_companies():
    with open('Review.json', 'r') as f:
        companies_dict = json.load(f)
        f.close()
    return companies_dict


if __name__ == '__main__':
    companies = final_review_companies()
    list_companies = list(companies.keys())
    print("processed")
