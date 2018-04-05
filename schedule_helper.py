import nltk
import json
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import data_converter as dc
import train_model_nn as tm



""" Settings """
def download_nltk(ROOT) :
    path = ROOT + "nltk_tools/"
    nltk.download('punkt', download_dir=path)
    nltk.download('wordnet', download_dir=path)
    nltk.download('stopwords', download_dir=path)



""" Functions """
def load_data (fp) :
    data_pd = pd.read_excel(fp, header=None)
    data_pd = data_pd.fillna(value="")
    return data_pd


def produce_top_sen_dict(top_sen_list) :
    top_sen_dict = dict()
    top_sen_dict["size"] = len(top_sen_list) #excluding -1 label
    top_sen_dict["list"] = top_sen_list
    top_sen_dict["more"] = ""
    return top_sen_dict


def produce_sen_dict (sen_list) :
    sen_dict = dict()
    sen_dict["size"] = len(sen_list) #excluding -1 label
    sen_dict["list"] = sen_list
    sen_dict["more"] = ""
    return sen_dict


def produce_top_index_dict (top_index_list) :
    top_index_dict = dict()
    top_index_dict["size"] = len(top_index_list) #excluding -1 label
    top_index_dict["list"] = top_index_list
    top_index_dict["more"] = ""
    return top_index_list


def train_toggle (data, data_threshold) :
    """
    To check whether there are abundant valid amount of data.
    :param data: [np.array] in the format,
                            [[.., ..., ......, label_index], ......, [......]]
    :param data_threshold: [int] how many valid data is enough
    :return: [bool]
    """
    count_valid = 0
    for sample in data :
        if sample[-1] != 0 : count_valid += 1

    if count_valid >= data_threshold : return True
    else : False


def save_json (data_dict, fn) :
    data_json = json.dumps(data_dict)
    with open(fn, "w") as f :
        json.dump(data_json, f)


def save_data (ROOT, model_index, data) :
    fp_data = ROOT + "/data/data_%s.npy" % model_index
    np.save(fp_data, data)



""" Main """
if __name__ == "__main__" :
    ### Parser
    parser = ArgumentParser()
    parser.add_argument("-d", action="store_true", default=False, help="Whether download nltk tools.")
    download = parser.parse_args().d


    ### Params
    ROOT = "./"
    size_col, size_row = 24, 7
    label_limitation = 10
    data_threshold = 5


    ### Download and load NLTK tools
    if download :
        download_nltk(ROOT)
    nltk.data.path.append(ROOT + "nltk_tools")


    ### Load data
    fp_data = ROOT + "/data/data_schedule.xlsx"
    data_pd = load_data(fp_data)
    data = dc.convert_to_format_data(data_pd)


    ### NLP
    dc.nlp(data)
    top_words = dc.top_freq_words(data)
    data_encoded, top_index_list, top_sen_list, sen_list = dc.schedule_encoding(data, top_words, label_limitation)


    ### Information saving
    fn_top_index_json = ROOT + "/top_info/top_index.json"
    fn_top_sen_json = ROOT + "/top_info/top_sen.json"
    fn_sen_json = ROOT + "/top_info/sen.json"

    top_index_dict = produce_top_index_dict(top_index_list)
    top_sen_dict = produce_top_sen_dict(top_sen_list)
    sen_dict = produce_sen_dict(sen_list)

    save_json(top_index_dict, fn_top_index_json)
    save_json(top_sen_dict, fn_top_sen_json)
    save_json(sen_dict, fn_sen_json)


    ### Models training and saving
    save_toggle = input("Do you want to save the data into .npy this time ? (y/n) : ")
    save_toggle = True if save_toggle == "y" else False

    print ("___Model Training___")

    for model_index in range(size_col * size_row):   #(tmp)
        print ("Model %s is traning ..." % model_index)
        data_train = dc.convert_to_train_data(data_encoded, model_index, top_index_list)
        if save_toggle :
            save_data(ROOT, model_index, data_train)
        if train_toggle(data_train, data_threshold) :
            max_labels = label_limitation   #for demo
            tm.model_training(data_train, model_index, max_labels)
        print ("... 100 %")

    print("___Model Training Finished___")