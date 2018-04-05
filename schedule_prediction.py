import os.path
import json
import numpy as np
from sklearn.externals import joblib
import data_converter as dc



def load_json (fn) :
    data_json = None
    with open(fn, "r") as f :
        data_json = json.load(f)
    data_json = json.loads(data_json)
    return data_json


def predictions_to_schedules (predictions, size_col, size_row, top_sen_dict) :
    schedules = []
    memory = 0

    for i, pred in enumerate(predictions) :
        date, block = i // size_col, i % size_col
        if pred > 0 and memory != pred:
            schedule_new = dict()
            content = top_sen_dict["list"][pred-1]
            schedule_new["schedule_content"] = content
            schedule_new["label_index"] = pred
            schedule_new["date"] = date
            schedule_block_start = block
            memory = pred
        elif memory != 0 :
            schedule_block_end = block - 1
            schedule_new["time_start"] = schedule_block_start
            schedule_new["time_end"] = schedule_block_end + 1
            schedules.append(schedule_new)
            memory = 0

    return schedules


def produce_new_table (sample_table, date, time_start, time_end, sen, sen_dict, size_col=24, size_row=7) :
    """
    :param sample_table: [np.array] the original table/schedule for a week 
    :param date: [int] 1~7
    :param time_start: [int] 0~23 (for block not time)
    :param time_end: [int] 0~23 (for block not time)
    :param sen: [str] the content for the blocks in the schedule table
    :return: [np.array, shape=(size_row, size_col)] new schedule table
    """
    date = date - 1
    start, end = time_start, time_end - 1
    sen_list = sen_dict["list"]
    sen_index = dc.sen_encoding(sen, sen_list)
    if sen_index != None :
        for i in range(start, end + 1) :
            sample_table[date][i] = sen_index

    return sample_table


def convert_to_predicted_sample (sample, model_index) :
    """
    :param sample: [np.array, shape=(size_row, size_col)]
    :return: [np.array, shape=(1,None)] 
    """
    sample = list(sample.flatten())
    sample.pop(model_index)
    sample = np.array(sample)
    return sample


def prediction_decoding (y_prob, label_limitation) :
    """
    Convert the predicted vector into the index(label).
    Also, this would take complementary labels into consideration
    :param v: [np.array, shape=vector] the predicted vector
    :return: [int] the decoded index(label) from the input
                   if > 0  -> normal labels
                   if == 0 -> the label of no schedule
                   if < 0  -> the complementary label
    """

    pivot = 0.5
    value, power = 0, -1
    for i, prob in enumerate(y_prob) :
        power += 1
        if prob > pivot :
            value += 2 ** i

    if y_prob[-1] == 1 : # complementary label
        value = value - 2 ** power
        if value <= label_limitation :
            return - value
        else : return 0
    else :
        if value <= label_limitation :
            return value
        else : return 0


def model_prediction (x_train, fn_model, top_sen_dict) :
    model = joblib.load(fn_model)
    y_prob = model.predict_proba(x_train)[0]

    """ Customed prediction rules (to be finished) """

    """ Label processing """
    label_limit = top_sen_dict["size"]
    y_pred = prediction_decoding(y_prob, label_limit)
    #print (y_pred, y_prob)

    return y_pred


def produce_feedback_data (table, schedule, top_sen_size, comple=False) :
    """
    Atfer receiving one feedback, we create one data and save it.
    :param table: [np.array, shape=(size_col, size_row)] the original table
    :param schedule: [dict]
    :param top_sen_size: [int]
    :param comple: [bool] whether this is a negative feedback
    :return: [list] the format is in pairs 
                    [(model_index, data_vector), ..., ......,label_index]
    """
    data_new = []
    date = schedule["date"]
    label_index = schedule["label_index"]
    time_start, time_end = schedule["time_start"], schedule["time_end"]
    block_start, block_end = time_start, time_end - 1
    table_new = np.copy(table)
    power = 0

    while 2 ** power < top_sen_size :
        power += 1

    label = label_index if not comple else label_index + 2 ** (power + 1)
    for i in range(block_start, block_end + 1) :
        sample_new = dc.convert_to_train_sample(table_new, i, label)
        data_new.append((i,sample_new))

    return data_new


def save_feedback (path, data) :
    for sample_pair in data :
        model_index, sample = sample_pair[0], sample_pair[1]
        fp = path + "/data/data_%s.npy" % model_index
        if os.path.exists(fp) :
            data = np.load(fp)
            data = np.vstack((data, sample))
        else :
            data = np.array([sample])
        np.save(fp, data)



if __name__ == "__main__" :
    size_col, size_row = 24, 7
    path = "/media/jack/Data/Ubuntu/PycharmProjects/MachineLearning/py_ML_8"
    table = [[-1 for i in range(size_col)] for j in range(size_row)]

    # Load information with json
    fn_sen_list = path + '/top_info/sen.json'
    sen_dict = load_json(fn_sen_list)
    fn_top_sen_list = path + '/top_info/top_sen.json'
    top_sen_dict = load_json(fn_top_sen_list)
    top_sen_size = top_sen_dict["size"]

    while True :
        """ Input """
        date = input("What is the date? (1~7) ")
        time_start = input("What is the start time? (0~23) ")
        time_end = input("What is the end time? (0~23) ")
        sen = input("What is the schedule? ")

        date = int(date)
        time_start = int(time_start)
        time_end = int(time_end)

        """ Convert input into table (array) """
        table_new = produce_new_table(table, date, time_start, time_end, sen, sen_dict)
        table_new = np.array(table_new)

        """ Predctions on each block (model) """
        predictions = []
        for model_index in range (24) : #(temp)
            # whether the model exisisting
            fn_model = path + "/models_nn/model_nn_%s.pkl" % model_index
            if os.path.exists(fn_model) :
                sample = convert_to_predicted_sample(table_new, model_index)
                x_test = np.array([sample])
                #print ("model_index__", model_index)
                y_pred = model_prediction(x_test, fn_model, top_sen_dict)
                predictions.append(y_pred)
            else :
                predictions.append(0)
        #print (predictions)

        """ Command the schedules """
        schedules_command = predictions_to_schedules(predictions, size_col, size_row, top_sen_dict)
        #print (schedules_command)
        for sch in schedules_command :
            print ("Is this you want?")
            print ("{0:<40} : from {1:<3} to {2:<3}".format(sch["schedule_content"], sch["time_start"], sch["time_end"]))
            res = input("(y/n) : ")
            # Use feedback to create new data and save it

            if res == "y" :
                data_new = produce_feedback_data(table_new, sch, top_sen_size, comple=False)
            else :
                data_new = produce_feedback_data(table_new, sch, top_sen_size, comple=True)
            save_feedback(path, data_new)

        """ Control of terminating """
        terminate = input("Do you want to terminate it ? (y/n) ")
        if terminate == "y" :
            break
        else :
            table = table_new
