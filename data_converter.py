import numpy as np
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords



### Download nltk source
"""
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
"""

### Functions

def convert_to_format_data (data, size_col=24, size_row=7) :
    data = np.array(data)
    output_tmp = []

    for i, row in enumerate(data) :
        if i % size_col == 0 :
            output_tmp.append([])
        output_tmp[-1].append(list(row[1:]))

    # Transpose
    output_tmp = np.array(output_tmp)
    output = []

    for i in range(len(output_tmp)) :
        output.append(list(output_tmp[i].T))
    output = np.array(output)

    return output


def nlp(data) :
    stemmer = WordNetLemmatizer()

    for week in data :
         for day in week :
             for i in range(len(day)) :
                sentence = day[i]
                if len(sentence) > 0 :
                    words = word_tokenize(str(sentence))
                    for j, word in enumerate(words) :
                        # Delete pure marks
                        if len(word) == 1 and word not in string.ascii_letters+string.digits :
                            words[j] = ""
                        else :
                            word_new = stemmer.lemmatize(word.lower())
                            if word_new != word : words[j] = word_new
                    sentence_new = " ".join(words)
                    if sentence_new != sentence : day[i] = sentence_new


def sen_split(sen) :
    words = []
    word = ""

    for elem in sen :
        if elem in string.digits + string.ascii_letters :
            word += elem
        elif elem not in string.digits + string.ascii_letters and len(word) > 0:
            words.append(word)
            word = ""

    # last word need to be added
    if len(word) > 0 :
        words.append(word)

    return words

def top_freq_words (data, num_top=10) :
    sw = stopwords.words("english")
    clean_words = []

    for week in data :
        for day in week :
            for sen in day :
                if len(sen) > 0 :
                    words = sen_split(sen)
                    for word in words :
                        if word not in sw and len(word) > 0 :
                            clean_words.append(word)

    freq_dist = nltk.FreqDist(clean_words)
    freq_list = sorted(list(freq_dist.items()), key=lambda x: x[1], reverse=True)
    top_freq = [freq_list[i][0] for i in range(num_top)]

    return top_freq


def has_top_words (s, top_words) :
    for word in s.split() :
        if word in top_words :
            return True
    return False


def is_rearrange_matching (s, s_list) :
    for i, s_compared in enumerate(s_list) :
        if len(s_compared) == len(s) :
            for w in s.split() :
                if w not in s_compared :
                    break
            else : return i
    return None


def is_children_matching (s, s_list) :
    banned_word = ["on", "with", "of"]  # for demo

    for i, s_compared in enumerate(s_list) :
        if len(s) < len(s_compared) : s1, s2 = s, s_compared
        else : s1, s2 = s_compared, s
        for word in s1.split() :
            if word not in s2 :
                break
        else :
            diff_list = list(set(s2) - set(s1))
            for word in diff_list :
                if word in banned_word :
                    break
            else :
                return i

    return None


def sen_encoding (sen, sen_list) :
    """ Encoding the sentence by checking whether
        it's in sen_list or not, if so the index
        is the encoding, otherwise return None"""
    index = None
    # exsisting / rearrangement matching
    if sen in sen_list :
        index = sen_list.index(sen)
    else :
        index = is_rearrange_matching(sen, sen_list)
        if index == None :
            index = is_children_matching(sen, sen_list)
    return index


def schedule_encoding(data, top_freq_words, label_limitation, size_col=24, size_row=7) :
    sen_list = [] # index is the label and element is the sentence
    count_list = []
    output_labeled_data = [ [ [ -1 for k in range(size_col) ] for j in range(size_row)] for i in range(data.shape[0])]

    for i, week in enumerate(data) :
        for j, day in enumerate(week) :
            for k, sen in enumerate(day) :
                if len(sen) > 0 and has_top_words(sen, top_freq_words) :
                    index = sen_encoding(sen, sen_list)
                    # Add new one into sen_list or it's already in sen_list
                    if index != None :
                        count_list[index] += 1
                    else :
                        sen_list.append(sen)
                        count_list.append(1)
                        index = len(sen_list) - 1
                    output_labeled_data[i][j][k] = index
    output_labeled_data = np.array(output_labeled_data)

    # Count the occurrence of index to get top_index_list
    top_list = sorted(list(zip(range(len(count_list)), sen_list, count_list)), key=lambda x: x[2], reverse=True)
    top_index_list = [top_list[i][0] for i in range(min(len(count_list), label_limitation))]
    top_sen_list = [top_list[i][1] for i in range(min(len(count_list), label_limitation))]

    return output_labeled_data, top_index_list, top_sen_list, sen_list


def convert_to_train_sample (sample, model_index, label_new) :
    """
    :param sample: [np.array, shape=(size_row, size_col)]
    :return: [np.array, shape=(1,None)] 
    """
    sample = list(sample.flatten())
    sample.pop(model_index)
    sample.append(label_new)
    sample = np.array(sample)
    return sample


def convert_to_train_data (data, model_index, top_index_list, size_row=7, size_col=24, complementary=False) :
    """
    :param data: [np.array]
    :param model_index: [int] this data is for which model (i.e. block), starting from 0
    :param label_limitation: [int] the limitation of num of labels
    :param complementary: [bool] is this for the complementary label
    :return: [np.array] format data prepared for training where
                        -1 -> None, and it does NOT contain complementary labels
    """

    data_new = []
    date, block = model_index // size_col, model_index % size_col

    for sample in data :
        if sample[date][block] in top_index_list:
            label_new = top_index_list.index(sample[date][block]) + 1
            sample = convert_to_train_sample(sample, model_index, label_new)
            data_new.append(sample)
        elif sample[date][block] == -1 :
            label_new = 0
            sample = convert_to_train_sample(sample, model_index, label_new)
            data_new.append(sample)

    return np.array(data_new)



if __name__ == "__main__" :
    """
    label_limitation = 10
    data = convert_to_format_data(data_pd)
    nlp(data)
    top_words = top_freq_words(data)

    data_encoded, top_index_list = schedule_encoding(data, top_words, label_limitation)
    print (top_index_list)

    model_index = 58
    data_train = convert_to_train_data(data_encoded, model_index, top_index_list)
    print (data_train[:5])
    """