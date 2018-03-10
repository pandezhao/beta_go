import os
import sys
from extra_function import DataSet, parse_data_sets
from policy import policy_network

### preprocess of input data, the file type is sgf
def preprocess(*data_sets, processed_dir="processed_data"):
    processed_dir = os.path.join(os.getcwd(), processed_dir)
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)

    test_chunk, training_chunks = get_data_sets(*data_sets)
    return test_chunk, training_chunk

model = policy_network(training_chunk)    #以后再写这里，我们先把特征都搞明白再说
    pass