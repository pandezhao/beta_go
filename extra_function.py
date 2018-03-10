import numpy as np
import os
import itertools
from get_game import replay_sgf

def load_sgf(*dataset_dirs): #
    for dataset_dir in dataset_dirs:
        full_dir = os.path.join(os.getcwd(), dataset_dir)
        dataset_files = [os.path.join(full_dir, name) for name in os.listdir(full_dir)]  #在这里返回full_dir之下所包含的所有文件和文件夹，并装进列表里
        for f in dataset_files:                                                          #判断语句，判断这个文件是不是以后缀名 .sgf结尾的文件，是的话就生成（yield）该文件
            if os.path.isfile(f) and f.endswith(".sgf"):
                yield f

def get_data_sets(*data_sets):
    sgf_files = list(load_sgf(*data_sets))                                               #load_sgf函数返回的是一个一个的f，这里用list把这些f装在成一个列表
    print("%s sgfs is found." % len(sgf_files))
    positions_w_context = itertools.chain(*map(extract_from_sgf, sgf_files))             #在这里建立files与函数提取文件的映射

    pass
    return test_chunk, training_chunks

def extract_from_sgf(file):
    with open(file,'r') as f:
         for position_w_context in replay_sgf(f.read()):
             yield position_w_context