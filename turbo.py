import multiprocessing
import os.path
import random

import requests
from retrying import retry
from tqdm import tqdm
from gpt import GPT
import json
import ssl
import re
from collections import defaultdict
from typing import List, Dict, Union, Any, Tuple
import numpy as np
import glob
from utils import *





if __name__ == '__main__':
    model_a_name = "new_huatuo_230K"
    model_b_name = "DoctorGLM"
    # model_b_name = "gpt4"

    turbo_referee = GPTReferee_NonChat(model_a_name, model_b_name, 
                                       eval_set='KUAKE-QIC100', 
                                       aspects='doc', 
                                       evaluation_method='classification_cot',
                                       referee='gpt-3.5-turbo',
                                       normalize=False,
                                       n_repeat=3,
                                       n_processes=50)
    
    # turbo_referee = GPTReferee_Chat(model_a_name, model_b_name, 
    #                                 eval_set='HaoDaiFu100', 
    #                                 aspects='doc_chat', 
    #                                 evaluation_method='classification_cot',
    #                                 referee='gpt-3.5-turbo',
    #                                 normalize=False,
    #                                 n_repeat=3,
    #                                 n_processes=50)
    turbo_referee.review()


