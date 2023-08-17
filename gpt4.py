import multiprocessing
import os.path
import random

import requests
from retrying import retry
from tqdm import tqdm
# from gpt import GPT
import gpt
import json
import ssl
import re
from collections import defaultdict
from typing import List, Dict, Union, Any, Tuple
import numpy as np
import glob
from utils import *





if __name__ == '__main__':
    model_a_name = "chatgpt"
    model_b_name = 'llamaace-v2'
    # model_b_name = "gpt4"

    gpt4_referee = GPTReferee_NonChat(model_a_name, model_b_name, 
                                      eval_set='vicuna', 
                                      aspects='vicuna', 
                                      evaluation_method='scoring_cot',
                                      referee='gpt-4-api-chatanywhere',
                                      api_key='sk-PnkXqX5jMGgrSWRGnaIVmkW6Z7MZ47YgN0QpUNiPThOPhWOQ',
                                      normalize=False,
                                      n_repeat=1,
                                      n_processes=3)

    # gpt4_referee = GPTReferee_Chat(model_a_name, model_b_name, 
    #                                eval_set='HaoDaiFu100', 
    #                                aspects='doc_chat', 
    #                                evaluation_method='classification_cot',
    #                                referee='gpt-4-api',
    #                                api_key='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',
    #                                normalize=False,
    #                                n_repeat=1,
    #                                n_processes=10)
    gpt4_referee.review()



