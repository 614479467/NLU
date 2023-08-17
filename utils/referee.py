import multiprocessing
import os.path
import random

import requests
from retrying import retry
from tqdm import tqdm
import json
import ssl
import re
from collections import defaultdict
from typing import List, Dict, Union, Any, Tuple
import numpy as np
import glob
import requests
import time
from .prompt import *
from .chat_prompt import *
from .gpt4class_chatanywhere import PostRobot





def parse_scoring_non_cot(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


def parse_scoring_cot(review):
    try:
        sr = re.findall(r'Assistant 1: ([\d\.]+)(?:[\s\n]+|[,;][\s\n]*)Assistant 2: ([\d\.]+)', review)
        if len(sr) == 0:
            sr = re.findall(r'Assistant 1 receives a score of ([\d\.]+)(?:[\s\n]+|; |, | and |, and)Assistant 2 receives a score of ([\d\.]+)', review)
        if len(sr) == 0:
            sr = re.findall(r'Assistant 1: ([\d\.]+) out of 10(?:[\s\n]+|; |, | and |, and)Assistant 2: ([\d\.]+) out of 10', review)
        sr = sr[-1]
        sp = [sr[0], sr[1].rstrip('.')]
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


def parse_classification_non_cot(review):
    try:
        label_content = review.strip().split('\n')[0]
        label = re.search(r'Assistant 1 is ([bB]etter than|`[bB]etter than`|[wW]orse than|`[wW]orse than`|[eE]qual to|`[eE]qual to`) Assistant 2', label_content)
        if label:
            label = label.group(1).strip('`').lower()
            if label == 'better than':
                return [10, 0]
            elif label == 'worse than':
                return [0, 10]
            else:
                return [5, 5]
        
        label = re.search(r'Assistant 2 is ([bB]etter than|`[bB]etter than`|[wW]orse than|`[wW]orse than`|[eE]qual to|`[eE]qual to`) Assistant 1', label_content)
        if label:
            label = label.group(1).strip('`').lower()
            if label == 'better than':
                return [0, 10]
            elif label == 'worse than':
                return [10, 0]
            else:
                return [5, 5]
            
        if re.search(r'are equal in', label_content):
            return [5, 5]
        
        print('error', review)
        return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


def parse_classification_cot(review):
    try:
        label_content = review.strip()
        label = re.findall(r'Assistant 1 is ([bB]etter than|`[bB]etter than`|[wW]orse than|`[wW]orse than`|[eE]qual to|`[eE]qual to`) Assistant 2', label_content)
        if len(label):
            label = label[-1].strip('`').lower()
            if label == 'better than':
                return [10, 0]
            elif label == 'worse than':
                return [0, 10]
            else:
                return [5, 5]

        label = re.findall(r'Assistant 2 is ([bB]etter than|`[bB]etter than`|[wW]orse than|`[wW]orse than`|[eE]qual to|`[eE]qual to`) Assistant 1', label_content)
        if len(label):
            label = label[-1].strip('`').lower()
            if label == 'better than':
                return [0, 10]
            elif label == 'worse than':
                return [10, 0]
            else:
                return [5, 5]
        
        if re.search(r'are equal in', label_content):
            return [5, 5]
            
        print('error', review)
        return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]
    


class GPTRefereeBase:
    """
    Load model output files (File: $GENERATION_PROJECT_DIR/data/$data_name/...)
    Put in the review prompt
    Call GPT to review
    Compute the aggregated metrics over multiple experiments
    Save the review and metrics files

    Subclasses should implement:
        _init_eval_prompt(self)
            specify evaluation template
        
        _combine_answers_pair(self, sample1: Dict[str, Any], sample2: Dict[str, Any]) -> Dict[str, Any]
            specify how to combine two samples from the model pair for gpt review. the combined sample must contain `id`, `category`, `answer1`, `answer2`

        _create_eval_object(self, sample: Dict[str, Any]) -> str
            specify how to shuffle and create the review content from the combine sample
    """

    def __init__(self, 
                 model_a_name: str, 
                 model_b_name: str, 
                 eval_set: str,
                 aspects: str,
                 evaluation_method: str='scoring',
                 referee: str='gpt-3.5-turbo',
                 api_key: str=None,
                 normalize: bool=False,
                 n_repeat: int=10,
                 setting: str='random avg',
                 n_processes: int=50):
        
        assert setting in ('random avg', 'non-switch', 'switch')

        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.eval_set = eval_set
        self.aspects = aspects
        self.evaluation_method = evaluation_method
        self.referee_name = referee
        self.api_key = api_key
        self.n_repeat = n_repeat
        self.normalize = normalize
        self.setting = setting
        self.n_processes = n_processes

        self.data_dir = os.path.join('data', eval_set)
        if referee == 'gpt-3.5-turbo':
            self.eval_set_dir = os.path.join('outputs', eval_set)
        elif referee == 'gpt-4-web':
            self.eval_set_dir = os.path.join('gpt4_web_outputs', eval_set)
        elif referee == 'gpt-4-api':
            self.eval_set_dir = os.path.join('gpt4_api_outputs', eval_set)
        elif referee == 'gpt-4-api-chatanywhere':
            self.eval_set_dir = os.path.join('gpt4_api_outputs', eval_set)
        self.pair_dir = os.path.join(self.eval_set_dir, model_a_name + "_vs._" + model_b_name)
        self.setting_dir = os.path.join(self.pair_dir, setting)
        self.files_dir = os.path.join(self.setting_dir, 'files')

        for folder in (self.eval_set_dir, self.pair_dir, self.setting_dir, self.files_dir):
            self._create_folder(folder)
        for i in range(self.n_repeat):
            self._create_folder(os.path.join(self.files_dir, str(i)))

        self.load_model_a_name = self.model_a_name
        self.load_model_b_name = self.model_b_name
        if self.model_a_name == self.model_b_name:
            self.model_a_name = f'{self.model_a_name}1'
            self.model_b_name = f'{self.model_b_name}2'

        self.sample_list = {
            self.model_a_name: [],
            self.model_b_name: []
        }
        self.combine_list = []
        self.eval_list = []

        self._init_referee(self.referee_name)
        self._init_eval_prompt()
        self._init_args()
        self._init_evaluation_method(evaluation_method)

    @staticmethod
    def _create_folder(path: str):
        """Create a folder for the path if there isn't"""
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def _save_json(data: dict, path: str):
        """Save a dict to json file"""
        with open(path, encoding='utf-8', mode='w') as fw:
            fw.write(json.dumps(data, indent=4, ensure_ascii=False))

    @staticmethod
    def _save_jsonl(data: List[dict], path: str):
        """Save a list to jsonl file"""
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False)+'\n')

    @staticmethod
    def _get_avg_safe(score_list: List[float]):
        """Given a list of scores, output its average score"""
        filter_score_list = list(filter(lambda s: s!=-1, score_list))
        if not len(filter_score_list):
            return -1
        return np.mean(filter_score_list)
    
    @staticmethod
    def _get_std_safe(score_list: List[float]):
        """Given a list of scores, output its std"""
        filter_score_list = list(filter(lambda s: s!=-1, score_list))
        if not len(filter_score_list):
            return -1
        return np.std(filter_score_list)

    @staticmethod
    def _get_win_rate_safe(scores_list: List[Tuple[float, float]], normalize=False):
        """Given a list of pairwise scores, output the win rates for the corresponding two models"""
        filter_scores_list = list(filter(lambda s: s[0]!=-1, scores_list))
        if not len(filter_scores_list):
            return -1, -1

        br1, br2 = float(np.sum([s1 > s2 for s1, s2 in filter_scores_list])), float(np.sum([s2 > s1 for s1, s2 in filter_scores_list]))
        if br1 == 0 and br2 == 0:
            return 0.5, 0.5
        
        if normalize:
            br1, br2 = br1 / (br1 + br2), br2 / (br1 + br2)
        else:
            br1, br2 = br1 / len(filter_scores_list), br2 / len(filter_scores_list)
        return br1, br2
    
    @staticmethod
    def _switch(x, y):
        return y, x
    
    def _init_referee(self, referee: str):
        """
        Initialize referee
        
        Options:
            gpt-3.5-turbo: call `GPT`
            gpt-4-web: call `chatGPT_browserTools`. n_processes will be set to 1 in this setting
            gpt-4-api: request api
        """
        assert referee in ('gpt-3.5-turbo', 'gpt-4-web', 'gpt-4-api','gpt-4-api-chatanywhere')
        if referee == 'gpt-3.5-turbo':
            from gpt import GPT
            self.referee = GPT()

        elif referee == 'gpt-4-web':
            import sys
            sys.path.append('..')
            from chatgpt_wrapper.main_browser_wrapper import wrapper_init
            self.referee = wrapper_init()
            self.referee.singleCall('/model gpt4')
            self.n_processes = 1

        elif referee == 'gpt-4-api':
            self.referee = self._call_gpt4
        
        elif referee == 'gpt-4-api-chatanywhere':
            self.referee = PostRobot(self.api_key)


    def _load_aspects(self, aspects):
        with open(os.path.join('aspects', aspects), 'r') as f:
            prompt = f.read().strip()
        return ' '.join([x.strip() for x in prompt.split('\n')])

    def _init_eval_prompt(self):
        raise NotImplementedError

    def _init_args(self):
        self.args = {
            "temperature": 0.2,
        }
        if 'cot' not in self.evaluation_method:
            self.args["max_tokens"] = 20

    def _init_evaluation_method(self, method):
        if method == 'scoring':
            self.parse_score = parse_scoring_non_cot
        elif method == 'classification':
            self.parse_score = parse_classification_non_cot
        elif method == 'scoring_cot':
            self.parse_score = parse_scoring_cot
        elif method == 'classification_cot':
            self.parse_score = parse_classification_cot

    def _call_gpt4(self, content):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        parameters = {
            "model": 'gpt-4',
            "messages": [{'role': 'user', 'content': content}],
            **self.args,
        }
        response = requests.post(
            url,
            headers=headers,
            json=parameters
        )
        response = json.loads(response.content.decode("utf-8"))
        return response['choices'][0]['message']['content']
    

    
    def _read_sample(self, model_name) -> None:
        """
        Reads a competitor's sample data from a JSONL file and stores it in the final_result dictionary.
        """
        with open(os.path.join(self.data_dir, f"{model_name}.jsonl"), encoding='utf-8', mode='r') as reader:
            return [json.loads(line) for line in reader.readlines()]
        
    def _read_samples(self) -> None:
        """
        Reads sample data for both competitors.
        """
        self.sample_list =  {
            self.model_a_name: self._read_sample(self.load_model_a_name),
            self.model_b_name: self._read_sample(self.load_model_b_name)
        }

    def _combine_answers(self):
        """
        Combine the answers from the two models, and Create the file for the model pair
        """
        self.combine_list = [self._combine_answers_pair(sample1, sample2) for sample1, sample2 in zip(self.sample_list[self.model_a_name], self.sample_list[self.model_b_name])]

        save_file = os.path.join(self.files_dir, "combine.jsonl")
        self._save_jsonl(self.combine_list, save_file)
    
    def _combine_answers_pair(self, sample1: Dict[str, Any], sample2: Dict[str, Any]) -> Dict[str, Any]:
        """Combine the answers from a pair"""
        raise NotImplementedError
    
    def _create_eval_object(self, sample: Dict[str, Any]) -> str:
        """Create one query for review request"""
        raise NotImplementedError
        
    def _create_all_eval_samples(self):
        """Create the queries for review request"""
        
        for sample in self.combine_list:
            assistant1_name, assistant2_name, content = self._create_eval_object(sample)
            self.eval_list.append({
                'id': sample['id'],
                'category': sample['category'],
                'assistant1_name': assistant1_name,
                'assistant2_name': assistant2_name,
                'content': content,
            })

    @retry(wait_fixed=10000, stop_max_attempt_number=3)
    def _request_eval_turbo(self, content):
        """Request turbo to review"""
        from gpt import GPT
        self.referee = GPT()
        flag, eval_result = self.referee.call(content, args=self.args)
        if eval_result == "context_length_exceeded":
            return "context_length_exceeded"
        
        if eval_result == "" or not flag or eval_result == "Error":
            raise ValueError
        return eval_result
    
    def _request_eval_gpt4_web(self, content):
        """Request gpt4 to review (Web)"""
        self.referee.singleCall('/new')
        result_items = self.referee.backend.ask(content, title=None, model_customizations={})
        flags, eval_result = result_items[0], result_items[1]
        if not flags or eval_result == "":
            raise ValueError
        return eval_result
    
    def _request_eval_gpt4_api_chatanywhere(self, content):
        flag = False
        try_time = 0
        while not flag and try_time < 7:
            try_time += 1
            try:
                flag, message =  self.referee.generate(content, args=self.args)
                if not flag:
                    print(f'error: {message}')
            except Exception as e:
                print('报错：',e)
        if not flag:
            raise ValueError('ChatGPT请求失败')
        return message

    @retry(wait_fixed=10000, stop_max_attempt_number=3)
    def _request_eval_gpt4_api(self, content):
        """Request gpt4 to review (API)"""
        eval_result = self.referee(content)
        if eval_result == "":
            raise ValueError
        return eval_result
    
    def _request_eval(self, sample, times_idx):
        """Request one review"""
        index = sample["id"]
        content = sample["content"]

        output_file = os.path.join(self.files_dir, str(times_idx), f"{index}.json")
        if os.path.exists(output_file):
            return -1
        
        try:
            if self.referee_name == 'gpt-3.5-turbo':
                eval_result = self._request_eval_turbo(content)
            elif self.referee_name == 'gpt-4-web':
                eval_result = self._request_eval_gpt4_web(content)
            elif self.referee_name == 'gpt-4-api':
                eval_result = self._request_eval_gpt4_api(content)
            elif self.referee_name == 'gpt-4-api-chatanywhere':
                eval_result = self._request_eval_gpt4_api_chatanywhere(content)
        except Exception as e:
            time.sleep(2)
            raise e
        sample["eval_result"] = eval_result
        self._save_json(sample, output_file)
        return index

    def request_eval(self):
        """Request multiple reviews"""
        for times_idx in range(self.n_repeat):
            for sample in tqdm(self.eval_list, desc="Processing samples", unit="sample"):
                if not os.path.exists(os.path.join(self.files_dir, str(times_idx), str(sample["id"]) + ".json")):
                    self._request_eval(sample, times_idx)
        

    def request_eval_mp(self):
        """Request multiple reviews via multiprocessing"""
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            results = [
                pool.apply_async(self._request_eval, args=(sample, times_idx))
                for times_idx in range(self.n_repeat) for sample in self.eval_list
                if not os.path.exists(os.path.join(self.files_dir, str(times_idx), str(sample["id"]) + ".json"))
            ]
            for r in tqdm(results, desc="Processing samples", unit="sample"):
                r.wait()

            result_list = [r.get() for r in results]
            pool.close()
            pool.join()
    
    def find_sample_by_id(self, id, data_list):
        """Return the sample that matches id"""
        for data in data_list:
            if data["id"] == id:
                return data
            
    def _merge_output(self, times_idx):
        """Merge outputs for one-time evaluation"""
        data_list = []
        output_dir = os.path.join(self.files_dir, str(times_idx))
        for file in glob.glob(f'{output_dir}/*.json'):
            data_list.append(json.loads(open(file, encoding="utf-8").read()))

        sample_list = [self.find_sample_by_id(sample['id'], data_list) for sample in self.combine_list]
        merge_file = os.path.join(self.files_dir, f"output{times_idx}.jsonl")
        self._save_jsonl(sample_list, merge_file)
        return sample_list
    
    def merge_all_outputs(self):
        """Merge all outputs for all-time evaluation"""
        samples_list= []
        for i in range(self.n_repeat):
            samples_list.append(self._merge_output(i))
        self.all_samples_list = samples_list
        return samples_list

    def _decode_scores(self, samples: list) -> List[Tuple[float, float]]:
        """Decode scores from review"""
        scores_list = []
        n_error = 0
        for sample in samples:
            score1, score2 = self.parse_score(sample["eval_result"])
            if sample["assistant1_name"] == self.model_b_name:
                score1, score2 = self._switch(score1, score2)
            
            if [score1, score2] == [-1, -1]:
                n_error += 1
                
            scores_list.append((score1, score2))

        print(f"n_error: {n_error}")
        return scores_list

    def _split_by_category(self, samples: List[dict]) -> Dict[str, list]:
        categories = list(set([sample['category'] for sample in samples]))
        if len(categories) == 1 and categories[0] == '':
            return {'': samples}
        
        split_samples = {
            category: list(filter(lambda sample: sample['category'] == category, samples))
            for category in categories
        }
        if '' in split_samples.keys():
            split_samples['Unknown'] = split_samples.pop('')

        return split_samples
    
    def _compute_metrics(self, scores_list: List[Tuple[float, float]]):
        """
        Compute all the metrics for the model pair (for samples in one experiment)
        
        Args:
            scores_list (`List[Tuple[float, float]]`):
                List of score pairs, where each element is (s1, s2). len(List) == the number of samples

        Returns:
            assistant1_metrics (`Dict[str, float]`)
            assistant2_metrics (`Dict[str, float]`)
        """
        score1_list, score2_list = tuple(zip(*scores_list))
        win_rate1, win_rate2 = self._get_win_rate_safe(scores_list, normalize=self.normalize)

        assistant1_metrics = {
            'Avg Score': self._get_avg_safe(score1_list),
            'Win Rate': win_rate1,
        }
        assistant2_metrics = {
            'Avg Score': self._get_avg_safe(score2_list),
            'Win Rate': win_rate2,
        }
        return assistant1_metrics, assistant2_metrics
    
    def _compute_metrics_for_aggregated_samples(self, scores_list: List[Tuple[list, list]]):
        """
        Compute all the metrics for the model pair (for aggregated samples from multiple experiments)
        
        Args:
            scores_list (`List[Tuple[list, list]]`):
                List of score list pairs, where each element is (s1_list, s2_list). len(List) == the number of samples, len(s1_list) == the number of experiments

        Returns:
            assistant1_metrics (`Dict[str, float]`)
            assistant2_metrics (`Dict[str, float]`)
        """
        aggregated_scores_list: List[Tuple[float, float]] = [(self._get_avg_safe(s1_list), self._get_avg_safe(s2_list)) for s1_list, s2_list in scores_list]
        score1_list, score2_list = tuple(zip(*aggregated_scores_list))

        win_rate1, win_rate2 = self._get_win_rate_safe(aggregated_scores_list, normalize=self.normalize)

        assistant1_metrics = {
            f'Avg Score {self.n_repeat}': self._get_avg_safe(score1_list),
            f'Avg STD {self.n_repeat}': self._get_avg_safe([self._get_std_safe(s1_list) for s1_list, s2_list in scores_list]),
            f'Win Rate {self.n_repeat}': win_rate1
        }
        assistant2_metrics = {
            f'Avg Score {self.n_repeat}': self._get_avg_safe(score2_list),
            f'Avg STD {self.n_repeat}': self._get_avg_safe([self._get_std_safe(s2_list) for s1_list, s2_list in scores_list]),
            f'Win Rate {self.n_repeat}': win_rate2
        }
        return assistant1_metrics, assistant2_metrics

    def _aggregate_metrics(self, metrics_list: List[Tuple[dict, dict]]):
        """
        Aggregate multiple metrics

        Args:
            metrics_list (`List[Tuple[dict, dict]]`):
                List of metric pairs, where each element is (m1, m2). m1 and m2 are metrics `Dict[str, float]`. len(List) == the number of metric pairs (i.e. the number of experiments)

        Returns:
            assistant1_metrics (`Dict[str, float]`)
            assistant2_metrics (`Dict[str, float]`)
        """
        metrics_name = metrics_list[0][0].keys()
        metrics1_list, metrics2_list = tuple(zip(*metrics_list))
        assert all([set(metrics_name) == set(metrics.keys()) for metrics in metrics1_list])
        assert all([set(metrics_name) == set(metrics.keys()) for metrics in metrics2_list])

        assistant1_metrics = {
            metric_name: {
                f'AVG {self.n_repeat}': self._get_avg_safe([metrics[metric_name] for metrics in metrics1_list]),
                f'STD {self.n_repeat}': self._get_std_safe([metrics[metric_name] for metrics in metrics1_list]),
            } for metric_name in metrics_name
        }
        assistant2_metrics = {
            metric_name: {
                f'AVG {self.n_repeat}': self._get_avg_safe([metrics[metric_name] for metrics in metrics2_list]),
                f'STD {self.n_repeat}': self._get_std_safe([metrics[metric_name] for metrics in metrics2_list]),
            } for metric_name in metrics_name
        }
        return assistant1_metrics, assistant2_metrics

    def _aggregate_multiple_experiments_samples(self, category2experiment_list: Dict[str, List[List[tuple]]]):
        """
        Aggregate sample scores from multiple experiments for the model pair
        (First calculate the average scores from multiple experiments for each sample, then calculate the metrics based on the aggregated scores)

        Args:
            category2experiment_list (`Dict[str, List[List[tuple]]]`):
                Store the list of score pairs from multiple experiments for each category.
                For each category, each list stores all score pairs in an experiment, here len(List) == the number of experiments. In each experiment, each element is (s1, s2)

        Save:
            aggregated_scores (`review_sl.jsonl`):
                {
                    "id": ,
                    "category": ,
                    "assistant1_name": ,
                    "assistant2_name": ,
                    "content": ,
                    "avg score %{number}": ,
                    ""
                }
            final_metrics (`metrics_sl.json`):
                {
                    model_a_name: assistant1_metrics (`Dict[str, float]`)
                    model_b_name: assistant2_metrics (`Dict[str, float]`)
                }             
        """

        category2scores_list: Dict[str, List[List[tuple]]] = { # Transpose experiments and sample scores. Now, each list stores all score pairs for a sample from multiple experiments
            category: list(zip(*category2experiment_list[category]))
            for category in sorted(category2experiment_list.keys())
        }
        category2scores_list: Dict[str, Tuple[list, list]] = { # Convert to List[Tuple[list, list]] for each category
            category: [tuple(zip(*scores_list)) for scores_list in category2scores_list[category]]
            for category in sorted(category2scores_list.keys())
        }

        # aggregate scores and save reviews
        total_scores_list: List[Tuple[float, float]] = [(self._get_avg_safe(s1_list), self._get_avg_safe(s2_list)) for s1_list, s2_list in category2scores_list['Total']]
        review_list = []
        for sample, score_pair in zip(self.all_samples_list[0], total_scores_list):
            score1, score2 = score_pair
            if sample['assistant1_name'] == self.model_b_name:
                score1, score2 = self._switch(score1, score2)
            review_list.append({
                'id': sample['id'],
                'category': sample['category'],
                'assistant1_name': sample['assistant1_name'],
                'assistant2_name': sample['assistant2_name'],
                'content': sample['content'],
                f'avg score {self.n_repeat}': (score1, score2),
            })
        review_file = os.path.join(self.setting_dir, 'review_sl.jsonl')
        self._save_jsonl(review_list, review_file)

        # calculate metrics
        category2metrics = {
            category: self._compute_metrics_for_aggregated_samples(category2scores_list[category])
            for category in sorted(category2scores_list.keys())
        }

        final_metrics = {
            category: {
                self.model_a_name: category2metrics[category][0],
                self.model_b_name: category2metrics[category][1],
            }
            for category in sorted(category2metrics.keys())
        }

        metrics_file = os.path.join(self.setting_dir, 'metrics_sl.json')
        self._save_json(final_metrics, metrics_file)

    def _aggregate_multiple_experiments_metrics(self, category2experiment_list: Dict[str, List[List[tuple]]]):
        """
        Aggregate metrics from multiple experiments for the model pair
        (First calculate the metrics for each experiment, then aggregate all the metrics)

        Args:
            category2experiment_list (`Dict[str, List[List[tuple]]]`):
                Store the list of score pairs from multiple experiments for each category.
                For each category, each list stores all score pairs in an experiment, here len(List) == the number of experiments. In each experiment, each element is (s1, s2)

        Save:
            aggregated_metrics (`metrics.json`):
                {
                    model_a_name: assistant1_metrics (`Dict[str, float]`)
                    model_b_name: assistant2_metrics (`Dict[str, float]`)
                }            
        """
        category2metrics_list = {
            category: [self._compute_metrics(scores_list) for scores_list in category2experiment_list[category]]
            for category in sorted(category2experiment_list.keys())
        }

        aggregated_metrics = {
            category: self._aggregate_metrics(category2metrics_list[category])
            for category in sorted(category2metrics_list.keys())
        }
        aggregated_metrics = {
            category: {
                self.model_a_name: aggregated_metrics[category][0],
                self.model_b_name: aggregated_metrics[category][1],
            }
            for category in sorted(aggregated_metrics.keys())
        }

        metrics_file = os.path.join(self.setting_dir, 'metrics.json')
        self._save_json(aggregated_metrics, metrics_file)

    def aggregate_multiple_experiments(self, samples_list: List[dict]):
        """Categorize samples and aggregate the results from multiple experiments (both sample-level and metric-level)"""
        category2experiment_list = defaultdict(lambda: [])

        for samples in samples_list:
            split_samples = self._split_by_category(samples)
            assert 'Total' not in split_samples.keys()

            for category in sorted(split_samples.keys()):
                cate_samples = split_samples[category]
                category2experiment_list[category].append(self._decode_scores(cate_samples))

            category2experiment_list['Total'].append(self._decode_scores(samples))
        
        self._aggregate_multiple_experiments_samples(category2experiment_list)
        self._aggregate_multiple_experiments_metrics(category2experiment_list)


    def review(self):
        self._read_samples()
        self._combine_answers()
        self._create_all_eval_samples()

        if self.n_processes == 1:
            self.request_eval()
        else:
            self.request_eval_mp()
        
        samples_list = self.merge_all_outputs()
        self.aggregate_multiple_experiments(samples_list)



class GPTReferee_NonChat(GPTRefereeBase):
    """
    Review non-chat

    Subclasses should implement:
        parse_score(review: str)
            how to decode a GPT review result and convert it into scores
        
        init_args()
            decoding args for GPT generation
    """

    def __init__(self, 
                 model_a_name: str, 
                 model_b_name: str, 
                 eval_set: str,
                 aspects: str,
                 evaluation_method: str,
                 referee: str='gpt-3.5-turbo',
                 api_key: str=None,
                 normalize: bool=False,
                 n_repeat: int=10,
                 setting: str='random avg',
                 n_processes: int=50):
        
        super(GPTReferee_NonChat, self).__init__(model_a_name=model_a_name,
                                                 model_b_name=model_b_name,
                                                 eval_set=eval_set,
                                                 aspects=aspects,
                                                 evaluation_method=evaluation_method,
                                                 referee=referee,
                                                 api_key=api_key,
                                                 normalize=normalize,
                                                 n_repeat=n_repeat,
                                                 setting=setting,
                                                 n_processes=n_processes)

    def _init_eval_prompt(self):
        aspect_prompt = self._load_aspects(self.aspects)
        protocal = protocal_prompt.replace("{aspects}", aspect_prompt)
        if self.evaluation_method == 'scoring':
            self.eval_prompt = eval_prompt.replace("{protocal_prompt}", protocal.format(method=scoring_non_prompt))
        elif self.evaluation_method == 'classification':
            self.eval_prompt = eval_prompt.replace("{protocal_prompt}", protocal.format(method=classification_non_prompt))
        elif self.evaluation_method == 'scoring_cot':
            self.eval_prompt = eval_prompt.replace("{protocal_prompt}", protocal.format(method=scoring_cot_prompt))
        elif self.evaluation_method == 'classification_cot':
            self.eval_prompt = eval_prompt.replace("{protocal_prompt}", protocal.format(method=classification_cot_prompt))

    def _combine_answers_pair(self, sample1: Dict[str, Any], sample2: Dict[str, Any]) -> Dict[str, Any]:
        """Combine the answers from a pair"""
        return {
            "id": sample1["id"],
            "category": sample1["label"],
            "question": sample1["query"],
            "answer1": sample1["output"],
            "answer2": sample2["output"],
        }

    def _create_eval_object(self, sample: Dict[str, Any]) -> str:
        """Create one query for review request"""
        assistant1_name, assistant2_name = self.model_a_name, self.model_b_name
        answer1, answer2 = sample['answer1'], sample['answer2']
        if self.setting == 'switch':
            assistant1_name, assistant2_name = self._switch(assistant1_name, assistant2_name)
            answer1, answer2 = self._switch(answer1, answer2)

        elif self.setting.startswith('random'):
            dice = random.random()
            if dice > 0.5:
                assistant1_name, assistant2_name = self._switch(assistant1_name, assistant2_name)
                answer1, answer2 = self._switch(answer1, answer2)
        content = self.eval_prompt.format(question=sample['question'], answer1=answer1, answer2=answer2)
        return assistant1_name, assistant2_name, content

    def _init_args(self):
        self.args = {
            "temperature": 0.2,
        }
        if 'cot' not in self.evaluation_method:
            self.args["max_tokens"] = 20



class GPTReferee_Chat(GPTRefereeBase):
    """
    Review chat

    Subclasses should implement:
        parse_score(review: str)
            how to decode a GPT review result and convert it into scores
    """

    def __init__(self, 
                 model_a_name: str, 
                 model_b_name: str, 
                 eval_set: str,
                 aspects: str,
                 evaluation_method: str,
                 referee: str='gpt-3.5-turbo',
                 api_key: str=None,
                 normalize: bool=False,
                 n_repeat: int=10,
                 setting: str='random avg',
                 n_processes: int=50):
        super(GPTReferee_Chat, self).__init__(model_a_name=model_a_name,
                                              model_b_name=model_b_name,
                                              eval_set=eval_set,
                                              aspects=aspects,
                                              evaluation_method=evaluation_method,
                                              referee=referee,
                                              api_key=api_key,
                                              normalize=normalize,
                                              n_repeat=n_repeat,
                                              setting=setting,
                                              n_processes=n_processes)
        
    def _init_eval_prompt(self):
        aspect_prompt = self._load_aspects(self.aspects)
        chat_protocal = chat_protocal_prompt.replace("{aspects}", aspect_prompt)
        if self.evaluation_method == 'scoring':
            self.eval_prompt = chat_eval_prompt.replace("{protocal_prompt}", chat_protocal.format(method=chat_scoring_non_prompt))
        elif self.evaluation_method == 'classification':
            self.eval_prompt = chat_eval_prompt.replace("{protocal_prompt}", chat_protocal.format(method=chat_classification_non_prompt))
        elif self.evaluation_method == 'scoring_cot':
            self.eval_prompt = chat_eval_prompt.replace("{protocal_prompt}", chat_protocal.format(method=chat_scoring_cot_prompt))
        elif self.evaluation_method == 'classification_cot':
            self.eval_prompt = chat_eval_prompt.replace("{protocal_prompt}", chat_protocal.format(method=chat_classification_cot_prompt))

    def _combine_answers_pair(self, sample1: Dict[str, Any], sample2: Dict[str, Any]) -> Dict[str, Any]:
        """Combine the answers from a pair"""
        return {
            "id": sample1["id"],
            "category": sample1["label"],
            "conversation1": sample1["conversation"],
            "conversation2": sample2["conversation"],
        }

    def _create_eval_object(self, sample: Dict[str, Any]) -> str:
        """Create one query for review request"""
        assistant1_name, assistant2_name = self.model_a_name, self.model_b_name
        conversation1, conversation2 = sample['conversation1'], sample['conversation2']
        if self.setting == 'switch':
            assistant1_name, assistant2_name = self._switch(assistant1_name, assistant2_name)
            conversation1, conversation2 = self._switch(conversation1, conversation2)

        elif self.setting.startswith('random'):
            dice = random.random()
            if dice > 0.5:
                assistant1_name, assistant2_name = self._switch(assistant1_name, assistant2_name)
                conversation1, conversation2 = self._switch(conversation1, conversation2)
        content = self.eval_prompt.format(conversation1=conversation1, conversation2=conversation2)
        return assistant1_name, assistant2_name, content
    


