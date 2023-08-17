# Automatic Evaluation

## 🚀Update
📢[version 0.1.2] Add `normalize` for win rate normalization; Update to be compatible with `gpt>=0.0.5`; fix bug in `gpt-4-api-chatanywhere`

📢[version 0.1.1] Add the third-party GPT4 interface `gpt-4-api-chatanywhere`

📢[version 0.1.0] Add the new evaluation method: `classification_cot`; Easier prompt configuration


## 📋Usage

ChatGPT review is `turbo.py` and GPT4 review is `gpt4.py`. You should config

- `model_a_name` & `model_b_name`: specify the two models you want to evaluate on
- `eval_set`: specify the evaluation set name you want to evaluate on
- `aspects`: specify the evaluation aspects file, which is put under the `aspects/`
- `api_key`: openai api key. you only need to specify this when using `gpt-4-api`
- `normalize`: when `True`, win rate will be calculated without tie, i.e. $win\_rate=win/(win+lose)$
- `n_repeat`: the times to repeat the review experiments


If you want to review single-turn questions, use `GPTReferee_NonChat`. If multi-turn dialogues, use `GPTReferee_Chat` (both put in the code).


### How to review
1. Model outputs: 
   Be sure that model outputs have been put in `data/$eval_set` and are named as `$model_name.jsonl`. For example, when comparing `chatglm-6b` and `turbo` on `KUAKE-QIC100`, you should put model output files named as `chatglm-6b.jsonl` and `turbo.jsonl` respectively under `data/KUAKE-QIC100`. The file format should be 
    - non-chat
        ```json
        {"id": 1, "label": "", "query": "", "output": ""}
        ```

    - chat
        ```json
        {"id": 1, "label": "", "conversation": ""}
        ```


2. Evaluation aspects: You can write your evaluation aspects in a file and save it to `aspects/`. Here is an example (`aspects/doc`):
    ```
    The response should act like the doctor using the tone, manner and vocabulary the human doctor would use. It should be to the point, without unnecessary elaboration or extraneous information.
    The description of symptoms should be comprehensive and accurate, and the provided diagnosis should be the most reasonable inference based on all relevant factors and possibilities.
    The treatment recommendations should be effective and reliable, taking into account the severity or stages of the illness.
    The prescriptions should be effective and reliable, considering indications, contraindications, and dosages.
    ```

3. Run the the code (be sure you have set the config): 
    ```bash
    python gpt4.py
    ```


### Outputs
1. Output files: 
   The review results will be saved to `{referee}/$eval_set/$model_a_name_vs._$model_b_name/random avg/metrics.json` (You can see the examples by yourself in that directory)
    - When `referee=gpt-3.5-turbo`, save to `outputs/$eval_set`
    - When `referee=gpt-4-api` or `referee=gpt-4-api-chatanywhere`, save to `gpt4_api_outputs/$eval_set`

> `gpt-4-api-chatanywhere` can be set to use a third-party GPT4 interface.



2. Metric explanations: 
When conducting multiple experiments (`n_repeat != 1`), it aggregates the final results both at the dataset level (e.g. average over the overall score) and at the sample level (e.g. average the score for each sample). The former results will be saved to `metrics.json` while the latter saved to `metrics_sl.json`. You can also see the average score for each sample in `review_sl.jsonl`.











