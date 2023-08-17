


chat_eval_prompt = """
[Assistant 1]
{conversation1}

[End of Assistant 1]

[Assistant 2]
{conversation2}

[End of Assistant 2]

[System]
{protocal_prompt}
""".strip()






chat_protocal_prompt = """
We would like to request your feedback on two multi-turn conversations between the AI assistant and the user displayed above.
Requirements: {aspects}
{method}
""".strip()


chat_scoring_non_prompt = f"""
Please rate the performance of the AI assistant in each conversation. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. You should consider whose behavior is more in line with the given requirements.
In the subsequent line, please provide a comprehensive explanation of your evaluation.
""".strip()


chat_scoring_cot_prompt = f"""
Please rate the performance of the AI assistant in each conversation. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better performance.
Please first compare their behavior and analyze which one is more in line with the given requirements.
In the last line, please output a single line scoring Assistant 1 and 2 in the form of 'Assistant 1: s1; Assistant 2: s2'.
""".strip()


chat_classification_non_prompt = f"""
Please compare the performance of the AI assistant in each conversation. You should tell me whether Assistant 1 is `better than`, `worse than`, or `equal to` Assistant 2.
Please first output a single line containing only a single label selecting from `Assistant 1 is better than Assistant 2`, `Assistant 1 is worse than Assistant 2`, and `Assistant 1 is equal to Assistant 2`. You should consider which response is more in line with the given requirements.
In the subsequent line, please provide a comprehensive explanation of your evaluation.
""".strip()


chat_classification_cot_prompt = f"""
Please compare the performance of the AI assistant in each conversation. You should tell me whether Assistant 1 is `better than`, `worse than`, or `equal to` Assistant 2.
Please first compare their responses and analyze which one is more in line with the given requirements.
In the last line, please output a single line containing only a single label selecting from `Assistant 1 is better than Assistant 2`, `Assistant 1 is worse than Assistant 2`, and `Assistant 1 is equal to Assistant 2`.
""".strip()




