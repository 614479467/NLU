


eval_prompt = """
[Question]
{question}

[Assistant 1]
{answer1}

[End of Assistant 1]

[Assistant 2]
{answer2}

[End of Assistant 2]

[System]
{protocal_prompt}
""".strip()





protocal_prompt = """
We would like to request your feedback on the two AI assistants in response to the user question displayed above.
Requirements: {aspects}
{method}
""".strip()
# protocal_prompt = """
# We would like to request your feedback on the two AI assistants in response to the user question displayed above.
# Please evaluate the helpfulness, relevance, accuracy, level of details of their responses. You should tell me whether Assistant 1 is `better than`, `worse than`, or `equal to` Assistant 2.
# Please first compare their responses and analyze which one is more in line with the given requirements.
# In the last line, please output a single line containing only a single label selecting from `Assistant 1 is better than Assistant 2`, `Assistant 1 is worse than Assistant 2`, and `Assistant 1 is equal to Assistant 2`, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
# """.strip()


scoring_non_prompt = f"""
Please rate the performance of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. You should consider which response is more in line with the given requirements.
In the subsequent line, please provide a comprehensive explanation of your evaluation.
""".strip()


scoring_cot_prompt = f"""
Please rate the performance of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better performance.
Please first compare their responses and analyze which one is more in line with the given requirements.
In the last line, please output a single line scoring Assistant 1 and 2 in the form of 'Assistant 1: s1; Assistant 2: s2'.
""".strip()


classification_non_prompt = f"""
Please compare the performance of their responses. You should tell me whether Assistant 1 is `better than`, `worse than`, or `equal to` Assistant 2.
Please first output a single line containing only a single label selecting from `Assistant 1 is better than Assistant 2`, `Assistant 1 is worse than Assistant 2`, and `Assistant 1 is equal to Assistant 2`. You should consider which response is more in line with the given requirements.
In the subsequent line, please provide a comprehensive explanation of your evaluation.
""".strip()


classification_cot_prompt = f"""
Please compare the performance of their responses. You should tell me whether Assistant 1 is `better than`, `worse than`, or `equal to` Assistant 2.
Please first compare their responses and analyze which one is more in line with the given requirements.
In the last line, please output a single line containing only a single label selecting from `Assistant 1 is better than Assistant 2`, `Assistant 1 is worse than Assistant 2`, and `Assistant 1 is equal to Assistant 2`.
""".strip()


