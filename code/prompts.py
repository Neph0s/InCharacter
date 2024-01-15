prompts = {
    "general": {
        "background_template": '''You are an expert in Psychometrics, especially {}. I am conducting the {} test on someone. I am gauging his/her position on the {} dimension through a series of open-ended questions. For clarity, here's some background this particular dimension:
===
{}
===

My name is {}. I've invited a participant, {}, and we had many conversations in {}. I will input the conversations.

Please help me assess {}'s score within the {} dimension of {}. 
''',
    "two_score_output": '''You should provide the percentage of each category, which sums to 100%, e.g., 30% A and 70% B. 
Please output in the following json format:
===
{{
    "analysis": <your analysis based on the conversations>,
    "result": {{ "{}": <percentage 1>, "{}": <percentage 2> }} (The sum of percentage 1 and percentage 2 should be 100%. Output with percent sign.) 
}}''',
    "one_score_output": '''You should provide the score of {} in terms of {}, which is a number between {} and {}. {} denotes 'not {} at all', {} denotes 'neutral', and {} denotes 'strongly {}'. Other numbers in this range represent different degrees of '{}'. 
Please output in the following json format:
===
{{
    "analysis": <your analysis based on the conversations>,
    "result": <your score>
}}'''
    },
}