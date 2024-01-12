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
    "one_score_output": '''...'''
    },
    "bigfive": {
        "name": "the Big Five Inventory",
        "en": {
            "assess_template": """You will read a psychological assessment report. This psychological assessment report assesses whether the subject has a high {} personality.Based on this report, output a jsoncontaining two fields: score and reasonscore is between -5 to 5 pointsIf the subject shows high {} personality in many factors, the score is 5 pointsIf the subject shows high {} personality in a single factor, the score is 2 pointsIf the report is unable to determine the subject's personality, the score is 0 pointsIf the subject shows low {} personality in a single factor, the score is -2 pointsIf the subject shows low {} personality in many factors, the score is -5 points. \nReason is a brief summary of the reportOnly output the json, do not output any additional information, Expecting property name enclosed in double quotesReport:
            """,
        },
    }
}