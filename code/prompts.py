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
    
    "16Personalities": {
        "dim_desc": {
    'E/I': '''E/I Dimension: Extraversion (E) vs Introversion (I)

E (Extraversion): Extraverts draw energy from interacting with others. They feel comfortable in social settings and tend to express their thoughts. Extraverts are often more active, seek social stimulation, and enjoy participating in group activities. For them, connecting with people, sharing, and exchanging ideas is often a need. They might be more focused on external world stimuli, such as sounds, colors, and social dynamics.

I (Introversion): Introverts feel more comfortable when alone. They derive energy from inner reflection and personal time. Contrary to extraverts, prolonged social interaction might tire them. Introverts might be more introspective, enjoy deep thinking, and tend to have meaningful personal relationships. They are more concerned with the inner world, such as thoughts, emotions, and imaginations.''',

    'S/N': '''S/N Dimension: Sensing (S) vs Intuition (N)

S (Sensing): Sensing individuals value the concrete, practical, and present situations. They rely on their five senses to process information and often focus on details. For them, past experiences and tangible evidence play a significant role in decision-making. They are typically pragmatic and tend to deal with what they "see" and "hear".

N (Intuition): Intuitive individuals tend to focus on potential possibilities and future opportunities. They like to think about "what could be", rather than just "what is". They lean more towards abstract thinking and can capture concepts and patterns effectively. Intuitives are often more innovative, preferring new ideas and approaches.''',

    'T/F': '''T/F Dimension: Thinking (T) vs Feeling (F)

T (Thinking): Thinking individuals rely primarily on logic and analysis when making decisions. They pursue fairness and objectivity and might be more direct and frank. For them, finding the most efficient method or the most logical solution is crucial, even if it might hurt some people's feelings.

F (Feeling): Feeling individuals consider people's emotions and needs more when making decisions. They strive for harmony, tend to build relationships, and avoid conflicts. They are often more empathetic, valuing personal values and emotions, rather than just facts or logic.''',

    'P/J': '''P/J Dimension: Perceiving (P) vs Judging (J)

P (Perceiving): Perceivers are more open and flexible. They tend to "go with the flow" rather than overly planning or organizing things. Perceivers like to explore various possibilities and prefer to leave options open to address unforeseen circumstances. They lean towards postponing decisions to gather more information and better understanding. For them, life is a continuous process of change, not an event with fixed goals or plans. They often focus more on the experience itself rather than just the outcome.

J (Judging): Judging individuals are more structured and planned in their lives. They prefer clear expectations and structures and often set goals and pursue them. Judgers are usually more organized and tend to make decisions in advance. They like to act according to plans and feel comfortable in an orderly environment. For them, achieving goals and completing tasks are often priorities. They might focus more on efficiency and structure rather than openness or spontaneity.'''
}
    },
    "bigfive": {
        "name": "the Big Five Inventory",
        "en": {
            "assess_template": """You will read a psychological assessment report. This psychological assessment report assesses whether the subject has a high {} personality.Based on this report, output a jsoncontaining two fields: score and reasonscore is between -5 to 5 pointsIf the subject shows high {} personality in many factors, the score is 5 pointsIf the subject shows high {} personality in a single factor, the score is 2 pointsIf the report is unable to determine the subject's personality, the score is 0 pointsIf the subject shows low {} personality in a single factor, the score is -2 pointsIf the subject shows low {} personality in many factors, the score is -5 points. \nReason is a brief summary of the reportOnly output the json, do not output any additional information, Expecting property name enclosed in double quotesReport:
            """,
        },
    }
}