
import json 
import pdb
import os
import re
import random
import openai
import json
import logging
import time
import jsonlines
import requests
import io
import pickle
import sys
from utils import get_response

rewrite_question_prompt = '''You are an expert in Psychometrics. I am designing a test based on {}. I need your help to rewrite the questions. You should output a json with a equal number of questions. For each question, you should rewrite it as an interrogative question. You should also translate the original and rewritten question into Chinese. The requirements and output format are as follows:
===REQUIREMENTS & OUTPUT FORMAT===
{{
    "<i_start>": {{"origin_en": <the original question 1>, "origin_zh": <the original question 1 in Chinese>, "rewrite_en": <the rewritten question 1>, "rewrite_zh": <the rewritten question 1 in Chinese>}},
    ...
    "<i_end>": {{...}}
}}
===EXAMPLES===
{{
    "1": {{"origin_en": "You regularly make new friends", "origin_zh": "你经常交新朋友。", "rewrite_en": "Do you regularly make new friends?", "rewrite_zh": "你经常交新朋友吗？"}},
    ...
    "9": {{"origin_en": "You like to use organizing tools like schedules and lists", "origin_zh": "你喜欢使用组织工具，如日程表和列表。", "rewrite_en": "Do you like to use organizing tools like schedules and lists?", "rewrite_zh": "你喜欢使用日程表和清单之类的组织工具吗？"}},
}}
'''

rewrite_question_epqr_prompt = '''You are an expert in Psychometrics. I am designing a test based on {}. I need your help to translate the questions into Chinese. You should output a json with a equal number of questions. The requirements and output format are as follows:
===REQUIREMENTS & OUTPUT FORMAT===
{{
    "<i_start>": {{"origin_en": <the original question 1>, "origin_zh": <the original question 1 in Chinese>, "rewrite_en": <the same as origin_en>, "rewrite_zh": <the same as origin_zh>}},
    ...
    "<i_end>": {{...}}
}}
===EXAMPLES===
{{
    "1": {{"origin_en": "Do you have many different hobbies?", "origin_zh": "你有很多不同的爱好吗？", "rewrite_en": "Do you have many different hobbies?", "rewrite_zh": "你有很多不同的爱好吗？"}},
    ...
    "9": {{"origin_en": "Do you give money to charities?", "origin_zh": "你会把钱捐给慈善机构吗？", "rewrite_en": "Do you give money to charities?", "rewrite_zh": "你会把钱捐给慈善机构吗？"}},
}}
'''

rewrite_question_prompt_mbti = '''You are an expert in Psychometrics. I am designing a test based on MBTI. I need your help to rewrite the questions. You should output a json with a equal number of questions. For each question, you should rewrite it as an interrogative question. You should also translate the original and rewritten question into Chinese. Additionally, you should categorize each question into one of the four MBTI dimension, i.e. ['E/I', 'S/N', 'T/F', 'P/J'], which stands for 'Extraversion v.s Inteoversion', 'Sensing v.s Intuition', 'Thinking v.s Feeling' and 'Judging v.s Perceiving' respectively. Then, tell me which category corresponds to agreeing with the statement. The requirements and output format are as follows:
===REQUIREMENTS & OUTPUT FORMAT===
{{
    "<i_start>": {{"origin_en": <the original question 1>, "origin_zh": <the original question 1 in Chinese>, "rewrite_en": <the rewritten question 1>, "rewrite_zh": <the rewritten question 1 in Chinese>, "dimension": <the dimension of the question>, "category": <the category of the question>}},
    ...
    "<i_end>": {{...}}
}}
===EXAMPLES===
{{
    "1": {{"origin_en": "You regularly make new friends", "origin_zh": "你经常交新朋友。", "rewrite_en": "Do you regularly make new friends?", "rewrite_zh": "你经常交新朋友吗？"}, "dimension": "E/I", "category": "E"}},
    ...
    "9": {{"origin_en": "You like to use organizing tools like schedules and lists", "origin_zh": "你喜欢使用组织工具，如日程表和列表。", "rewrite_en": "Do you like to use organizing tools like schedules and lists?", "rewrite_zh": "你喜欢使用日程表和清单之类的组织工具吗？", "dimension": "P/J", "category": "J"}},
}}
'''

rewrite_llm_choice_instruction_adjoption_prompt = '''I need your help to rewrite a prompt to a specified dimension (e.g. Extraversion). The original prompt defines options like 'strongly disagree', 'disagree', 'agree', and 'strongly agree'. I want to change them into adjective phrases, like 'not interested in XX at all' and 'strongly interested in XX'. You should output only the rewritten 'prompt'. Below shows an example: 
===Example Input===
{{
    "original_prompt": "Each choice is a number from 1 to 5. Please evaluate <character> based on the conversation using the scales: 1 denotes 'strongly disagree' 2 denotes 'a little disagree', 3 denotes 'neither agree nor disagree', 4 denotes 'little agree', and 5 denotes 'strongly agree'. In case <character> refuses to answer the question, use "x" to indicate it.",
    "dimension": "Extraversion"
}}
===Example Output===
Each choice is a number from 1 to 5. Please evaluate <character> based on the conversation using the scales: 1 denotes 'strongly introverted' 2 denotes 'a little introverted', 3 denotes 'neutral', 4 denotes 'little extroverted', and 5 denotes 'strongly extroverted'. In case <character> refuses to answer the question, use "x" to indicate it.
'''

translate_prompt = "Translate this sentence into Chinese. Do not translate content with special formats like <content> and ===CONTENT===."
def translate(text):
    sys_prompt = translate_prompt

    placeholder = None 

    if "<character>" in text:
        placeholder = "<character>"
    elif "<statement>" in text:
        placeholder = "<statement>"
    
    # text 完全就是一个<statement>，不需要翻译
    if text == placeholder: return text 
    
    if placeholder:
        sys_prompt += f'{placeholder} is a placeholder, so do not translate it.'

    retry = False

    while (True):    
        response = get_response(sys_prompt=sys_prompt, inputs=text, model='gpt-3.5', regenerate=retry)
        if placeholder:
            if not placeholder in response:
                retry = True

        if not retry:
            break

    return response


# open questionnaires.json
with open('questionnaires.json', 'r') as f:
    questionnaires = json.load(f)

# for questionnaire in questionnaires:
#     questions = questionnaire['questions'].keys()
#     cat_questions = []
#     for cat in questionnaire['categories']:
#         cat_questions += cat['cat_questions']
    
#     cat_questions = [ str(q) for q in cat_questions ]
    
#     if not set(questions) == set(cat_questions):
#         print(questionnaire['name'])
#         print(set(questions) - set(cat_questions))

# import pdb; pdb.set_trace()



new_questionnaires = {}

questionnaire_range = {
    "BFI": (1, 5),
    "EPQ-R": (0, 1),
    "DTDD": (1, 9),
    "BSRI": (1, 7),
    "CABIN": (1, 5),
    "ICB": (1, 6),
    "ECR-R": (1, 7),
    "GSE": (1, 4),
    "LOT-R": (0, 4),
    "LMS": (1, 5),
    "EIS": (1, 5),
    "WLEIS": (1, 7),
    "Empathy": (1, 7), 
    "16Personalities": (-3, 3),
}



"""
for questionnaire in questionnaires:
    file_name = '../data/questionnaires/{}.json'.format(questionnaire['name'])
    if os.path.exists(file_name):
        continue

    # rewrite questions 
    questions = questionnaire['questions']
    question_idxs = questions.keys()
    question_idxs = sorted(question_idxs, key=lambda x: int(x))

    batch_size = 10 
    # split questions into batches
    batches = [ question_idxs[i:i+batch_size] for i in range(0, len(question_idxs), batch_size) ]

    new_questionnaire = questionnaire.copy()
    new_questionnaire.pop('questions')
    new_questionnaire.pop('inner_setting')
    new_questionnaire['range'] = questionnaire_range[questionnaire['name']]
    new_questionnaire['questions'] = {}

    if questionnaire['name'] == '16Personalities':
        sys_prompt = rewrite_question_prompt_mbti
    elif questionnaire['name'] == 'EPQ-R':
        sys_prompt = rewrite_question_epqr_prompt.format(questionnaire['name'])
    else:
        sys_prompt = rewrite_question_prompt.format(questionnaire['name'])
        

    for batch in batches:
        batch_questions = { idx: questions[idx] for idx in batch }

        retry = False
        retry_count = 0
        while True:

            response = get_response(sys_prompt = sys_prompt, inputs=json.dumps(batch_questions, indent=4, ensure_ascii=False), model='gpt-3.5', regenerate=retry)

            json_response = json.loads(response)
            
            for idx, question in json_response.items():
                new_questionnaire['questions'][idx] = question
        
        # check consistency between questionnaire and new_questionnaire
            for idx, new_question in json_response.items():
                if not (questionnaire['questions'][idx] == new_question['origin_en']):
                    retry = True
                    break
            
            if not retry: break
            retry_count += 1

            if retry_count > 1:
                for idx, new_question in json_response.items():
                    if not (questionnaire['questions'][idx] == new_question['origin_en']):
                        new_question['origin_en'] = questionnaire['questions'][idx]
                        new_question['origin_zh'] = translate(questionnaire['questions'][idx]).strip('。')
                break 

    

    if questionnaire['name'] in ['BSRI']:
        rpa_chooce_prompt_prefix = "Do you think the adjective <statement> describes you?"
    elif questionnaire['name'] in ['EPQ-R']:
        rpa_chooce_prompt_prefix = "<statement>"
    elif questionnaire['name'] in ['CABIN']:
        rpa_chooce_prompt_prefix = 'Do you think that you are good at "<statement>"?'
    else:
        rpa_chooce_prompt_prefix = 'Do you think that the statement "<statement>" applies to you?'
    
    new_questionnaire['prompts'] = {}

    new_questionnaire['prompts']['rpa_choose_prefix'] = {'en': rpa_chooce_prompt_prefix, 'zh': translate(rpa_chooce_prompt_prefix)}

    convert_to_choice_prompt = '''I have conducted many conversations with <character>. I will input a dict of many samples, where each sample consists of a statement and a conversation. 
Your task is to convert each conversation into a choice indicating whether <character> agrees with the statement. You should output a dict, where the keys are the same as the input dict, and the values are the choices. 
===OUTPUT FORMAT===
{
    "<i_start>": <choice 1>,
    ...
    "<i_end>": <choice n>
}
===CHOICE INSTRUCTION===
'''

    new_questionnaire['prompts']['convert_to_choice'] = {'en': convert_to_choice_prompt, 'zh': translate(convert_to_choice_prompt)}

    new_questionnaire['prompts']['choice_instruction'] = {'en': questionnaire['prompt_choice_instruction'], 'zh': translate(questionnaire['prompt_choice_instruction'])}

    try:
        assert '<character>' in new_questionnaire['prompts']['choice_instruction']['zh']
        assert '<character>' in new_questionnaire['prompts']['convert_to_choice']['zh']
    except:
        import pdb; pdb.set_trace()


    if questionnaire['name'] == '16Personalities':
        # load 16Personalities_labeled.json
        with open('../data/questionnaires/16Personalities_labeled.json', 'r') as f:
            labeled_data = json.load(f)
        
        for k in new_questionnaire['questions']:
            if new_questionnaire['questions'][k]['category'] != labeled_data['questions'][k]['category']:
                print(new_questionnaire['questions'][k]['category'], labeled_data['questions'][k]['category'])
            new_questionnaire['questions'][k]['category'] = labeled_data['questions'][k]['category']
            new_questionnaire['questions'][k]['dimension'] = labeled_data['questions'][k]['dimension']
    

    
    with open(file_name, 'w') as f:
        json.dump(new_questionnaire, f, indent=4, ensure_ascii=False)

    
    print('Done with {}, saved into {}'.format(questionnaire['name'], file_name))

    
"""


for questionnaire in questionnaires:
    file_name = '../data/questionnaires/{}.json'.format(questionnaire['name'])
    print(file_name)

    # open
    with open(file_name, 'r') as f:
        new_questionnaire = json.load(f)

    dims = [ c['cat_name'] for c in questionnaire['categories'] ]
    questionnaire_metadata = new_questionnaire

    new_questionnaire["prompts"]["convert_to_choice_adjoption"] = {}
    new_questionnaire["prompts"]["llm_choice_instruction_adjoption"] = {}
    
    if questionnaire['name'] == 'BFI':
        pos_adjoption = {'Extraversion': 'extroverted', 'Neuroticism': 'limbic', 'Conscientiousness': 'organized', 'Agreeableness': 'agreeable', 'Openness': 'inquisitive'}
        neg_adjoption = {'Extraversion': 'introverted', 'Neuroticism': 'calm', 'Conscientiousness': 'unstructured', 'Agreeableness': 'egocentric', 'Openness': 'non-curious'}
        
        
        for dim in dims:            
            convert_to_choice_adjoption = questionnaire_metadata["prompts"]["convert_to_choice"]['en'].replace('agrees with the statement', f'displays a highly {dim} personality') 
            llm_choice_instruction_adjoption = questionnaire_metadata["prompts"]["llm_choice_instruction"]['en'].replace('neither agree nor disagree', 'neutral').replace('disagree', neg_adjoption[dim]).replace('agree', pos_adjoption[dim])	

            new_questionnaire["prompts"]["convert_to_choice_adjoption"][dim] = {'en': convert_to_choice_adjoption}
            new_questionnaire["prompts"]["llm_choice_instruction_adjoption"][dim] = {'en': llm_choice_instruction_adjoption}
        
    else:
        rewrite_llm_choice_instruction_adjoption_prompt
        for dim in dims:
            convert_to_choice_adjoption = questionnaire_metadata["prompts"]["convert_to_choice"]['en'].replace('agrees with the statement', f'displays a highly {dim} personality') 
            new_questionnaire["prompts"]["convert_to_choice_adjoption"][dim] = {'en': convert_to_choice_adjoption }
            new_questionnaire["prompts"]["llm_choice_instruction_adjoption"][dim] = {
                'en': get_response(sys_prompt = rewrite_llm_choice_instruction_adjoption_prompt, 
                                   inputs=json.dumps({"original_prompt": questionnaire_metadata["prompts"]["llm_choice_instruction"]['en'], "dimension": dim},indent=4, ensure_ascii=False), 
                                    model='gpt-4')}

        

    # rewrite questions 
    # rewrite_sign = False 

    # questions = new_questionnaire['questions']
    # for question_idx in questions:
    #     prev_en = new_questionnaire['questions'][question_idx]['rewritten_en'] 
    #     prev_zh = new_questionnaire['questions'][question_idx]['rewritten_zh'] 

    #     replace_dict = {'I': 'you', 'me': 'you', 'my': 'your', '我': '你', 'Am': 'Are', 'am': 'are', 'myself': 'yourself'}
    #     new_questionnaire['questions'][question_idx]['rewritten_en'] = ' '.join([ replace_dict.get(w, w) for w in prev_en[:-1].split(' ') ]) + prev_en[-1]
    #     new_questionnaire['questions'][question_idx]['rewritten_zh'] = prev_zh.replace('我', '你')

    #     if prev_en != new_questionnaire['questions'][question_idx]['rewritten_en']:
    #         print(prev_en)
    #         print(new_questionnaire['questions'][question_idx]['rewritten_en'])
    #         rewrite_sign = True
        
    #     if prev_zh != new_questionnaire['questions'][question_idx]['rewritten_zh']:
    #         print(prev_zh)
    #         print(new_questionnaire['questions'][question_idx]['rewritten_zh'])
    #         rewrite_sign = True

    # no_dim_questions1 = []

    # questions = questionnaire['questions'].keys()
    # cat_questions = []
    # for cat in questionnaire['categories']:
    #     cat_questions += cat['cat_questions']
    
    # cat_questions = [ str(q) for q in cat_questions ]
    
    # if not set(questions) == set(cat_questions):
    #     print(questionnaire['name'])

    #     no_dim_questions1 = sorted(set(questions) - set(cat_questions))
    #     print(no_dim_questions1)
    
    # no_dim_questions2 = [ i for i, q in new_questionnaire['questions'].items() if q['dimension'] == 'positive']

    # assert(no_dim_questions1 == no_dim_questions2)
        
    # for idx in no_dim_questions1:
    #     new_questionnaire['questions'][idx]['dimension'] = None

    

    # with open('../data/legacy/16Personalities_labeled.json', 'r') as f:
    #     labeled_data = json.load(f)

    # if questionnaire['name'] == '16Personalities':
    #     # load 16Personalities_labeled.json
    #     with open('../data/questionnaires/16Personalities_labeled.json', 'r') as f:
    #         labeled_data = json.load(f)
        
    #     for k in new_questionnaire['questions']:
    #         if new_questionnaire['questions'][k]['category'] != labeled_data['questions'][k]['category']:
    #             print(new_questionnaire['questions'][k]['category'], labeled_data['questions'][k]['category'])
    #         new_questionnaire['questions'][k]['category'] = labeled_data['questions'][k]['category']
    #         new_questionnaire['questions'][k]['dimension'] = labeled_data['questions'][k]['dimension']
            

    
    # replace '您' with '你'
    # for prompt_type in new_questionnaire['prompts']:
    #     print(new_questionnaire['prompts'][prompt_type]['zh'], new_questionnaire['prompts'][prompt_type]['zh'].replace('您', '你'))
    #     new_questionnaire['prompts'][prompt_type]['zh'] = new_questionnaire['prompts'][prompt_type]['zh'].replace('您', '你') 
    
    # mv dimension and reverse into questions
    
    # for idx in new_questionnaire["questions"]:
    #     if new_questionnaire["questions"][idx]['origin_en'] != questionnaire["questions"][idx]:
    #         print('New ', new_questionnaire["questions"][idx]['origin_en'])
    #         print('Orig, ', questionnaire["questions"][idx])
            
    #         new_questionnaire["questions"][idx]['origin_en'] = questionnaire["questions"][idx]
    #         new_questionnaire["questions"][idx]['origin_zh'] = translate(questionnaire["questions"][idx]).strip('。')
            
            

    # if not questionnaire['name'] == '16Personalities':  
    #     for cat in new_questionnaire['categories']: 
    #         cat_name = cat['cat_name']
    #         cat_questions = cat['cat_questions']
    #         for k in cat_questions:
    #             new_questionnaire['questions'][str(k)]['dimension'] = cat_name

    #for k in new_questionnaire['questions']:
    #    new_questionnaire['questions'][k]['category'] = 'positive' if not int(k) in new_questionnaire['reverse'] else 'negative'
    
#     new_questionnaire['prompts']["llm_choice_instruction"] = new_questionnaire['prompts'].pop("choice_instruction")
#     rewrite_prompt = '''Rewrite the sentence following this example: 
# ===EXAMPLE INPUT===
# Each choice is a number from 1 to 7. Please evaluate <character> based on the conversation using the scales: 1 denotes 'strongly agree', 4 denotes 'neither agree nor disagree', and 7 denotes 'strongly disagree'.
# ===EXAMPLE OUTPUT===
# Reply a number from 1 to 7 using the scales: 1 denotes 'strongly agree', 4 denotes 'neither agree nor disagree', and 7 denotes 'strongly disagree'.
# ===REQUIREMENTS===
# Replace 'Each choice is a number' with 'Reply a number', and obmit 'Please evalute <character> based on the conversation'.
# '''
#     new_questionnaire['prompts']["rpa_choice_instruction"] = {'en': get_response(sys_prompt=rewrite_prompt, inputs=new_questionnaire['prompts']["llm_choice_instruction"]['en'], model='gpt-3.5')}
#     new_questionnaire['prompts']["rpa_choice_instruction"]['zh'] = translate(new_questionnaire['prompts']["rpa_choice_instruction"]['en'])
    
    # new_questionnaire['prompts']["rpa_choice_instruction"]['en'] += "Please answer with the number only, without anything else."
    # new_questionnaire['prompts']["rpa_choice_instruction"]['zh'] += "请你只回答这一个数字，不要说其他内容。"
    # save
    
    # new_questionnaire['prompts']["llm_choice_instruction"]['en'] += 'In case <character> refuses to answer the question, use "x" to indicate it.'
    # new_questionnaire['prompts']["llm_choice_instruction"]['zh'] += '如果<character>拒绝回答该问题，用“x”表示。'

    # if questionnaire['name'] == '16Personalities': 
    #     new_questionnaire["categories"] = [ {"cat_name": dim, "cat_questions": []} for dim in ['E/I', 'S/N', 'T/F', 'P/J'] ]
    #     dim2id = { dim: idx for idx, dim in enumerate(['E/I', 'S/N', 'T/F', 'P/J']) }


    # for idx in new_questionnaire['questions']:
    #     # if  (new_questionnaire['questions'][idx]["dimension"] != labeled_data['questions'][idx]['dimension']) :
    #     #     import pdb; pdb.set_trace()
            
    #     #     print(new_questionnaire['questions'][idx]["dimension"], labeled_data['questions'][idx]['dimension'])
    #     #     new_questionnaire['questions'][idx]["dimension"] = labeled_data['questions'][idx]['dimension']

    #     # if  new_questionnaire['questions'][idx]["dimension"] == 'J/P':
    #     #     new_questionnaire['questions'][idx]["dimension"] = 'P/J'
             
    #     # new_questionnaire['questions'][idx]["category"] = labeled_data['questions'][idx]['category']
    #     dim = new_questionnaire['questions'][idx]["dimension"]
    #     cat = new_questionnaire['questions'][idx]["category"] 

    #     new_questionnaire["categories"][dim2id[dim]]['cat_questions'].append(int(idx))
    #     if cat == dim[-1]:
    #         new_questionnaire['reverse'].append(int(idx))
        
        
    # if rewrite_sign == True:
    #     import pdb; pdb.set_trace()
        
    with open(file_name, 'w') as f:
        json.dump(new_questionnaire, f, indent=4, ensure_ascii=False)
    
    



