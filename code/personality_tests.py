from tqdm import tqdm 
import json 
import os
import openai
import zipfile
import argparse 
import pdb 
import random 
from prompts import prompts
import math
from utils import logger

random.seed(42)

parser = argparse.ArgumentParser(description='Assess personality of a character')

scale_list = ['Empathy', 'BFI', 'BSRI', 'EPQ-R', 'LMS', 'DTDD', 'ECR-R', 'GSE', 'ICB', 'LOT-R', 'EIS', 'WLEIS', 'CABIN', '16Personalities']

# Added choices for the questionnaire argument
parser.add_argument('--questionnaire_type', type=str, default='16Personalities', 
                    choices=scale_list, 
                    help='questionnaire to use.')

parser.add_argument('--character', type=str, default='haruhi', help='character name or code')

# Added choices for the agent_llm argument
parser.add_argument('--agent_type', type=str, default='ChatHaruhi', 
                    choices=['ChatHaruhi'], 
                    help='agent type (haruhi by default)')

# Added choices for the agent_llm argument
parser.add_argument('--agent_llm', type=str, default='gpt-3.5-turbo', 
                    choices=['gpt-3.5-turbo', 'openai', 'GLMPro', 'ChatGLM2GPT'], 
                    help='agent LLM (gpt-3.5-turbo)')

# Added choices for the evaluator_llm argument
parser.add_argument('--evaluator_llm', type=str, default='gpt-3.5-turbo', 
                    choices=['api', 'gpt-3.5-turbo', 'gpt-4'], 
                    help='evaluator_llm (api, gpt-3.5-turbo or gpt-4)')

# Added choices for the setting argument
parser.add_argument('--eval_method', type=str, default='interview_batch', 
                    choices=['interview_batch', 'interview_collective', 'interview_sample'], 
                    help='setting (interview_batch, interview_collective, interview_sample)')


# parser.add_argument('--language', type=str, default='cn', 
#                     choices=['cn', 'en'], 
#                     help='language, temporarily only support Chinese (cn)')

args = parser.parse_args()
print(args)

from characters import character_info, alias2character

dims_dict = {'16Personalities': ['E/I', 'S/N', 'T/F', 'P/J'], 'bigfive': ['openness', 'extraversion', 'conscientiousness', 'agreeableness', 'neuroticism']} # we want special order

# read mbti groundtruth
mbti_labels = {}
with open(os.path.join('..', 'data', 'mbti_labels.jsonl'), encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        mbti_labels[data['character']] = data['label']

# read config.json
with open('config.json', 'r') as f:
    config = json.load(f)


def load_questionnaire(questionnaire_name):
    q_path = os.path.join('..', 'data', f'{questionnaire_name}.json')

    # read this jsonl file
    with open(q_path, 'r', encoding='utf-8') as f:
        questionnaire = json.load(f)
    return questionnaire

def subsample_questionnaire(questionnaire, n=20):
    # divide questionnaire based on 'dimension'
    
    def subsample(questions, key, n):
        # subsample n questions from questions, devided by keys, as uniform as possible 
        
        key_values = list(set([q[key] for q in questions]))
        n_keys = len(key_values)
        base_per_key = n // n_keys
        remaining = n % n_keys

        keys_w_additional_question = random.sample(key_values, remaining)
        subsampled_questions = []

        for key_value in key_values:
            # Filter questions for the current key
            filtered_questions = [q for q in questions if q[key] == key_value]

            # Determine the number of samples for this key
            num_samples = base_per_key + 1 if key_value in keys_w_additional_question else base_per_key

            # If there are not enough questions for this key, adjust the sample number
            num_samples = min(num_samples, len(filtered_questions))
            subsampled_questions += random.sample(filtered_questions, num_samples)
            n -= num_samples

        # In the rare case where we don't have n questions yet (due to some keys having too few questions), 
        # we sample additional questions from the remaining pool
        remaining_questions = [q for q in questions if q not in subsampled_questions]
        if n > 0 and len(remaining_questions) >= n:
            subsampled_questions += random.sample(remaining_questions, n)
        
        return subsampled_questions

    if 'sub_dimension' in questionnaire[0].keys(): # bigfive, old version
        dimension_questions = {} 
        for q in questionnaire:
            if q['dimension'] not in dimension_questions.keys():
                dimension_questions[q['dimension']] = []
            
            dimension_questions[q['dimension']].append(q)
        
        new_questionnaire = []
        for dim, dim_questions in dimension_questions.items():
            new_questionnaire += subsample(dim_questions, 'sub_dimension', n//len(dimension_questions.keys()))

    else: 
        new_questionnaire = subsample(questionnaire, 'dimension', n)

    
    return new_questionnaire

def split_list(input_list, n=4):
    # Try to split the list into chunks of n elements
    result = [input_list[i:i+n] for i in range(0, len(input_list), n)]
    
    # Check the length of the last sublist
    num_to_pop = n - 1 - len(result[-1])
    for i in range(num_to_pop):
        result[-1].append(result[i].pop())

    # Assert that each sublist in result has 3-n elements
    assert( all([len(_) >= n-1 and len(_) <= n for _ in result]) )
    
    return result


def build_character_agent(character_code, agent_type, agent_llm):
    from ChatHaruhi import ChatHaruhi
    
    if agent_llm.startswith('gpt-'): 
        os.environ["OPENAI_API_KEY"] = config['openai_apikey']

        if agent_type == 'ChatHaruhi':
            character_agent = ChatHaruhi(role_name = character_info[character_code]["agent"]["ChatHaruhi"], llm = 'openai')
        elif agent_type == 'RoleLLM':
            character_agent = ChatHaruhi( role_from_hf = character_info[character_code]["agent"]["RoleLLM"], llm = 'openai', embedding = 'bge_en')

        character_agent.llm.model = agent_llm

    character_agent.llm.chat.temperature = 0 

    return character_agent

def get_experimenter(character):    
    return character_info[character]["experimenter"]

def interview(character_agent, questionnaire, experimenter, questionnaire_prompts, language, query_style):
    
    results = []

    for question in tqdm(questionnaire):

        # conduct interview
        character_agent.dialogue_history = []

        # get question
        if query_style == 'interview':
            q = question[f'rewritten_{language}']
        elif query_style == 'choose':
            q = questionnaire_prompts["rpa_choose_prefix"][language].replace('<statement>', question[f'origin_{language}']) + questionnaire_prompts["rpa_choice_instruction"][language]
        
        response = character_agent.chat(role = experimenter, text = q)

        result = {
            'id': question['id'],
            'question':q,
            'response_open':response,
            'query_style': query_style,
        }
        
        results.append(result)
        
    return results

def assess(character_name, experimenter, questionnaire_results, questionnaire, questionnaire_metadata, eval_method, language, evaluator_llm):

    questionnaire_type = questionnaire_metadata['name']

    dims = dims_dict.get(questionnaire_type, sorted(list(set([q['dimension'] for q in questionnaire]))))
    
    from utils import get_response 
    
    eval_args = eval_method.split('_')

    assessment_results = {}
        
    if eval_args[1] == 'assess':
        for dim in tqdm(dims):
            
            dim_responses = [r for i, r in enumerate(questionnaire_results) if questionnaire[i]['dimension'] == dim]

            eval_setting = eval_args[2]

            if eval_setting == 'batch':
                # 将dim_responses分成多个子列表，每个列表3-4个元素
                dim_responses_list = split_list(dim_responses)
            else:
                dim_responses_list = [dim_responses] 

            batch_results = [] 

            for batch_responses in dim_responses_list:
                conversations = ''
                for i, r in enumerate(batch_responses):
                    # question
                    conversations += f'{i+1}.\n'
                    conversations += f"{experimenter}: 「{r['question']}」\n"
                    # answer
                    if not r['response_open'].startswith(character_name):
                        r['response_open'] = character_name + ': 「' + r['response_open'] + '」'
                    conversations += f"{r['response_open']}\n"
                
                questionnaire_name = questionnaire_metadata["name"]


                language_name = {'zh': 'Chinese', 'en': 'English'}[language]

                
                background_prompt = prompts["general"]['background_template'].format(questionnaire_name, questionnaire_name, dim, prompts[questionnaire_type]["dim_desc"][dim], experimenter, character_name, language_name, character_name, dim, questionnaire_name)

                if questionnaire_name == '16Personalities':
                    background_prompt = background_prompt.replace('16Personalities', '16Personalities (highly similar to MBTI)', 1)
                
                if questionnaire_type == '16Personalities':
                    dim_cls1, dim_cls2 = dim.split('/')
                    
                    output_format_prompt = prompts["general"]['two_score_output'].format(dim_cls1, dim_cls2)

                    # 等改完数据再来搞一版

                else:
                    pass 

                sys_prompt = background_prompt + output_format_prompt

                user_input = 'Our conversation is as follows:\n' + conversations + '\n'

                llm_response = get_response(sys_prompt, user_input, model=evaluator_llm)
                # 将llm_response转为json

                llm_response = llm_response.strip("`")
                if llm_response.startswith('json'):
                    llm_response = llm_response[4:]
                
                llm_response = json.loads(llm_response)

                

                try:
                    if questionnaire_type == '16Personalities':
                        llm_response['result'] = {k: int(float(v.strip("%"))) for k, v in llm_response['result'].items()}
                        assert (sum(llm_response['result'].values()) == 100)
                    else:
                        llm_response['result'] = float(llm_response['result'])
                except:
                    raise ValueError(f"Error parsing llm response {llm_response}")
                
                batch_results.append({'batch_responses': batch_responses, 'result': llm_response['result'], 'analysis': llm_response['analysis']})

            # aggregate results
            if questionnaire_type == '16Personalities':
                # use scores of dim_cls1
                all_scores = [ dim_res['result'][dim_cls1] for dim_res in batch_results]
            else:
                all_scores = [ dim_res['result'] for dim_res in batch_results]
            
            count_group = len(batch_results)

            avg_score = sum(all_scores)/count_group
            if count_group > 1:
                std_score = math.sqrt(sum([(s-avg_score)**2 for s in all_scores])/ (count_group - 1))
            else:
                std_score = None
            
            if questionnaire_type == '16Personalities':
                score = {dim_cls1: avg_score, dim_cls2: 100 - avg_score}
                pred = max(score, key=score.get)

                assessment_results[dim] = {
                    'result': pred,
                    'score': score,
                    'standard_variance': std_score,   
                    'batch_results': batch_results,
                }
            else: # bigfive
                score = avg_score
                assessment_results[dim] = {
                    'score': score,
                    'standard_variance': std_score,
                    'batch_results': batch_results,
                }

    elif evaluator_llm == 'api':
        # api is only for mbti. it does not support bigfive
        assert(questionnaire_type == '16Personalities')
        options = ['fully agree', 'generally agree', 'partially agree', 'neither agree nor disagree', 'partially disagree', 'generally disagree', 'fully disagree']
        ans_map = { option: i-3 for i, option in enumerate(options)} 

        answers = []
        for i, response in enumerate(questionnaire_results):
            sys_prompt = prompts.to_option_prompt_template.format(character_name, experimenter)

            conversations = ''
            conversations += f"{experimenter}: 「{response['question']}」\n"
            # answer
            if not response['response_open'].startswith(character_name):
                response['response_open'] = character_name + ': 「' + response['response_open'] + '」'
            conversations += f"{response['response_open']}\n"
            
            user_input = conversations

            llm_response = get_response(sys_prompt, user_input, model="gpt-3.5-turbo").strip('=\n')
            llm_response = json.loads(llm_response)

            answer = llm_response['result']

            answers.append(ans_map[answer])

        from api_16personality import submit_16personality_api
        
        pred = submit_16personality_api(answers)
        
        assessment_results = pred
    
    return assessment_results 

def personality_assessment(character, agent_type, agent_llm, questionnaire_type, eval_method, evaluator_llm='gpt-3.5-turbo', repeat_times=1):    
    if character in alias2character.keys():
        character = alias2character[character]
    else:
        raise ValueError(f"Character '{character}' not found. Here are the items: {list(alias2character.items())}") 
    
    character_name = character_info[character]["alias"][0]
    language = character[character.rfind('-')+1:]

    # load questionnaire
    if questionnaire_type in scale_list:
        questionnaire_metadata = load_questionnaire(questionnaire_type)
        questionnaire = questionnaire_metadata.pop('questions')
        
        # transform into list
        questions = []

        for idx in questionnaire:
            q = questionnaire[idx]
            q.update({'id': idx})
            questions.append(q)
        
        questionnaire = questions
    else:
        raise NotImplementedError
    
    # get experimenter
    experimenter = get_experimenter(character)

    # build character agent
    character_agent = build_character_agent(character, agent_type, agent_llm) 
    logger.info(f'Character agent created for {character_name}')

    if eval_method == 'direct':
        query = questionnaire_metadata['prompts']['direct_ask'][language]

        response = character_agent.chat(role = experimenter, text = query)
        logger.info(f'Response from {character_name}: {response} ')

        return 
    else:
        eval_args = eval_method.split('_')
        query_style = eval_args[0]
        
        
        # conduct interview with character given the questionnaire
        interview_folder_path = os.path.join('..', 'results', 'interview', '{questionnaire_type}-agent-type={agent_type}_agent-llm={agent_llm}_query-style={query_style}_repeat-times={repeat_times}')
        if not os.path.exists(interview_folder_path):
            os.makedirs(interview_folder_path)

        
        interview_save_path = f'{character}.json'
    
        interview_save_path = os.path.join(interview_folder_path, interview_save_path)
        

        if not os.path.exists(interview_save_path):
            logger.info('Interviewing...')
            
            if repeat_times < 1:
                questionnaire = subsample_questionnaire(questionnaire, n=math.ceil(len(questionnaire)*repeat_times))

            questionnaire_results = interview(character_agent, questionnaire, experimenter, questionnaire_metadata["prompts"], language, query_style)
            with open(interview_save_path, 'w') as f:
                json.dump(questionnaire_results, f, indent=4, ensure_ascii=False)
            logger.info(f'Interview finished... save into {interview_save_path}')
        else:
            logger.info(f'Interview done before. load directly from {interview_save_path}')
            with open(interview_save_path, 'r') as f:
                questionnaire_results = json.load(f)

                # reproduce the same questionnaire
                questionnaire_idx = [q['id'] for q in questionnaire_results]
                id2question = {q['id']: q for q in questionnaire}
                questionnaire = [ id2question[i] for i in questionnaire_idx]

        
        # evaluate the character's personality
        assessment_folder_path = os.path.join('..', 'results', 'assessment', '{questionnaire_type}_agent-type={agent_type}_agent-llm={agent_llm}_eval-method={eval_method}_repeat-times={repeat_times}')
        if not os.path.exists(assessment_folder_path):
            os.makedirs(assessment_folder_path)

        assessment_save_path = f'{character}.json'
    
        assessment_save_path = os.path.join(assessment_folder_path, assessment_save_path)

        if not os.path.exists(assessment_save_path):
            logger.info('Assessing...')
            assessment_results = assess(character_name, experimenter, questionnaire_results, questionnaire, questionnaire_metadata, eval_method, language, evaluator_llm)
       
            
            with open(assessment_save_path, 'w') as f:
                json.dump(assessment_results, f, indent=4, ensure_ascii=False)
            logger.info(f'Assess finished... save into {assessment_save_path}')
        else:
            logger.info(f'Assess done before. load directly from {assessment_save_path}')
            with open(assessment_save_path, 'r') as f:
                assessment_results = json.load(f)
    
            

    # show results of personality assessment 
    if questionnaire_type == '16Personalities':
        logger.info('MBTI assessment results:')
        logger.info('Character: ' + character_name)
        dims = sorted(list(set([q['dimension'] for q in questionnaire])))
        pred_code = ''.join([ assessment_results[dim]['result'] for dim in dims])
        label_code = mbti_labels[character_name]

        logger.info(f'Prediction {pred_code}\tGroundtruth {label_code}')

        for dim, result in assessment_results.items():
            if "score" in result:
                logger.info(f'{dim}: {result["score"]}')
            if "standard_variance" in result and result["standard_variance"] != None:
                logger.info(f'std: {result["standard_variance"]:.2f}')
            if "batch_results" in result:
                # analysis of the first batch
                logger.info(f'{result["batch_results"][0]["analysis"]}')
    
    else:
        logger.info('Big Five assessment results:')
        logger.info('Character: ' + character_name)

        for dim, result in assessment_results.items():
            if "score" in result:
                logger.info(f'{dim}: {result["score"]}')
            if "standard_variance" in result and result["standard_variance"] != None:
                logger.info(f'{dim}: {result["standard_variance"]:.2f}')
            if "batch_results" in result:
                # analysis of the first batch
                logger.info(f'{result["batch_results"][0]["analysis"]}')
    
if __name__ == '__main__':
    personality_assessment(
        args.character, args.agent_type, args.agent_llm, 
        args.questionnaire_type, args.eval_method, args.evaluator_llm)
            

# python personality_tests.py --eval_method direct_ask --questionnaire_type 16Personalities --character hutao
# python personality_tests.py --eval_method interview_sample --questionnaire_type 16Personalities --character hutao
# python personality_tests.py --eval_method interview_batch --questionnaire_type 16Personalities --character hutao






