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
from utils import logger, get_response_json, avg, std
import re 

random.seed(42)

parser = argparse.ArgumentParser(description='Assess personality of a character')

scale_list = ['Empathy', 'BFI', 'BSRI', 'EPQ-R', 'LMS', 'DTDD', 'ECR-R', 'GSE', 'ICB', 'LOT-R', 'EIS', 'WLEIS', 'CABIN', '16Personalities']

# Added choices for the questionnaire argument
parser.add_argument('--questionnaire_name', type=str, default='16Personalities', 
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
					#choices=['interview_batch', 'interview_collective', 'interview_sample'], 
					help='setting (interview_batch, interview_collective, interview_sample)')


# parser.add_argument('--language', type=str, default='cn', 
#                     choices=['cn', 'en'], 
#                     help='language, temporarily only support Chinese (cn)')

args = parser.parse_args()
print(args)

from characters import character_info, alias2character

dims_dict = {'16Personalities': ['E/I', 'S/N', 'T/F', 'P/J'], 'BFI': ['Extraversion', 'Neuroticism', 'Conscientiousness', 'Agreeableness', 'Openness'] } # we want special order


# read config.json
with open('config.json', 'r') as f:
	config = json.load(f)


def load_questionnaire(questionnaire_name):
	q_path = os.path.join('..', 'data', 'questionnaires', f'{questionnaire_name}.json')

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

	if len(input_list) < 2 * (n-1):
		return [input_list]

	result = [input_list[i:i+n] for i in range(0, len(input_list), n)]
	
	# Check the length of the last sublist
	num_to_pop = n - 1 - len(result[-1])
	for i in range(num_to_pop):
		result[-1].append(result[i].pop())
		
	return result


def build_character_agent(character_code, agent_type, agent_llm):
	from ChatHaruhi import ChatHaruhi
	
	if agent_llm.startswith('gpt-'): 
		if agent_llm.startswith('gpt-3.5'):
			agent_llm = 'gpt-3.5-turbo-1106'
		elif agent_llm.startswith('gpt-4'):
			agent_llm = 'gpt-4-1106-preview'

		os.environ["OPENAI_API_KEY"] = config['openai_apikey']

		if agent_type == 'ChatHaruhi':
			character_agent = ChatHaruhi(role_name = character_info[character_code]["agent"]["ChatHaruhi"], llm = 'openai')
		elif agent_type == 'RoleLLM':
			character_agent = ChatHaruhi( role_from_hf = f'silk-road/ChatHaruhi-from-RoleLLM/{character_info[character_code]["agent"]["RoleLLM"]}', llm = 'openai', embedding = 'bge_en')
			character_agent.role_name = 'RoleLLM/' + character_info[character_code]["agent"]["RoleLLM"]

		character_agent.llm.model = agent_llm
		character_agent.llm_type = agent_llm # just to set different keys for cache 

	character_agent.llm.chat.temperature = 0 

	return character_agent

def get_experimenter(character):    
	return character_info[character]["experimenter"]

def interview(character_agent, questionnaire, experimenter, questionnaire_prompts, language, query_style, nth_test):
	
	results = []

	for question in tqdm(questionnaire):

		# conduct interview
		character_agent.dialogue_history = []

		# get question
		if query_style == 'interview':
			q = question[f'rewritten_{language}']
		elif query_style == 'choose':
			q = questionnaire_prompts["rpa_choose_prefix"][language].replace('<statement>', question[f'origin_{language}']) + ' ' + questionnaire_prompts["rpa_choice_instruction"][language]
		else:
			raise NotImplementedError
		
		# this can cause error when changing ChatHaruhi RPA into RoleLLM RPAs, just rerun it in that case.
		response = character_agent.chat(role = experimenter, text = q, nth_test=nth_test)

			

		result = {
			'id': question['id'],
			'question':q,
			'response_open':response,
			'query_style': query_style,
		}
		
		results.append(result)
		
	return results

def assess(character_aliases, experimenter, questionnaire_results, questionnaire, questionnaire_metadata, eval_method, language, evaluator_llm, nth_test):

	character_name = character_aliases[0]

	questionnaire_name = questionnaire_metadata['name']

	dims = dims_dict.get(questionnaire_name, sorted(list(set([q['dimension'] for q in questionnaire]))))
	
	eval_args = eval_method.split('_')
	
	results = []

	if eval_args[0] == 'choose' or eval_args[1] == 'convert':
		# collect choices 

		idx2dimension = {q['id']: q['dimension'] for q in questionnaire}
		idx2category = {q['id']: q['category'] for q in questionnaire}
		id2results = {q['id']: q for q in questionnaire_results}

		options = [ str(i) for i in  range(questionnaire_metadata['range'][0], questionnaire_metadata['range'][1]+1) ]
		choices = {} 

		if eval_args[0] == 'choose':
			
			need_convert = []
			for r in questionnaire_results:
				# r = {'id': '20', 'question': '你认为"你往往担心事情会变得更糟。"这个说法适用于你吗？请用1到7的等级来回答，1代表“非常同意”，4代表“既同意也不同意”，7代表“非常不同意”。请你只回答这一个数字，不要说其他内容。', 'response_open': '胡桃: 4', 'query_style': 'choose'}
				
				# replace character name
				response = r['response_open'].replace(character_name, '') 
				if ':' in response:
					response = response.split(':', 1)[-1]
				elif '：' in response:
					response = response.split('：', 1)[-1]

				response = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5]", "", response) 
				if response in options:
					choices[r['id']] = response
				else:
					need_convert.append(r)
			
			# print('The following choices can not be recognized as numbers, need LLM convert.')
			# for r in need_convert:
			# 	print(r['response_open'])
		else:
			need_convert = questionnaire_results
		
		
		if 'adjoption' in eval_args:
			# split need_convert based on dimension 
			need_convert_dict = { d: [] for d in dims}
			for q in need_convert:
				need_convert_dict[idx2dimension[q['id']]].append(q)
			
			need_convert_list = list(need_convert_dict.values())
			
		else:
			# split need_convert, at most 60 qa pairs per batch
			need_convert_list = [ need_convert[i:i+60] for i in range(0, len(need_convert), 60)]

		for need_convert in need_convert_list:
			# process need_convert to json format
			need_convert_ = {}
			
			for r in need_convert:
				r_ = r.copy()
				r_['question'] = r_['question'].replace(questionnaire_metadata["prompts"]["rpa_choice_instruction"][language], '')
				r_.pop('id')
				if 'query_style' in r : r_.pop('query_style')
				need_convert_[r['id']] = r_
			
			need_convert = need_convert_
			
			if 'adjoption' in eval_args:
				assert(questionnaire_name in ['BFI']) #, '16Personalities']) # prompt not implemented for other questionnaires

				need_convert_dim = set([ idx2dimension[q] for q in need_convert])
				assert(len(need_convert_dim) == 1)
				need_convert_dim = need_convert_dim.pop()

				pos_adjoption = {'Extraversion': 'extroverted', 'Neuroticism': 'limbic', 'Conscientiousness': 'organized', 'Agreeableness': 'agreeable', 'Openness': 'inquisitive'}
				neg_adjoption = {'Extraversion': 'introverted', 'Neuroticism': 'calm', 'Conscientiousness': 'unstructured', 'Agreeableness': 'egocentric', 'Openness': 'non-curious'}
				
				sys_prompt = (questionnaire_metadata["prompts"]["convert_to_choice"]['en'].replace('agrees with the statement', f'displays a highly {need_convert_dim} personality') + '\n' + questionnaire_metadata["prompts"]["llm_choice_instruction"]['en']).replace('neither agree nor disagree', 'neutral').replace('disagree', neg_adjoption[need_convert_dim]).replace('agree', pos_adjoption[need_convert_dim])				
				
				sys_prompt = sys_prompt.replace('<character>', character_name)
			else:
				sys_prompt = (questionnaire_metadata["prompts"]["convert_to_choice"]['en'] + '\n' + questionnaire_metadata["prompts"]["llm_choice_instruction"]['en']).replace('<character>', character_name)
			

			if evaluator_llm.startswith('gpt'):
				# call llm to convert to choices
				converted_choices = get_response_json(sys_prompt = sys_prompt, inputs = json.dumps(need_convert, indent=4, ensure_ascii=False), model=evaluator_llm)							
			else:
				from utils import string2json_ensure_choice_format
				sys_prompt = sys_prompt + '\n===OUTPUT EXAMPLE===\n{\n    \"1\": 1,\n    ...\n    \"9\": 0\n}'
				
				converted_choices = get_response_json(string2json_ensure_choice_format, sys_prompt = sys_prompt, inputs = json.dumps(need_convert, indent=4, ensure_ascii=False), model=evaluator_llm)	
			
			

			if 'adjoption' in eval_args:
				# convert 'negative' question choices. I.e. strongly extraverted (5) -> strongly disagree (1). 
				for idx, choice in converted_choices.items():
					if idx2category[idx] == 'negative' and choice != 'x':
						converted_choices[idx] = questionnaire_metadata['range'][0] + questionnaire_metadata['range'][1] - choice
					

			assert( converted_choices.keys() == need_convert.keys() )

			choices.update(converted_choices)
		
		
		if 'api' in eval_args:
			assert(questionnaire_name == '16Personalities' and len(choices) == 60) 
			for idx, choice in choices.items():
				if choice == 'x': choice = 4 # To make it possible to call api 
				choice = float(choice)

				dim = idx2dimension[idx]
				
				score = 4 - choice # from [7, 1] to [-3, 3]

				results.append({'id': [idx], 'dim': dim, 'responses': [id2results[idx]], 'choice': choices[idx], 'score': score}) 
		else:
			for idx, choice in choices.items():
				if choice == 'x': 
					choice = (questionnaire_metadata['range'][0] + questionnaire_metadata['range'][1] ) / 2
				
				choice = float(choice)
				
				dim = idx2dimension[idx]
				category = idx2category[idx]

				if questionnaire_name == '16Personalities':   
					category = 'positive' if category == dim[0] else 'negative'
				
				if category == 'positive':
					score = choice
				else: 
					score = questionnaire_metadata['range'][0] + questionnaire_metadata['range'][1] - choice
					
				results.append({'id': [idx], 'dim': dim, 'responses': [id2results[idx]], 'choice': choices[idx], 'score': score}) 
				#dim_results = {dim: sum(scores) / len(scores) for dim, scores in dim_scores.items()}		
		
	elif eval_args[1] == 'assess':
		
		if 'anonymous' in eval_args:
			character_name = '<the participant>'
			experimenter = '<the experimenter>'

		for dim in tqdm(dims):
			
			dim_responses = [r for i, r in enumerate(questionnaire_results) if questionnaire[i]['dimension'] == dim]

			if nth_test > 0:
				random.shuffle(dim_responses)

			eval_setting = eval_args[2]

			if eval_setting == 'batch':
				# 将dim_responses分成多个子列表，每个列表3-4个元素
				dim_responses_list = split_list(dim_responses)
			else:
				dim_responses_list = [dim_responses] 

			for batch_responses in dim_responses_list:
				conversations = ''
				for i, r in enumerate(batch_responses):
					# question
					conversations += f'{i+1}.\n'
					conversations += f"{experimenter}: 「{r['question']}」\n"
					# answer
					response = r['response_open'] 
					colon_idx1 = response.find(':')
					colon_idx2 = response.find('：')

					if colon_idx1 > -1 and colon_idx2 > -1:
						colon_idx = min(colon_idx1, colon_idx2)
					else:
						colon_idx = max(colon_idx1, colon_idx2)                    

					if colon_idx == -1: # colon find in response, consider anonymous
						response = character_name + ': 「' + response + '」'
					else:
						character_aliases.append(response[:colon_idx])

					conversations += f"{response}\n"
				
				# 将character_aliases转按长度排序
				character_aliases = sorted(set(character_aliases), key=lambda x: len(x), reverse=True)
				if 'anonymous' in eval_args:
					for a in character_aliases:
						conversations = conversations.replace(a, '<the participant>')

				questionnaire_name = questionnaire_metadata["name"]

				language_name = {'zh': 'Chinese', 'en': 'English'}[language]

				
				background_prompt = prompts["general"]['background_template'].format(questionnaire_name, questionnaire_name, dim, questionnaire_metadata["prompts"]["dim_desc"][dim], experimenter, character_name, language_name, character_name, dim, questionnaire_name)

				if questionnaire_name == '16Personalities':
					background_prompt = background_prompt.replace('16Personalities', '16Personalities (highly similar to MBTI)', 1)
							
				if questionnaire_name == '16Personalities':
					dim_cls1, dim_cls2 = dim.split('/')
					
					output_format_prompt = prompts["general"]['two_score_output'].format(dim_cls1, dim_cls2)

					# 等改完数据再来搞一版
				else:
					neutural_score = (questionnaire_metadata['range'][0] + questionnaire_metadata['range'][1]) / 2
					# if neutural_score is integer 
					if neutural_score == int(neutural_score): neutural_score = int(neutural_score)

					output_format_prompt = prompts["general"]['one_score_output'].format(dim, questionnaire_name, questionnaire_metadata['range'][0], questionnaire_metadata['range'][1], questionnaire_metadata['range'][0], dim, neutural_score, questionnaire_metadata['range'][1], dim, dim)
					
					

				sys_prompt = background_prompt + output_format_prompt

				user_input = 'Our conversation is as follows:\n' + conversations + '\n'

				llm_response = get_response_json(sys_prompt=sys_prompt, inputs=user_input, model=evaluator_llm)

				if questionnaire_name == '16Personalities':
					llm_response['result'] = {k: float(v.strip("%")) for k, v in llm_response['result'].items()}
					assert (sum(llm_response['result'].values()) == 100)
					# use the score of dim_cls1
					llm_response['result'] = llm_response['result'][dim_cls1]
				else:
					llm_response['result'] = float(llm_response['result'])
												
				
				results.append({'id': [r['id'] for r in batch_responses], 'dim': dim, 'responses': batch_responses, 'score': llm_response['result'], 'analysis': llm_response['analysis']})

	# now, we have all the results. Let's aggregate them				
	assessment_results = {}

	if not 'api' in eval_args:
		# categorize results by dimension
		dim_results = { dim: [] for dim in dims } 
		for result in results:
			dim = result['dim']
			dim_results[dim].append(result)
			# result : {'dim': dim, 'responses': [id2results[idx]], 'score': score}
		
		# aggregate results in each dim 
		for dim, dim_res in dim_results.items():
			all_scores = [ result['score'] for result in dim_res]

			if questionnaire_name == '16Personalities' and not 'assess' in eval_args:
				# convert scores from [1, 7] to [0, 100]
				all_scores = [ (score - 1) / 6 * 100 for score in all_scores]

			count_group = len(all_scores)

			avg_score = avg(all_scores)

			if count_group > 1:
				std_score = std(all_scores)
			else:
				std_score = None
			
			
			assessment_results[dim] = {
				'score': avg_score, 
				'intra_std': std_score,
				'details': dim_res, 
			}
		

	else:
		# api is only for mbti. it does not support bigfive
		assert(questionnaire_name == '16Personalities')
		
		id2answer = { r['id'][0]: r['score'] for r in results}
		answers = [ id2answer[str(i)] for i in range(1, 61)]
		
		from api_16personality import submit_16personality_api
		
		pred = submit_16personality_api(answers)
		
		assessment_results = { dim: {'score': pred[dim]['score'][dim[0]]} for dim in dims }

	return assessment_results 

def personality_assessment(character, agent_type, agent_llm, questionnaire_name, eval_method, evaluator_llm='gpt-3.5-turbo', repeat_times=1):   

	
	if character in character_info.keys():
		pass
	elif character in alias2character.keys():
		character = alias2character[character]
	else:
		raise ValueError(f"Character '{character}' not found. Here are the items: {list(alias2character.items())}") 

	# load questionnaire
	if questionnaire_name in scale_list:
		questionnaire_metadata = load_questionnaire(questionnaire_name)
		questionnaire = questionnaire_metadata.pop('questions')
		
		# transform into list
		questions = []

		for idx in questionnaire:
			q = questionnaire[idx]
			q.update({'id': idx})
			if q['dimension']: 
				# remove None-dimension questions
				questions.append(q)
		
		questionnaire = questions
	else:
		raise NotImplementedError
		
	# assign a code for BFI/16P 
	dims_ = dims_dict.get(questionnaire_name, sorted(list(set([q['dimension'] for q in questionnaire]))))
	dims = dims_dict.get(questionnaire_name, sorted( c['cat_name'] for c in questionnaire_metadata['categories'] ))
	assert(dims_ == dims)

	final_folder_path = os.path.join('..', 'results', 'final', f'{questionnaire_name}_agent-type={agent_type}_agent-llm={agent_llm}_eval-method={eval_method}-{evaluator_llm}_repeat-times={repeat_times}')
	if not os.path.exists(final_folder_path):
		os.makedirs(final_folder_path)
	
	final_save_path = os.path.join(final_folder_path, f'{character}.json' )

	character_name = character_info[character]["alias"][0]
	language = character[character.rfind('-')+1:]
	
	if language == 'zh' or (not os.path.exists(final_save_path)):
		
		# get experimenter
		experimenter = get_experimenter(character)

		# build character agent
		character_agent = build_character_agent(character, agent_type, agent_llm) 
		logger.info(f'Character agent created for {character_name}')

		multitime_assess_results = []

		if eval_method == 'direct':
			query = questionnaire_metadata['prompts']['direct_ask'][language]

			response = character_agent.chat(role = experimenter, text = query)
			logger.info(f'Response from {character_name}: {response} ')

			return 
		else:
			eval_args = eval_method.split('_')
			query_style = eval_args[0]
			
			if repeat_times < 1: 
				subsample_questionnaire_folder_path = os.path.join('..', 'data', 'subsample_questionnaire', f'ratio={repeat_times}')
				if not os.path.exists(subsample_questionnaire_folder_path):
					os.makedirs(subsample_questionnaire_folder_path)

				subsample_questionnaire_path = os.path.join(subsample_questionnaire_folder_path, f'{questionnaire_name}.json')

				if not os.path.exists(subsample_questionnaire_path):
					questionnaire = subsample_questionnaire(questionnaire, n=math.ceil(len(questionnaire)*repeat_times))
					with open(subsample_questionnaire_path, 'w') as f:
						json.dump(questionnaire, f, indent=4, ensure_ascii=False)
					
					logger.info(f'Subsample questionnaire and save into {subsample_questionnaire_path}')
				else:
					logger.info(f'Load subsampled questionnaire from {subsample_questionnaire_path}')
					with open(subsample_questionnaire_path, 'r') as f:
						questionnaire = json.load(f)
			
			# conduct interview with character given the questionnaire
			interview_folder_path = os.path.join('..', 'results', 'interview', f'{questionnaire_name}-agent-type={agent_type}_agent-llm={agent_llm}_query-style={query_style}')
			if not os.path.exists(interview_folder_path):
				os.makedirs(interview_folder_path)

			for nth_test in range(max(repeat_times, 1)):			
				if repeat_times < 1:
					interview_save_path = f'{character}_{repeat_times}-test.json'
				else:
					interview_save_path = f'{character}_{nth_test}-test.json'

				interview_save_path = os.path.join(interview_folder_path, interview_save_path)
				
				if not os.path.exists(interview_save_path):
					logger.info('Interviewing...')

					questionnaire_results = interview(character_agent, questionnaire, experimenter, questionnaire_metadata["prompts"], language, query_style, nth_test)
					with open(interview_save_path, 'w') as f:
						json.dump(questionnaire_results, f, indent=4, ensure_ascii=False)
					logger.info(f'Interview finished... save into {interview_save_path}')
				else:
					logger.info(f'Interview done before. load directly from {interview_save_path}')
					with open(interview_save_path, 'r') as f:
						questionnaire_results = json.load(f)

				
				# evaluate the character's personality
				assessment_folder_path = os.path.join('..', 'results', 'assessment', f'{questionnaire_name}_agent-type={agent_type}_agent-llm={agent_llm}_eval-method={eval_method}-{evaluator_llm}')

				if not os.path.exists(assessment_folder_path):
					os.makedirs(assessment_folder_path)

				if repeat_times < 1:
					assessment_save_path = f'{character}_{repeat_times}th-test.json'
				else:
					assessment_save_path = f'{character}_{nth_test}th-test.json'
			
				assessment_save_path = os.path.join(assessment_folder_path, assessment_save_path)
			
				if language == 'zh' or (not os.path.exists(assessment_save_path)):
					logger.info('Assessing...')
					assessment_results = assess(character_info[character]["alias"], experimenter, questionnaire_results, questionnaire, questionnaire_metadata, eval_method, language, evaluator_llm, nth_test)
			
					with open(assessment_save_path, 'w') as f:
						json.dump(assessment_results, f, indent=4, ensure_ascii=False)
					logger.info(f'Assess finished... save into {assessment_save_path}')
				else:
					logger.info(f'Assess done before. load directly from {assessment_save_path}')
					with open(assessment_save_path, 'r') as f:
						assessment_results = json.load(f)
				
				multitime_assess_results.append(assessment_results)
		

				
		# show results of personality assessment 
		logger.info(f'{questionnaire_name} assessment results:')
		logger.info('Character: ' + character_name)

		# average multitime_assess_results
		# traverse all possible structures of assessment_results
		assessment_results = {}
		for dim in multitime_assess_results[0].keys():
			
			a_results_keys = multitime_assess_results[0][dim].keys()

			assessment_results[dim] = {
				'score': avg([a_results[dim]['score'] for a_results in multitime_assess_results]),
			}
			
			if repeat_times > 1:
				assessment_results[dim]['inter_std'] = std([a_results[dim]['score'] for a_results in multitime_assess_results])

			if 'intra_std' in a_results_keys:
				assessment_results[dim]['intra_std'] = [a_results[dim]['intra_std'] for a_results in multitime_assess_results]
			
			if 'details' in a_results_keys:
				assessment_results[dim]['details'] = [a_results[dim]['details'] for a_results in multitime_assess_results]
			

		# assign a code for BFI/16P 
		# dims = dims_dict.get(questionnaire_name, sorted(list(set([q['dimension'] for q in questionnaire]))))
		
		if questionnaire_name in ['BFI', '16Personalities']:
			thresh = (questionnaire_metadata['range'][0] + questionnaire_metadata['range'][1]) / 2
			if questionnaire_name == '16Personalities': 
				thresh = 50
				pos_tags = { dim: dim[0] for dim in dims}
				neg_tags = { dim: dim[-1] for dim in dims}
			elif questionnaire_name == 'BFI':
				pos_tags = {'Extraversion': 'S', 'Neuroticism': 'L', 'Conscientiousness': 'O', 'Agreeableness': 'A', 'Openness': 'I'}
				neg_tags = {'Extraversion': 'R', 'Neuroticism': 'C', 'Conscientiousness': 'U', 'Agreeableness': 'E', 'Openness': 'N'}

			code = ''
			
			for dim in pos_tags.keys():
				result = assessment_results[dim]
				if result['score'] > thresh:
					code += pos_tags[dim]
				else:
					code += neg_tags[dim]

			assessment_results['code'] = code 
		
		logger.info(f'Score range: {questionnaire_metadata["range"]}')

		if 'code' in assessment_results:
			pred_code = assessment_results['code']
			logger.info(f'Prediction {pred_code}')
		
		if questionnaire_name in character_info[character]["labels"]: 
			label_code = character_info[character]["labels"][questionnaire_name]
			logger.info(f'Groundtruth {label_code}')

		for dim, result in assessment_results.items():
			dim_result_info = ''
			if "score" in result:
				if questionnaire_name == '16Personalities':
					dim_result_info += f'{dim[0]}: {result["score"]:.2f}\t{dim[-1]}: {(100 - result["score"]):.2f}\t'
				else:
					dim_result_info += f'{dim}: {result["score"]:.2f}\t'
			
			if "inter_std" in result and result["inter_std"] != None:
				dim_result_info += f'inter std: {result["inter_std"]:.2f}\t'
			
			if "intra_std" in result and result["intra_std"] != None:
				dim_result_info += f'intra std: {result["intra_std"]}\t'

			if False: #"details" in result:
				if 'analysis' in result["details"][0][0]:
					# analysis of the first batch
					dim_result_info += f'{result["details"][0][0]["analysis"]}\t'
				else:
					dim_result_info += f'{result["details"][0][0]["responses"][0]}\t'
				
					
			
			logger.info(dim_result_info)
	
	
		logger.info(f'Save final results into {final_save_path}')

		with open(final_save_path, 'w') as f:
			json.dump(assessment_results, f, indent=4, ensure_ascii=False)
	
	else:
		logger.info(f'Load final results from {final_save_path}')

		with open(final_save_path, 'r') as f:
			assessment_results = json.load(f)
		
		
	
	for dim in dims:
		if 'details' in assessment_results[dim].keys():
			assessment_results[dim].pop('details')

	return assessment_results

	
if __name__ == '__main__':
	personality_assessment(
		args.character, args.agent_type, args.agent_llm, 
		args.questionnaire_name, args.eval_method, args.evaluator_llm)
			

# python personality_tests.py --eval_method direct_ask --questionnaire_name 16Personalities --character hutao
# python personality_tests.py --eval_method interview_sample --questionnaire_name 16Personalities --character hutao
# python personality_tests.py --eval_method interview_batch --questionnaire_name 16Personalities --character hutao
# python personality_tests.py --eval_method interview_assess_batch_anonymous --questionnaire_name 16Personalities --character haruhi-zh
#python personality_tests.py --eval_method interview_convert --questionnaire_name 16Personalities --character haruhi-zh





