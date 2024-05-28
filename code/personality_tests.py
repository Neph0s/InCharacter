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

rerun = False #True

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
					choices=['ChatHaruhi', 'RoleLLM'], 
					help='agent type (haruhi by default)')

# Added choices for the agent_llm argument
parser.add_argument('--agent_llm', type=str, default='gpt-3.5-turbo', 
					#choices=['gpt-3.5-turbo', 'openChat', 'mistral', 'ChatGLM2GPT',"qwen-118k","llama2","Mixtral"], 
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
#print(args)

problem_types = ['is_multilanguage', 'not_into_character', 'contain_repeation', 'is_multiround'] 

from characters import character_info, alias2character, character_labels

dims_dict = {'16Personalities': ['E/I', 'S/N', 'T/F', 'P/J'], 'BFI': ['Extraversion', 'Neuroticism', 'Conscientiousness', 'Agreeableness', 'Openness'] } # we want special order

previous_file_path = ''

# read config.json
with open('config.json', 'r') as f:
	config = json.load(f)


def load_questionnaire(questionnaire_name):
	q_path = os.path.join('..', 'data', 'questionnaires', f'{questionnaire_name}.json')

	# read this jsonl file
	with open(q_path, 'r', encoding='utf-8') as f:
		questionnaire = json.load(f)
	
	if questionnaire_name not in dims_dict:
		dims_dict[questionnaire_name] = [ _['cat_name'] for _ in  questionnaire['categories'] ]
		
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
	
	agent_type_args = agent_type.split('=', 1)

	if agent_llm.startswith('gpt-'): 
		if agent_llm.startswith('gpt-3.5'):
			agent_llm = 'gpt-3.5-turbo-1106'
		elif agent_llm.startswith('gpt-4'):
			agent_llm = 'gpt-4-1106-preview'
	

	os.environ["OPENAI_API_KEY"] = config['openai_apikey']
	
	if agent_type_args[0] == 'ChatHaruhi':
		character_agent = ChatHaruhi(role_name = character_info[character_code]["agent"]["ChatHaruhi"], llm = agent_llm)
	elif agent_type_args[0] == 'RoleLLM':
		character_agent = ChatHaruhi( role_from_hf = f'silk-road/ChatHaruhi-from-RoleLLM/{character_info[character_code]["agent"]["RoleLLM"]}', llm = agent_llm, embedding = 'bge_en')
		character_agent.role_name = 'RoleLLM/' + character_info[character_code]["agent"]["RoleLLM"]

	character_agent.nickname = character_info[character_code]['alias'][0]

	character_agent.llm.model = agent_llm

	character_agent.llm_type = agent_llm # just to set different keys for cache 
			
	
	
	if len(agent_type_args) > 1:
		character_agent.llm_type = character_agent.llm_type + '=' + agent_type_args[1]

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
		elif query_style.startswith('choose'):
			q = questionnaire_prompts["rpa_choose_prefix"][language].replace('<statement>', question[f'origin_{language}']) + ' ' + questionnaire_prompts["rpa_choice_instruction"][language]

			if query_style == 'choosecot':
				if language == 'en':
					q = q.replace('Please answer with the number only, without anything else.', 'Please think step by step. Start by sharing your thoughts, then proceed to present the number.')
				else:
					q = q.replace('请你只回答这一个数字，不要说其他内容。', '请给出你的理由。')
				


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

def assess(character_aliases, experimenter, questionnaire_results, questionnaire, questionnaire_metadata, eval_method, language, evaluator_llm, nth_test, agent_llm):

	character_name = character_aliases[0]

	questionnaire_name = questionnaire_metadata['name']

	dims = dims_dict.get(questionnaire_name, sorted(list(set([q['dimension'] for q in questionnaire]))))
	
	eval_args = eval_method.split('_')
	
	results = []

	from utils import find_colon_idx

	if agent_llm.startswith('gpt'):
		# collect character aliases
		for r in questionnaire_results:
			response = r['response_open']
			colon_idx = find_colon_idx(response)
			if colon_idx != -1:			
				character_aliases.append(response[:colon_idx])
	
	character_aliases = sorted(set(character_aliases), key=lambda x: len(x), reverse=True)

	error_counts = None 
	if not ( agent_llm.startswith('gpt') ):
		error_counts = { k: 0 for k in problem_types }

	# correct response without speaker name
	
	
	for r in questionnaire_results:
		response = r['response_open']
		question = r['question']
		
		if not ( agent_llm.startswith('gpt') ):
			from utils import is_multiround, is_multilanguage, not_into_character, contain_repeation, truncate
			#import pdb; pdb.set_trace()
			
			if is_multilanguage(question, response):
				error_counts['is_multilanguage'] = error_counts.get('is_multilanguage', 0) + 1
				# print(f'{response}\nis_multilanguage')
				# import pdb; pdb.set_trace()
			if not_into_character(response, experimenter):
				error_counts['not_into_character'] = error_counts.get('not_into_character', 0) + 1
				# print(f'{response}\nnot_into_character')
				# import pdb; pdb.set_trace()
			if contain_repeation(response):
				error_counts['contain_repeation'] = error_counts.get('contain_repeation', 0) + 1
				# print(f'{response}\ncontain_repeation')
				# import pdb; pdb.set_trace()
				
				response = contain_repeation(response)
			if is_multiround(response):
				# print(f'is_multiround: {response}')
				# import pdb; pdb.set_trace()
				error_counts['is_multiround'] = error_counts.get('is_multiround', 0) + 1
				response = is_multiround(response)

			
			response = truncate(response)
		
		r['response_open'] = response
		colon_idx = find_colon_idx(response)
		
		if colon_idx == -1 and not any([response.startswith(a) for a in character_aliases]):
			r['response_open'] = character_name + ': 「' + r['response_open'].strip('「」 :') + '」'
		
		
	global previous_file_path
	previous_file_path_cp = previous_file_path.replace('../results/assessment/', '../results/assessment_cp/')
	if os.path.exists(previous_file_path_cp):
		previous_file_path = previous_file_path_cp

	if True and os.path.exists(previous_file_path):
		
		with open(previous_file_path, 'r') as f:
			previous_file_results = json.load(f)
	
		
		results = []
		for dim in dims:
			results += previous_file_results[dim]['details']
	
	elif eval_args[0].startswith('choose') or eval_args[1] == 'convert':
		# collect choices 

		idx2dimension = {q['id']: q['dimension'] for q in questionnaire}
		idx2category = {q['id']: q['category'] for q in questionnaire}
		id2results = {q['id']: q for q in questionnaire_results}

		options = [ str(i) for i in  range(questionnaire_metadata['range'][0], questionnaire_metadata['range'][1]+1) ]
		choices = {} 

		if eval_args[0].startswith('choose'):
			
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

				need_convert_dim = set([ idx2dimension[q] for q in need_convert])
				assert(len(need_convert_dim) == 1)
				need_convert_dim = need_convert_dim.pop()

				
				sys_prompt = (questionnaire_metadata["prompts"]["convert_to_choice"]['en'].replace('agrees with the statement', f'displays a highly {need_convert_dim} personality') + '\n' + questionnaire_metadata["prompts"]["llm_choice_instruction_adjoption"][need_convert_dim]['en'])
				
				sys_prompt = sys_prompt.replace('<character>', character_name)
			else:
				sys_prompt = (questionnaire_metadata["prompts"]["convert_to_choice"]['en'] + '\n' + questionnaire_metadata["prompts"]["llm_choice_instruction"]['en']).replace('<character>', character_name)
			
			# control batch size
			from utils import num_tokens_from_messages

			if evaluator_llm == 'gpt-3.5' and num_tokens_from_messages([{
				"role": "user", "content": json.dumps(need_convert, indent=4, ensure_ascii=False)
				}], 'gpt-3.5-turbo-1106') > 15500:
				
				need_convert_list = [ {str(j+1): need_convert[str(j+1)] for j in range(i, i+30)} for i in range(0, len(need_convert), 30)]
			else:
				need_convert_list = [ need_convert ]

			for need_convert in need_convert_list:
				user_input = json.dumps(need_convert, indent=4, ensure_ascii=False)

				if 'anonymous' in eval_args:
					for a in character_aliases:
						sys_prompt = sys_prompt.replace(a, '<the participant>')
						user_input = user_input.replace(a, '<the participant>')
					sys_prompt = sys_prompt.replace(experimenter, '<the experimenter>')
					user_input = user_input.replace(experimenter, '<the experimenter>')

				from utils import string2json_ensure_keys
				

				if evaluator_llm.startswith('gpt'):
					# call llm to convert to choices
					converted_choices = get_response_json([string2json_ensure_keys], sys_prompt = sys_prompt, inputs = user_input, model=evaluator_llm)		

											
				else:
					from utils import string2json_ensure_choice_format
					sys_prompt = sys_prompt + '\n===OUTPUT EXAMPLE===\n{\n    \"1\": 1,\n    ...\n    \"9\": 0\n}===My Input Is==='
					
					
					converted_choices = get_response_json([string2json_ensure_choice_format, string2json_ensure_keys], sys_prompt = sys_prompt, inputs = user_input, model=evaluator_llm)	
					
						
				

				if 'adjoption' in eval_args:
					# convert 'negative' question choices. I.e. strongly extraverted (5) -> strongly disagree (1). 
					for idx, choice in converted_choices.items():
						dim = idx2dimension[idx]
						category = idx2category[idx]
						
						if questionnaire_name == '16Personalities':   
							category = 'positive' if category == dim[0] else 'negative' 

						if category == 'negative' and choice != 'x':
							converted_choices[idx] = questionnaire_metadata['range'][0] + questionnaire_metadata['range'][1] - float(choice)
						
							
				try:
					assert( len(need_convert.keys() - converted_choices.keys()) == 0 )
				except:
					import pdb; pdb.set_trace()
		
				choices.update(converted_choices)
		

		for idx, choice in choices.items():
			if choice == 'x' or choice is None: 
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
				
			results.append({'id': [idx], 'dim': dim, 'responses': [id2results[idx]], 'choice': choice, 'score': score}) 
					
		
	elif eval_args[1] == 'assess':

		for dim in tqdm(dims):
			
			dim_responses = [r for i, r in enumerate(questionnaire_results) if questionnaire[i]['dimension'] == dim]
			
			if nth_test > 0:
				random.seed(nth_test)
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
					conversations += f"{response}\n"
				
				questionnaire_name = questionnaire_metadata["name"]

				language_name = {'zh': 'Chinese', 'en': 'English'}[language]

				
				background_prompt = prompts["general"]['background_template'].format(questionnaire_name, questionnaire_name, dim, questionnaire_metadata["prompts"]["dim_desc"][dim], experimenter, character_name, language_name, character_name, dim, questionnaire_name)

				if questionnaire_name == '16Personalities':
					background_prompt = background_prompt.replace('16Personalities', '16Personalities (highly similar to MBTI)', 1)
							
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
				
			

				if 'anonymous' in eval_args:
					for a in character_aliases:
						sys_prompt = sys_prompt.replace(a, '<the participant>')
						user_input = user_input.replace(a, '<the participant>')
					sys_prompt = sys_prompt.replace(experimenter, '<the experimenter>')
					user_input = user_input.replace(experimenter, '<the experimenter>')
					sys_prompt = sys_prompt.replace('I ', 'I (<the experimenter>) ', 1)
				
				

				user_input = user_input.replace(character_name, '<the participant>')

				# for evaluating the personality of vanilla GPT
				bad_words = ['as an AI language model,', 'As an AI language model,', 'As an AI,', 'as an AI,', 'I am an AI language model,', 'being an AI,']

				for bad_word in bad_words:
					user_input = user_input.replace(bad_word, '')
				
				sys_prompt = sys_prompt.replace("Other numbers in this range represent different degrees of 'Conscientiousness'.", "Other numbers in this range represent different degrees of 'Conscientiousness'. You must give a score, and you are not allowed to give answers like 'N/A' and 'not applicable'.", 1)

				llm_response = get_response_json(sys_prompt=sys_prompt, inputs=user_input, model=evaluator_llm)
				
				
					

				if questionnaire_name == '16Personalities':
					llm_response['result'] = {k: float(str(v).strip("%")) for k, v in llm_response['result'].items()}
					
						
					assert (sum(llm_response['result'].values()) == 100)
					# use the score of dim_cls1
					llm_response['result'] = llm_response['result'][dim_cls1]
				else:
					if llm_response['result']:
						try:
							llm_response['result'] = float(llm_response['result'])
						except:
							llm_response['result'] = (questionnaire_metadata['range'][0] + questionnaire_metadata['range'][1]) / 2
							
					else:
						llm_response['result'] = (questionnaire_metadata['range'][0] + questionnaire_metadata['range'][1]) / 2

						
												
				
				results.append({'id': [r['id'] for r in batch_responses], 'dim': dim, 'responses': batch_responses, 'score': llm_response['result'], 'analysis': llm_response['analysis']})


	# now, we have all the results. Let's aggregate them				
	assessment_results = {}
	# categorize results by dimension

	dim_results = { dim: [] for dim in dims } 
	for result in results:
		dim = result['dim']
		dim_results[dim].append(result)
		# result : {'dim': dim, 'responses': [id2results[idx]], 'score': score, 'choice' (optional): choice}
	
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
		

	if questionnaire_name == '16Personalities' and not 'assess' in eval_args:
		# use api
		
		# api is only for 16p. it does not support bigfive
		assert(questionnaire_name == '16Personalities')
		

		id2answer = { r['id'][0]: r['choice'] for r in results}
		answers = [ 4 - int(id2answer[str(i)]) if id2answer[str(i)] != 'x' else 0 for i in range(1, 61) ]
		
			
		
		from api_16personality import submit_16personality_api
		pred = submit_16personality_api(answers)
		
			
		
		#assessment_results = { dim: {'score': pred[dim]['score'][dim[0]]} for dim in dims }
		for dim in dims:
			#print('Old {} New {}'.format(assessment_results[dim]['score'], pred[dim]['score'][dim[0]]))	
			assessment_results[dim]['score'] = pred[dim]['score'][dim[0]]

	if error_counts:
		assessment_results['error_counts'] = error_counts

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
		print(f'Questionnaire {questionnaire_name} not found. Here are the items: {scale_list}')
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
	
	eval_args = eval_method.split('_')

	if rerun or not os.path.exists(final_save_path): 
		# need to get multitime assessment results

		# get experimenter
		experimenter = get_experimenter(character)

		multitime_assess_results = []

		character_agent = None

		if eval_method == 'direct':
			if not character_agent:
				# build character agent
				character_agent = build_character_agent(character, agent_type, agent_llm) 
				logger.info(f'Character agent created for {character_name}')

			query = questionnaire_metadata['prompts']['direct_ask'][language]

			response = character_agent.chat(role = experimenter, text = query)
			logger.info(f'Response from {character_name}: {response} ')

			#return 
		else:
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
			if agent_llm != 'cAI':
				interview_folder_path = os.path.join('..', 'results', 'interview', f'{questionnaire_name}-agent-type={agent_type}_agent-llm={agent_llm}_query-style={query_style}')
			else:
				interview_folder_path = os.path.join('..', 'results', 'interview', f'{questionnaire_name}-agent-type=cAI_agent-llm=gpt-3.5-turbo_query-style={query_style}')
			
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

					if not character_agent:
						# build character agent
						character_agent = build_character_agent(character, agent_type, agent_llm) 
						logger.info(f'Character agent created for {character_name}')

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
			
				if rerun or not os.path.exists(assessment_save_path): 
					global previous_file_path 
					previous_file_path = assessment_save_path
					
					logger.info('Assessing...')
					assessment_results = assess(character_info[character]["alias"], experimenter, questionnaire_results, questionnaire, questionnaire_metadata, eval_method, language, evaluator_llm, nth_test, agent_llm)
			
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
		assessment_results = {
			'dims': {},
			'analysis': {},
			'code': ''
		}

		if 'error_counts' in  multitime_assess_results[0]:
		
			assessment_results['error_counts'] = { k: sum([a['error_counts'].get(k, 0) for a in multitime_assess_results]) for k in  problem_types}

		if 'assess' not in eval_method:
			multitime_item_results = {}

		for dim in dims: #multitime_assess_results[0].keys():
			
			a_results_keys = multitime_assess_results[0][dim].keys()
			try:
				assessment_results['dims'][dim] = {
					'score': avg([a_results[dim]['score'] for a_results in multitime_assess_results]),
					'all_scores': [a_results[dim]['score'] for a_results in multitime_assess_results]
				}
			except:
				import pdb; pdb.set_trace()
				
			
			if repeat_times > 1:
				assessment_results['dims'][dim]['inter_std'] = std([a_results[dim]['score'] for a_results in multitime_assess_results])

			if 'intra_std' in a_results_keys:
				assessment_results['dims'][dim]['intra_std'] = [a_results[dim]['intra_std'] for a_results in multitime_assess_results]
			
			if 'details' in a_results_keys:
				assessment_results['dims'][dim]['details'] = [a_results[dim]['details'] for a_results in multitime_assess_results]
			
			if 'assess' not in eval_method:
				for a_results in multitime_assess_results:
					for item in a_results[dim]['details']:
						assert(len(item['id']) == 1)
						item_id = item['id'][0]
						multitime_item_results[item_id] = multitime_item_results.get(item_id, []) + [item['score']]
		
		score_span = questionnaire_metadata['range'][1] - questionnaire_metadata['range'][0]

		if questionnaire_name == '16Personalities':
			score_span2 = 100
		else:
			score_span2 = score_span
		

		if 'assess' not in eval_method:
			assessment_results['analysis']['item_consistency'] = avg([ std(scores) for item_id, scores in multitime_item_results.items()]) / score_span

			assessment_results['analysis']['dim_consistency'] = avg([ avg(assessment_results['dims'][dim]['intra_std']) for dim in dims]) / score_span2
		
		if repeat_times > 1:
			assessment_results['analysis']['robustness'] = avg([ assessment_results['dims'][dim]['inter_std'] for dim in dims]) / score_span2
		
	

		# assign a code for BFI/16P 
		# dims = dims_dict.get(questionnaire_name, sorted(list(set([q['dimension'] for q in questionnaire]))))
		
		if questionnaire_name in ['BFI', '16Personalities']:
			label_settings = ['pdb', 'annotation']
		else:
			label_settings = ['annotation']


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
				result = assessment_results['dims'][dim]
				if result['score'] > thresh:
					code += pos_tags[dim]
				else:
					code += neg_tags[dim]

			assessment_results['code'] = code 
			logger.info(f'Prediction {code}')

			for label_setting in label_settings:
				labels = character_labels[label_setting][character][questionnaire_name]

				label_code = '' 
				for dim in dims:
					l = labels[dim]['type']
					if l == 'X': 
						label_code += 'X'
					elif l == 'H':
						label_code += pos_tags[dim]
					elif l == 'L':
						label_code += neg_tags[dim]
					else:
						import pdb; pdb.set_trace()
						
						raise NotImplementedError
				
				logger.info(f'{label_setting} Label {label_code}')

				
			
		logger.info(f'Score range: {questionnaire_metadata["range"]}')

		for dim in dims:
			result = assessment_results['dims'][dim]

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
									
			logger.info(dim_result_info)
		
		for analysis, result in assessment_results['analysis'].items():
			logger.info(f'{analysis}: {result:.5f}')
		
	
		logger.info(f'Save final results into {final_save_path}')

		with open(final_save_path, 'w') as f:
			json.dump(assessment_results, f, indent=4, ensure_ascii=False)

		
	else:
		logger.info(f'Load final results from {final_save_path}')

		with open(final_save_path, 'r') as f:
			assessment_results = json.load(f)
		
	
	# for dim in dims:
	# 	if 'details' in assessment_results['dims'][dim].keys():
	# 		assessment_results['dims'][dim].pop('details')

	return assessment_results

def calculate_measured_alignment(preds, labels, questionnaire_name, labels_pdb):
	assert(preds.keys() == labels.keys())	

	repeat_times = len(list(list(preds.values())[0].values())[0])

	agent_types = list(set([ rpa[1] for rpa in preds.keys()]))
	
	dims = dims_dict[questionnaire_name] 
	
	questionnaire_metadata = load_questionnaire(questionnaire_name) 

	if questionnaire_name == '16Personalities':
		range_max = 100
		range_min = 0
	else:
		range_max = questionnaire_metadata['range'][1]
		range_min = questionnaire_metadata['range'][0]

	range_middle = (range_max + range_min) / 2
	range_span = range_max - range_min
	
	multitime_metrics = []

	for nth_test in range(repeat_times):

		sum_mse_each_dim = { a: { d: 0 for d in dims } for a in agent_types }
		sum_mae_each_dim = { a: { d: 0 for d in dims } for a in agent_types }
		correct_single_each_dim = { a: { d: 0 for d in dims } for a in agent_types }
		correct_full = { a: 0 for a in agent_types }

		count_single_each_dim = { a: { d: 0 for d in dims } for a in agent_types }
		count_full = { a: 0 for a in agent_types }

		for rpa in preds.keys():
			pred= preds[rpa]
				
			label = labels[rpa]


			a = rpa[1]

			full_correct = True
			full_X = True 

			for dim in label.keys():
				label_score = label[dim]['score']
				label_type = label[dim]['type']
				
				if labels_pdb[rpa][dim]['type'] == 'X': 
					continue
				else:
					full_X = False

				pred_score = pred[dim][nth_test]
				pred_type = 'H' if pred_score > range_middle else 'L'

				count_single_each_dim[a][dim] += 1 

				if pred_type == label_type:
					correct_single_each_dim[a][dim] += 1
				else:
					full_correct = False

				sum_mse_each_dim[a][dim] += ((pred_score - label_score) / range_span) ** 2
				
				sum_mae_each_dim[a][dim] += abs((pred_score - label_score) / range_span) 

			if not full_X:
				if full_correct: 
					correct_full[a] += 1
					# print(rpa)
					# print(f'Pred {pred} Label {label}')
				count_full[a] += 1
		
		# aggreagate individual dims 
		for count in [sum_mse_each_dim, sum_mae_each_dim, correct_single_each_dim, count_single_each_dim]:
			for a in agent_types:
				count[a]['all'] = sum(count[a].values()) 

		for count in [sum_mse_each_dim, sum_mae_each_dim, correct_single_each_dim, correct_full, count_single_each_dim, count_full]:
			if isinstance( count[agent_types[0]], dict):
				count['all'] = {}
				for dim in dims + ['all']:
					count['all'][dim] = sum(count[a][dim] for a in agent_types)
			else:
				count['all'] = sum(count.values())
		
		
		metrics = {}
				
		for a in agent_types + ['all']:
			single_acc = {}
			single_mse = {}
			single_mae = {}

			for dim in dims + ['all']:
				single_acc[dim] = correct_single_each_dim[a][dim] / count_single_each_dim[a][dim] 
				single_mse[dim] = sum_mse_each_dim[a][dim] / count_single_each_dim[a][dim]
				single_mae[dim] = sum_mae_each_dim[a][dim] / count_single_each_dim[a][dim]
			
			full_acc = correct_full[a] / count_full[a]
			
			metrics[a] = { 'single_acc': single_acc, 'single_mse': single_mse, 'single_mae': single_mae, 'full_acc': full_acc}

		multitime_metrics.append(metrics)


	
	#print(count_single_each_dim)

	for a in agent_types + ['all']:
		for metric in metrics[a].keys():
			if isinstance(metrics[a][metric], dict):
				for dim in dims + ['all']:
					metrics[a][metric][dim] = avg([ m[a][metric][dim] for m in multitime_metrics])
			else:
				metrics[a][metric] = avg([ m[a][metric] for m in multitime_metrics])

	
	return metrics

		
	
if __name__ == '__main__':
	eval_method_map = {
		'self_report': 'choose',
		'self_report_cot': 'choosecot',
		'expert_rating': 'interview_assess_batch_anonymous',
		'expert_rating_collective': 'interview_assess_collective_anonymous',
		'option_conversion': 'interview_convert',
		'dimension_option_conversion': 'interview_convert_adjoption_anonymous'
	}

	args.eval_method = eval_method_map.get(args.eval_method, args.eval_method)
	
	
	personality_assessment(
		args.character, args.agent_type, args.agent_llm, 
		args.questionnaire_name, args.eval_method, args.evaluator_llm)
	
	
			

# python personality_tests.py --eval_method direct_ask --questionnaire_name 16Personalities --character hutao
# python personality_tests.py --eval_method interview_sample --questionnaire_name 16Personalities --character hutao
# python personality_tests.py --eval_method interview_batch --questionnaire_name 16Personalities --character hutao
# python personality_tests.py --eval_method interview_assess_batch_anonymous --questionnaire_name 16Personalities --character haruhi-zh
#python personality_tests.py --eval_method interview_convert --questionnaire_name 16Personalities --character haruhi-zh





