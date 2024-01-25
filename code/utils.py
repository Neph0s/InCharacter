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
import __main__

quiet = False

logger = logging.getLogger(__name__)

file_handler = logging.FileHandler('log.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)  # 设置日志级别
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) 
# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
if not quiet:
	logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
console_handler.setFormatter(formatter)
 
logger_main = logging.getLogger(__name__ + '_main')
file_handler_main = logging.FileHandler(f'{__main__.__file__}.log', encoding='utf-8')
logger_main.setLevel(logging.INFO)
logger_main.addHandler(file_handler_main)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler_main.setFormatter(formatter)

cache_sign = True

with open('config.json', 'r') as f:
	config = json.load(f)

#openai.proxy = config['proxy']
if config.get('proxy', None):
	openai.proxy = config['proxy']

if config.get('openai_apibase', None):
	openai.api_base = config['openai_apibase']

if config.get('gemini_apikey', None):
	import google.generativeai as genai
	genai.configure(api_key=config['gemini_apikey'])

# -----------------------------------------------------------------------------
# utilities for safe writing of a pickle file

# Context managers for atomic writes courtesy of
# http://stackoverflow.com/questions/2333872/atomic-writing-to-file-with-python
# from contextlib import contextmanager
# import tempfile
# @contextmanager
# def _tempfile(*args, **kws):
#     """ Context for temporary file.
#     Will find a free temporary filename upon entering
#     and will try to delete the file on leaving
#     Parameters
#     ----------
#     suffix : string
#         optional file suffix
#     """

#     fd, name = tempfile.mkstemp(*args, **kws)
#     os.close(fd)
#     try:
#         yield name
#     finally:
#         try:
#             os.remove(name)
#         except OSError as e:
#             if e.errno == 2:
#                 pass
#             else:
#                 raise e

# def safe_pickle_dump(obj, fname):
#     """
#     prevents a case where one process could be writing a pickle file
#     while another process is reading it, causing a crash. the solution
#     is to write the pickle file to a temporary file and then move it.
#     """
#     with open_atomic(fname, 'wb') as f:
#         pickle.dump(obj, f, -1) # -1 specifies highest binary protocol

# @contextmanager
# def open_atomic(filepath, *args, **kwargs):
#     """ Open temporary file object that atomically moves to destination upon
#     exiting.
#     Allows reading and writing to and from the same filename.
#     Parameters
#     ----------
#     filepath : string
#         the file path to be opened
#     fsync : bool
#         whether to force write the file to disk
#     kwargs : mixed
#         Any valid keyword arguments for :code:`open`
#     """
#     fsync = kwargs.pop('fsync', False)

#     original_permissions = os.stat(filepath).st_mode if os.path.exists(filepath) else None 

#     with _tempfile(dir=os.path.dirname(filepath)) as tmppath:
#         with open(tmppath, *args, **kwargs) as f:
#             yield f
#             if fsync:
#                 f.flush()
#                 os.fsync(f.fileno())
#         os.rename(tmppath, filepath)
#         if original_permissions is not None:
#             os.chmod(filepath, original_permissions)

cache = None 
def cached(func):
	def wrapper(*args, **kwargs):		
		global cache
		cache_path = 'cache.pkl'
		if cache == None:
			if not os.path.exists(cache_path):
				cache = {}
			else:
				cache = pickle.load(open(cache_path, 'rb'))  

		key = ( func.__name__, str(args), str(kwargs.items()))
		

		if (cache_sign and key in cache and cache[key] not in [None, '[TOKEN LIMIT]']):
			return cache[key]
		else:
			result = func(*args, **kwargs)
			if result != 'busy' and result != None:
				cache[key] = result
				pickle.dump(cache, open(cache_path, 'wb'))
				#safe_pickle_dump(cache, cache_path)
			return result

	return wrapper

def get_response(sys_prompt, inputs, model='gpt4', nth_generation=0):
	model = model.lower().replace(' ', '')
	if model.startswith('gpt-3.5'):
		model = 'gpt-3.5-turbo-1106'
		return get_response_gpt(sys_prompt, inputs, model, nth_generation=nth_generation)
	elif model.startswith('gpt-4'):
		model = 'gpt-4-1106-preview'
		return get_response_gpt(sys_prompt, inputs, model, nth_generation=nth_generation)
	elif model.startswith('gemini'):
		model = 'gemini-pro'
		return get_response_gemini(sys_prompt, inputs, model, nth_generation=nth_generation)

from openai import OpenAI
client = OpenAI(
    # This is the default and can be omitted
    api_key=config['openai_apikey'],
)

@cached 
def get_response_gpt(sys_prompt, inputs, model='gpt-4', retry_count=0, nth_generation=0):

	query = [ {'role': 'system', 'content': sys_prompt}]
	if len(inputs) > 0:
		query.append({'role': 'user', 'content': inputs})
	
	try:
		temperature = 0.2 if nth_generation else 0 
		logger.info('ChatGPT SysPrompt:  ' + sys_prompt[:100])
		logger.info('ChatGPT Input:  ' + inputs[:100])
		response = client.chat.completions.create(
			model= model ,  # 对话模型的名称
			messages=query,
			temperature=temperature,  # 值在[0,1]之间，越大表示回复越具有不确定性
			top_p=1,
			frequency_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
			presence_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容,
			timeout=60
		)

		logger.info('GPT Output: ' + response.choices[0].message.content[:100])
		return response.choices[0].message.content

	except openai.BadRequestError as e:
		logger.exception(e)
		
		return '[TOKEN LIMIT]'

	except Exception as e:
		# unknown exception
		logger.exception(e)

		if retry_count < 2:
			time.sleep(5)
			logger.warn("[OPEN_AI] RateLimit exceed, 第{}次重试".format(retry_count+1))
			return get_response_gpt(sys_prompt, inputs, model, retry_count+1, nth_generation) 

		print(f'Fail to get response after {retry_count} retry')

@cached 
def get_response_gemini(sys_prompt, inputs, model='gemini-pro', retry_count=0, nth_generation=0):

	try:
		gemini_model = genai.GenerativeModel(model_name=model)

		logger.info('Gemini SysPrompt:  ' + sys_prompt[:100])
		logger.info('Gemini Input:  ' + inputs[:100])
			
		response = gemini_model.generate_content(sys_prompt + inputs, generation_config=genai.types.GenerationConfig(
			candidate_count=1, 
			temperature = 0.2 if nth_generation else 0,
		))		

		if nth_generation >= 5:
			# can not get valid output
			return get_response_gpt(sys_prompt, inputs, model='gpt-4')
		
		response = response.text
		logger.info('Gemini Output: ' + response[:100])
		
		return response

	except Exception as e:
		# unknown exception
		logger.exception(e)

		if retry_count < 5:
			time.sleep(5)
			logger.warn("[GEMINI_AI] RateLimit exceed, 第{}次重试".format(retry_count+1))
			return get_response_gemini(sys_prompt, inputs, model, retry_count+1, nth_generation) 
		else:
			# blocked ... gpt instead ? 
			return get_response_gpt(sys_prompt, inputs, model='gpt-4')

def string2json(llm_response, nth_generation=0, **kwargs):
	llm_response = llm_response.strip("`")
	if llm_response.startswith('json'):
		llm_response = llm_response[4:]
	
	try:
		json_response = json.loads(llm_response)
	except:
		try:
			llm_response = llm_response[llm_response.find("{"):]
			json_response = json.loads(llm_response)
		except:
			return False
	
	return json_response

def string2json_ensure_choice_format(llm_response, nth_generation=0, **kwargs):
	if not llm_response: return False
	
	if isinstance(llm_response, str):
		json_response = string2json(llm_response)
	else:
		json_response = llm_response
		
	if json_response:
		# gemini's ift ability need further improvement. it can not output the format I want
		try:
			for idx, choice in json_response.items():
			# 如果choice是一个dict
				if choice == 'x': continue 
				
				if isinstance(choice, dict):
					choice = choice['choice']
				if isinstance(choice, str):
					choice = int(choice)

				json_response[idx] = choice
			
			return json_response
				
		except:
			logger.info('Not a valid json response for choice format')
			return False
				
	else:
		return False

def string2json_ensure_keys(llm_response, nth_generation=0, **kwargs):
	if not llm_response: return False

	if isinstance(llm_response, str):
		json_response = string2json(llm_response)
	else:
		json_response = llm_response
		
	if json_response:
		input_json = json.loads(kwargs['inputs'])
		
		missing_keys = input_json.keys() - json_response.keys()
		if missing_keys: 
			# the json response does not contain all the keys in input_json
			if nth_generation < 10:
				return False
			else:
				for k in missing_keys:
					json_response[k] = 'x'
				return json_response
		else:
			return json_response
	else:
		return False


def get_response_json(post_processing_funcs=[string2json], **kwargs):
	
	nth_generation = 0


	while (True):
		response = get_response(**kwargs, nth_generation=nth_generation)
		
		for post_processing_func in post_processing_funcs:
			response = post_processing_func(response, nth_generation, **kwargs)
		#print(f'parse results: {json_response}')
		json_response = response 
		
		if json_response:
			break 
		else:
			nth_generation += 1
			if nth_generation > 12:
				break	


	return json_response



def avg(lst):
	return sum(lst)/len(lst)

def std(lst):
	import math
	avg_score = avg(lst)
	return math.sqrt(sum([(s-avg_score)**2 for s in lst])/ (len(lst)))

def find_colon_idx(response):
	colon_idx1 = response.find(':')
	colon_idx2 = response.find('：')

	if colon_idx1 > -1 and colon_idx2 > -1:
		colon_idx = min(colon_idx1, colon_idx2)
	else:
		colon_idx = max(colon_idx1, colon_idx2) 
	return colon_idx


if __name__ == '__main__':
	print(get_response('Act as a calculator', '123+456=?', 'gpt-3.5-turbo'))
		


