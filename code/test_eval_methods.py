from personality_tests import personality_assessment

from characters import character_info
from utils import logger_main as logger

characters = character_info.keys()
questionnaires = ['BFI', '16Personalities']
agent_types = ['RoleLLM', 'ChatHaruhi']

logger.info('Start testing eval methods')

for questionnaire in questionnaires: 
    for eval_method in ['interview_convert_adjoption', 'interview_convert_adjoption_api', 'interview_convert_api', 'interview_convert', 'choose', 
                        'interview_assess_batch_anonymous', 'interview_assess_collective_anonymous']:  
        for eval_llm in ['gpt-3.5', 'gpt-4', 'gemini'] :
            for repeat_times in [0.25, 1, 2]: # 0.25 1, 
                for agent_llm in ['gpt-3.5', 'gpt-4']:
                    # if agent_llm == 'gpt-3.5' and eval_llm == 'gpt-3.5': 
                    #     continue
                    if eval_llm == 'gpt-4' and repeat_times == 2: continue #costly?
                    if agent_llm == 'gpt-4' and eval_method == 'choose': continue # costly

                    if eval_method == 'interview_convert_api':
                        if not questionnaire == '16Personalities': continue 
                        if repeat_times != 1: continue 
                    
                    # there is a bug in transformer when interleave with luotuo embeddings and bge embeddings, which may sometimes cause failure. To minimize the change of embeddings, we run haruhi and rolellm characters separately.

                    results = {}

                    for agent_type in agent_types:
                        for character in characters:
                        #for agent_type in [ a for a in character_info[character]['agent'] if a in agent_types]:
                            if not agent_type in character_info[character]['agent']: continue

                            result = personality_assessment(
                                character, agent_type, agent_llm, 
                                questionnaire, eval_method, eval_llm, repeat_times=repeat_times)
                            
                            
                            results[(character, agent_type)] = result['code']
                    
                    count_dimension = { a: 0 for a in agent_types }
                    count_correct_dimension = { a: 0 for a in agent_types }
                    
                    count_full = { a: 0 for a in agent_types }
                    count_correct_full = { a: 0 for a in agent_types }
                    
                    for (character, a), code in results.items():
                        label = character_info[character]['labels'][questionnaire]

                        full_correct = True
                        for p, l in zip(code, label):
                            if l == 'X': continue 

                            count_dimension[a] += 1
                            if p == l:
                                count_correct_dimension[a] += 1
                            else: 
                                full_correct = False

                        count_full[a] += 1
                        if full_correct: 
                            count_correct_full[a] += 1
                    
                    
                    for count in [count_dimension, count_correct_dimension, count_full, count_correct_full]:
                        count['all'] = sum(count.values())

                    logger.info('Questionnaire: {}, Eval Method: {}, Repeat Times: {}, Agent LLM: {}, Eval LLM: {}'.format(
                        questionnaire, eval_method, repeat_times, agent_llm, eval_llm))                    
                    for a in agent_types + ['all']:
                        dim_acc = count_correct_dimension[a] / count_dimension[a]
                        full_acc = count_correct_full[a] / count_full[a]

                        logger.info('Agent Type: {}, Dimension Accuracy: {:.4f}, Full Accuracy: {:.4f}'.format(
                            a, dim_acc, full_acc))
                    
                    
                        
                        
                            
            
                
            
            

    
        
    
