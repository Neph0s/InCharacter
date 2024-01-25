from personality_tests import personality_assessment
from characters import character_info
from utils import logger_main as logger

characters = character_info.keys()
questionnaires = ['BFI', '16Personalities']
agent_typess = [['RoleLLM', 'ChatHaruhi'], ['character.ai']] # 'Character LLM'会有另一套逻辑要写.. 

logger.info('Start testing eval methods')

for eval_llm in ['gpt-3.5']: 
    for questionnaire in questionnaires: 
        for eval_method in ['interview_assess_batch_anonymous']: 
            for repeat_times in [1]: 
                for agent_types in agent_typess:
                    # 代码有点丑陋，就是2个setting，1个是Haruhi(Role和Haruhi的32个角色)，1个是c.ai
                    if agent_types == ['RoleLLM', 'ChatHaruhi']:
                        agent_llms = ['Qwen-7B', 'Qwen-7B-finetuned', 'CharacterGLM', 'gemini', 'LlaMA2', 'Qwen-1.3B', 'Phi'] # Your LLMs
                    elif agent_types == ['character.ai']:
                        agent_llms = ['character.ai']

                    for agent_llm in agent_llms: 
                        # if eval_method.endswith('api'):
                        #     if not questionnaire == '16Personalities': continue 
                        
                        # there is a bug in transformer when interleave with luotuo embeddings and bge embeddings, which may sometimes cause failure. To minimize the change of embeddings, we run haruhi and rolellm characters separately.

                        results = {}

                        for agent_type in agent_types:
                            for character in characters:
                                if not agent_type in character_info[character]['agent']: 
                                    assert(agent_type != 'character.ai')
                                    continue
                                if character == 'Sheldon-en' and agent_type == 'RoleLLM': continue 

                                result = personality_assessment(
                                    character, agent_type, agent_llm, 
                                    questionnaire, eval_method, eval_llm, repeat_times=repeat_times)
                                
                                
                                results[(character, agent_type)] = result['code']
                                #multitime_results[(character, agent_type)] = result['raw']
                        
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

                        logger.info('Questionnaire: {}, Eval Method: {}, Repeat Times: {}, Agent LLM: {}, Eval LLM: {}'.format(questionnaire, eval_method, repeat_times, agent_llm, eval_llm))                    
                        for a in agent_types + ['all']:
                            dim_acc = count_correct_dimension[a] / count_dimension[a]
                            full_acc = count_correct_full[a] / count_full[a]

                            logger.info('Agent Type: {}, Dimension Accuracy: {:.4f}, Full Accuracy: {:.4f}'.format(
                                a, dim_acc, full_acc))
                            

                    
                        
                        
                            
            
                
            
            

    
        
    
