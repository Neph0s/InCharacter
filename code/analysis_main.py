from personality_tests import personality_assessment

from characters import character_info

characters = character_info.keys()
questionnaires = ['BFI', '16Personalities']
agent_types = ['RoleLLM', 'ChatHaruhi']

dims_dict = {'16Personalities': ['E/I', 'S/N', 'T/F', 'P/J'], 'BFI': ['Extraversion', 'Neuroticism', 'Conscientiousness', 'Agreeableness', 'Openness'] } # we want special order

def load_assessment_results(character, agent_type, agent_llm, 
                                questionnaire, eval_method, eval_llm, repeat_times):
    import os 
    import pdb 
    import json 
    from personality_tests import load_questionnaire 
    
    questionnaire = load_questionnaire(questionnaire)

    questionnaire_name = questionnaire['name']
    evaluator_llm = eval_llm
    # evaluate the character's personality
    assessment_folder_path = os.path.join('..', 'results', 'assessment', f'{questionnaire_name}_agent-type={agent_type}_agent-llm={agent_llm}_eval-method={eval_method}-{evaluator_llm}')

    multitime_assess_results = []

    for nth_test in range(repeat_times):
        assessment_save_path = f'{character}_{nth_test}th-test.json'

        assessment_save_path = os.path.join(assessment_folder_path, assessment_save_path)

        if (not os.path.exists(assessment_save_path)): 
            print(f'Assess not done before. Skip {assessment_save_path}')
        else:
            #print(f'Assess done before. load directly from {assessment_save_path}')
            with open(assessment_save_path, 'r') as f:
                assessment_results = json.load(f)
        
        multitime_assess_results.append(assessment_results)
    
    return multitime_assess_results

def get_final_assessment(multitime_assess_results, questionnaire):
    from utils import avg, std 

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
    
    from personality_tests import load_questionnaire 
    questionnaire = load_questionnaire(questionnaire)
    questionnaire_name = questionnaire['name']
    
    
    dims = dims_dict.get(questionnaire_name, sorted(list(set([q['dimension'] for q in questionnaire['questions'].values()]))))


    if questionnaire_name in ['BFI', '16Personalities']:
        thresh = (questionnaire['range'][0] + questionnaire['range'][1]) / 2
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
    
    return assessment_results

from utils import logger_main as logger
logger.info('Start Testsing')

eval_methods = ['interview_convert', 'interview_convert_api', 'interview_convert_adjoption_anonymous', 'interview_convert_adjoption_anonymous_api',  'interview_assess_batch_anonymous', 'interview_assess_collective_anonymous'] #'choose'

eval_llms = ['gpt-3.5', 'gpt-4', 'gemini']

results_for_table1 = { q: { m: { llm: {} for llm in eval_llms} for m in eval_methods} for q in questionnaires}

for questionnaire in questionnaires: 
    for eval_method in eval_methods: 
        for eval_llm in eval_llms: 
            for repeat_times in [3]: 
                for agent_llm in ['gpt-3.5']: 
                    if eval_method.endswith('api') and not questionnaire == '16Personalities': continue 

                    logger.info(f'Settings: Questionnaire: {questionnaire}, Eval Method: {eval_method}, Repeat Times: {repeat_times}, Agent LLM: {agent_llm}, Eval LLM: {eval_llm}')

                    multitime_full_results = {}
                    
                    #multitime_results = {}

                    # calculate robustness
                    count_robustness = 0
                    sum_robustness = 0 

                    for agent_type in agent_types:
                        for character in characters:
                        #for agent_type in [ a for a in character_info[character]['agent'] if a in agent_types]:
                            if not agent_type in character_info[character]['agent']: continue
                            if character == 'Sheldon-en' and agent_type == 'RoleLLM': continue 

                            multitime_assess_results = load_assessment_results(character, agent_type, agent_llm, 
                                questionnaire, eval_method, eval_llm, repeat_times)
                            
                            multitime_full_results[(character, agent_type)] = multitime_assess_results
                            
                            dims = multitime_assess_results[0].keys()

                            for dim in dims:
                                dim_results = [a_results[dim]['score'] for a_results in multitime_assess_results]

                                from utils import avg, std
                                
                                dim_std = std(dim_results)
                                
                                if questionnaire == '16Personalities':
                                    range_span = 100
                                elif questionnaire == 'BFI':
                                    range_span = 4 

                                robustness = dim_std / range_span

                                
                                count_robustness += 1
                                sum_robustness += robustness

                                #print(f'Character: {character}, Agent Type: {agent_type}, Dimension: {dim}, Robustness: {robustness:.4f}')
                            
                    avg_robustness = sum_robustness / count_robustness

                    logger.info(f'Average Robustness: {avg_robustness:.4f}')                    
                    
                    import itertools
                    for num_exps in [1]: #range(1, repeat_times+1):
                        # assume that we have 1 or 2, ..., or repeat_times, experiments. what is the result?
                        exp_combinations = list(itertools.combinations(range(repeat_times), num_exps))
                        
                        accs = []
                        for exp_combination in exp_combinations:
                            # e.g exp_combination == (1, 2)
                            observed_exp_results = {k: [vv for i, vv in enumerate(v) if i in exp_combination] for k, v in multitime_full_results.items()}
                            assert(len(observed_exp_results) == len(multitime_full_results))
                            assert(len(list(observed_exp_results.values())[0]) == num_exps)

                            observed_final_results = {k: get_final_assessment(v, questionnaire)  for k, v in observed_exp_results.items()}

                            results = {k: v['code'] for k, v in observed_final_results.items()}

                            from personality_tests import accuracy
                            acc = accuracy(agent_types, results, questionnaire)

                            accs.append(acc)
                        
                        logger.info(f'Number of Exps: {num_exps}')
                        from utils import avg, std
                        
                            
                        for agent_type in ['all']: #accs[0].keys():
                            logger.info(f'Agent Type: {agent_type}')
                            for metric in ['dim_acc', 'full_acc']: 
                                values = [acc[agent_type][metric] for acc in accs]
                                
                                avg_metric = avg(values)
                                std_metric = std(values)
                                logger.info(f'{metric}: {avg_metric:.4f} +- {std_metric:.4f}')

                                if agent_type == 'all' and num_exps == 1:
                                    results_for_table1[questionnaire][eval_method][eval_llm][metric] = {'avg': avg_metric, 'std': std_metric}

                            # individual dim accuracy 
                            dims = dims_dict[questionnaire]
                            for dim in dims:
                                values = [acc[agent_type]['each_dim_acc'][dim] for acc in accs]
                                avg_metric = avg(values)
                                std_metric = std(values)
                                logger.info(f'{dim}: {avg_metric:.4f} +- {std_metric:.4f}')



eval_method_map = {
    #'choose': 'Likert-Scale', 
    'interview_convert': '$\\text{OC}$',
    'interview_convert_adjoption_anonymous': '$\\text{OC}_{adj}$',
    'interview_assess_batch_anonymous': '$\\text{EG}_{batch}$',
    'interview_assess_collective_anonymous': '$\\text{EG}$'
}

# reverse eval_method_map
eval_method_map = {v: k for k, v in eval_method_map.items()}

table1_line_template = '''\\multirow{{3}}{{*}}{{ {} }}  & \\chatgpt & {} & {} & {} & {} \\\\
    & \\gptfour & {} & {} & {} & {} \\\\ 
    & \\gemini & {} & {} & {} & {} \\\\
    \\midrule'''


def result_string(acc_dict):
    #return '${:.2f} \pm \scriptstyle{:.2f}$'.format(acc_dict['avg'], acc_dict['std'])
    return '${:.3f}$'.format(acc_dict['avg'])


for eval_med in eval_method_map.keys(): 
    eval_method_bfi = eval_method_map[eval_med]
    # if eval_method_bfi + '_api' in eval_methods:
    #     eval_method_16p = eval_method_bfi + '_api'
    eval_method_16p = eval_method_bfi

    models = ['gpt-3.5', 'gpt-4', 'gemini']

    result_strs = []

    # placeholder
    # results_for_table1['BFI'][eval_method_bfi]['gemini'] = {"dim_acc": {"avg": 0.0, "std": 0.0}, "full_acc": {"avg": 0.0, "std": 0.0}}
    # results_for_table1['16Personalities'][eval_method_16p]['gemini'] = {"dim_acc": {"avg": 0.0, "std": 0.0}, "full_acc": {"avg": 0.0, "std": 0.0}}

    for model in models:
        result_strs += [ result_string(acc_dict) for acc_dict in [
            results_for_table1['BFI'][eval_method_bfi][model]['dim_acc'],
            results_for_table1['BFI'][eval_method_bfi][model]['full_acc'],
            results_for_table1['16Personalities'][eval_method_16p][model]['dim_acc'],
            results_for_table1['16Personalities'][eval_method_16p][model]['full_acc'],
        ]]

    
    table1_line = table1_line_template.format(eval_med, *result_strs)
    print(table1_line)

    print('\n\n')
import pdb; pdb.set_trace()
                            


                            
                            
            #                 results[(character, agent_type)] = result['code']
            #                 #multitime_results[(character, agent_type)] = result['raw']
                    
            #         print('Questionnaire: {}, Eval Method: {}, Repeat Times: {}, Agent LLM: {}, Eval LLM: {}'.format(questionnaire, eval_method, repeat_times, agent_llm, eval_llm))       

            #         print('Agent Type: {}, Dimension Accuracy: {:.4f}, Full Accuracy: {:.4f}'.format(
			# a, dim_acc, full_acc))
                        

                    
                    
                        
                        
                            
            
                
            
            

    
        
    
