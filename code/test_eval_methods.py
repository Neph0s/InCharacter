from personality_tests import personality_assessment

characters = ['haruhi-zh']# ['Snape-en']# ['wangduoyu-zh', "xiaofeng-zh", "tongxiangyu-zh", "haruhi-zh", 'Snape-en', 'hutao-zh']



    
for repeat_times in [1]: #, 0.25, 2]:
    for eval_method in [#'direct', 
       'interview_assess_batch_anonymous', 
                        'interview_convert', 'interview_convert_api', 'choose', 
                        'interview_assess_batch', 'interview_assess_collective', 'interview_assess_batch_anonymous']:
        print(f'EVALUATION METHOD: {eval_method}')
        for character in characters:
            personality_assessment(
                character, 'ChatHaruhi', 'gpt-3.5', 
                '16Personalities', eval_method, 'gpt-3.5', repeat_times=repeat_times)
        
        import pdb; pdb.set_trace()
            

    
        
    
