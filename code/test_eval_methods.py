from personality_tests import personality_assessment

characters = ['wangduoyu-zh', "xiaofeng-zh", "tongxiangyu-zh", "haruhi-zh", 'Snape-en', 'hutao-zh']



    
for repeat_times in [0.25, 1, 2]:
    for eval_method in [#'direct', 
                        'choose', 
                        'interview_convert', 'interview_convert_api'
                        'interview_assess_batch', 'interview_assess_collective']:
        for character in characters:
            personality_assessment(
                character, 'ChatHaruhi', 'gpt-3.5', 
                '16Personalities', eval_method, 'gpt-4', repeat_times=repeat_times)

    
        
    
