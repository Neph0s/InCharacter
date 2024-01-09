from personality_tests import personality_assessment

characters = ['hutao-zh']

for character in characters:
    # direct
    personality_assessment(
        character, 'ChatHaruhi', 'gpt-4', 
        '16Personalities', 'direct')
    
    personality_assessment(
        character, 'ChatHaruhi', 'gpt-4', 
        '16Personalities', 'interview_assess_collective', 'gpt-4', repeat_times=0.25)
    
    personality_assessment(
        character, 'ChatHaruhi', 'gpt-4', 
        '16Personalities', 'close-ended', 'gpt-4')

    personality_assessment(
        character, 'ChatHaruhi', 'gpt-4', 
        '16Personalities', 'interview_convert_api')

    personality_assessment(
        character, 'ChatHaruhi', 'gpt-4', 
        '16Personalities', 'interview_convert', 'gpt-4')
    
    personality_assessment(
        character, 'ChatHaruhi', 'gpt-4', 
        '16Personalities', 'interview_assess_batch', 'gpt-4')

    personality_assessment(
        character, 'ChatHaruhi', 'gpt-4', 
        '16Personalities', 'interview_assess_collective', 'gpt-4')