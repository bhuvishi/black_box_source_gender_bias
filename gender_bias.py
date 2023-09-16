import re
from nltk.tokenize import word_tokenize


def get_pronouns():
    all_pronouns = set()
    for pronouns in ALL_CATCH_WORD_TO_PRONOUNS.values():
        for pronoun in pronouns:
            all_pronouns.add(pronoun)
    return all_pronouns


ALL_CATCH_WORD_TO_PRONOUNS = {
    'he/she': ('he', 'she'),
    'him/her': ('him', 'her'),
    'they': ('he', 'she'),
    'he': ('he', 'she'),
    'she': ('he', 'she'),
    'their': ('his', 'her'),

    'he\'s': ('he\'s', 'she\'s'),
    'she\'s': ('he\'s', 'she\'s'),
    'they\'r': ('he\'s', 'she\'s'),
    'them': ('him', 'her'),
    'him': ('him', 'her'),
    'his': ('him', 'her'),
    'her': ('him', 'her'),

    'her\'s': ('his', 'her\'s'),
    'his/her': ('his', 'her')
}

CATCH_WORD_TO_PRONOUNS = {
    'he/she': ('he', 'she'),
    # 'him/her': ('him', 'her'),
    'they': ('he', 'she'),
    'he': ('he', 'she'),
    'she': ('he', 'she'),
    # 'their': ('his', 'her'),

    # 'he\'s': ('he\'s', 'she\'s'),
    # 'she\'s': ('he\'s', 'she\'s'),
    # 'they\'r': ('he\'s', 'she\'s'),
    #'them': ('him', 'her'),
    #'him': ('him', 'her'),
    #'his': ('him', 'her'),
    #'her': ('him', 'her'),

    #'her\'s': ('his', 'her\'s'),
    #'his/her': ('his', 'her')
}

ALL_PRONOUNS = get_pronouns()


def _get_male_and_female_tokens():
    male_tokens = set()
    female_tokens = set()
    for male_token, female_token in ALL_CATCH_WORD_TO_PRONOUNS.values():
       male_tokens.add(male_token)
       female_tokens.add(female_token)
    return male_tokens, female_tokens

_MALE_TOKENS, _FEMALE_TOKENS = _get_male_and_female_tokens()

def is_male(token):
    return token.lower() in _MALE_TOKENS

def is_female(token):
    return token.lower() in _FEMALE_TOKENS
 

def get_masked_str_and_targets(text):
    text_words = list(word_tokenize(text))  # splits the string into a list
    pronouns_free_text_words = list(text_words)
    for i in range(len(pronouns_free_text_words)):
        if pronouns_free_text_words[i].lower() in ALL_PRONOUNS:
            pronouns_free_text_words[i] = ''
    # print(pronouns_free_text_words)

    for i in range(len(text_words)):
        pronouns = CATCH_WORD_TO_PRONOUNS.get(text_words[i].lower(), None)
        if pronouns:
            masked_str = ' '.join(  # joins all the elemnts into one string
                pronouns_free_text_words[:i] + ['[MASK]'] + pronouns_free_text_words[i + 1:])
            masked_str = masked_str.replace('  ', ' ')
            yield (masked_str, pronouns)


def remove_words(text, token):
    text = text.replace(token, " ")
    return text