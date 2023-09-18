# importing libraries
from transformers import pipeline
import wiki_scraper
import gender_bias
import numpy as np
import dataset

# defining a function to obtain results. arguments passed are url and unmasker


def get_results_for_url(url, unmasker):
    '''defining a variable to get chunks of text from the urls provided 
    and wikiscraper is used to extract data from the wikipedia sites'''
    text_chunks = wiki_scraper.get_text_chunks_from_url(url)
    for text_chunk in text_chunks:
        masked_str_and_targets = gender_bias.get_masked_str_and_targets(
            text_chunk)
        for masked_str, targets in masked_str_and_targets:
            results = unmasker(masked_str, targets=targets)
            # making sure that the length of results is 2
            assert (len(results) == 2)
            yield results  # returns the results


def get_statistics_from_results(results):
    assert (len(results) == 2)
    male_score = 0
    female_score = 0
    for result in results:
        if gender_bias.is_female(result['token_str']):
            female_score = result['score']
        elif gender_bias.is_male(result['token_str']):
            male_score = result['score']
    ratio = male_score / female_score
    #if ratio > 1000:
    #    print("the ratio is very odd for ",  results)
    return ratio


def collect_statistics_from_results(results, output_score_statistics):
    output_score_statistics.append(get_statistics_from_results(results))


def get_statistics_for_urls(urls, unmasker):
    score_statistics = []
    for url in urls:
        results = get_results_for_url(url, unmasker)
        for result in results:
            collect_statistics_from_results(result, score_statistics)
    return score_statistics


urls_male_dominated = [
    'https://en.m.wikipedia.org/wiki/Engineer',
    'https://en.wikipedia.org/wiki/Chief_executive_officer',
    'https://en.wikipedia.org/wiki/Police_officer',
    'https://en.wikipedia.org/wiki/Technician',
    'https://en.wikipedia.org/wiki/Aircraft_pilot',
    'https://en.wikipedia.org/wiki/Author',
    'https://en.wikipedia.org/wiki/Businessperson']


urls_female_domninated_one = [
    'https://en.wikipedia.org/wiki/Nursing']

urls_female_domninated = [
    'https://en.wikipedia.org/wiki/Nursing',
    'https://en.wikipedia.org/wiki/Teacher',
    'https://en.wikipedia.org/wiki/Housekeeping']




def get_modified_text_chunk(input_text):
    input_words = input_text.split(' ')
    visited = set()
    unique_words = dataset.get_unique_words(input_text)
    for word in unique_words:
        if word not in visited:
            visited.add(word)
            new_text = gender_bias.remove_words(input_text, word)
            yield (new_text, word)

def get_results_for_url_post_removing(url, unmasker):
    text_chunks = wiki_scraper.get_text_chunks_from_url(url)
    #text_chunks = ["this is geetika, she is a girl", "this is neetika, she is a girl too"]
    visited = set()
    results_url = {}
    for text_chunk in text_chunks:
        new_text_and_words = get_modified_text_chunk(text_chunk)
        for new_text, word in new_text_and_words:
            if new_text not in visited:
                new_text = new_text.lower()
                visited.add(word)
                masked_str_and_targets = gender_bias.get_masked_str_and_targets(
                    new_text)
                result_word = []
                for masked_str, targets in masked_str_and_targets:
                    result_from_masking = unmasker(masked_str, targets=targets)
                    # making sure that the length of results is 2
                    assert (len(result_from_masking) == 2)
                    collect_statistics_from_results(result_from_masking, result_word)
            if len(result_word) != 0:
                results_url[word] = results_url.get(word, [])  + result_word
    return results_url

def get_statistics_for_url_post_removing(urls, unmasker):
    score_statistics = {}
    for url in urls:
        results = get_results_for_url_post_removing(url, unmasker)
        for word in results.keys():
            score_statistics = results[word]
            if len(score_statistics) >  5:
                count  = len(score_statistics)
                mean = np.mean(score_statistics)
                median = np.median(score_statistics)
                results[word] = {}
                results[word]["count"] = count
                results[word]["mean"] = mean
                results[word]["median"] = median
        
        count = 0
        for word in results.keys():
            print (word, results[word]) 
            count += 1
            if count == 20:
                break


def main():
    unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
    get_statistics_for_url_post_removing(urls_female_domninated_one, unmasker)

    '''
    score_statistics_male = get_statistics_for_urls(urls_male_dominated, unmasker)
    print("stats for male dominated profession")
    print( 
        'count={0}, mean={1}, median={2}'.format(
            len(score_statistics_male),
            np.mean(score_statistics_male),
            np.median(score_statistics_male)))

    #score_statistics_female = get_statistics_for_urls(urls_female_domninated, unmasker)


    #print("stats for female dominated profession")
    #print( 
        'count={0}, mean={1}, median={2}'.format(
            len(score_statistics_female),
            np.mean(score_statistics_female),
            np.median(score_statistics_female)))
    '''


if __name__ == "__main__":
    main()