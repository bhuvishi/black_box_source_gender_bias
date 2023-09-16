import urllib.request
import bs4 as bs
import re


def get_text_from_url(url_str):
    source = urllib.request.urlopen(url_str).read()
    soup = bs.BeautifulSoup(source, features="html.parser")

    paragraph_texts = []
    for paragraph in soup.find_all('p'):
        paragraph_text = paragraph.text
        paragraph_text = re.sub(r'\[[0-9]*\]', ' ', paragraph_text)
        paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
        paragraph_text = paragraph_text.lower()
        paragraph_text = re.sub(r'\d', ' ', paragraph_text)
        paragraph_texts.append(paragraph_text)

    text = ' '.join(paragraph_texts)
    return text


def create_text_chunks(input_text, chunk_size=200):
    curr_index = 0
    input_words = input_text.split(' ')
    num_words = len(input_words)
    while curr_index < num_words:
        chunk = ' '.join(input_words[curr_index: curr_index + chunk_size])
        yield chunk
        curr_index += chunk_size


def get_text_chunks_from_url(url, chunk_size=200):
    text = get_text_from_url(url)
    chunks = create_text_chunks(text, chunk_size)
    return chunks


