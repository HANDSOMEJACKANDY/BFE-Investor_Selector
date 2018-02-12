from crawler import *

import multiprocessing
import numpy as np
import pandas as pd
from gensim import models
from gensim import corpora
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import sent_tokenize, word_tokenize
from stop_words import get_stop_words
import re


class Searcher():
    """This Searcher class clustered all the similarity matching functions needed for the investor selector
    Variables:
        _w2v: the word2vector model that can convert each word to a fixed length vector
        _database: a pandas dataframe that stored the info of each companies, including manually summarized info and crawled info
                   The last col of _database is used to store the similarity to the input description
    """
    def __init__(self, w2v=None):
        """This init usually tasks a while, as it might have to load a very large w2v model, as well as crawling data for a large dataset
        input:
            w2v: either a string as the file directory of the interested w2v model, or an already loaded w2v model in the form of KeyedVectors

        process:
            first a w2v model will be loaded according to the input of w2v
            then, it will either load a database that is already filled with crawled data, or start to crawl a untouched new dataset"""

        # load w2v model
        if type(w2v) is str:
            print("start loading w2v, this might take a while")
            self._w2v = KeyedVectors.load_word2vec_format(w2v)
        else:
            self._w2v = w2v

        # get and process database
        try:
            self._database = pd.read_csv('crawled_database.csv').iloc[:, [1, 2, 3]]
            print("load crawled database successful")
        except:
            # if no crawled database given
            # load the dataset : including only each company's name, url and summary
            print('fail to load crawled database')
            self._database = pd.read_csv("../input/InvestData_2017-Nov-22_0101.csv").iloc[:, [1, 5, 6]]
            self.crawl_database()
        self.process_database()

    def process_database(self):
        """Preprocessing a already crawled database
        firstly, all the Nan data will be filtered
        then, tokenize all the relevent texts, i.e. manually summarized texts and crawled texts in the dataset
        """
        # 1: company name, 5: company website, 6: company manual desc
        raw_texts = []
        # preprocess all the text data and remove any row without any useful data, and segment each word
        drop_list = []
        for row in self._database.itertuples():
            if not type(row[1]) is str or (not type(row[2]) is str and not type(row[3]) is str):
                # check if the row has data
                drop_list.append(row[0])
            else:
                # process text data of both manually summarized or crawled data
                tmp_text = []
                for col in [2, 3]:
                    real_col = col - 1
                    text = row[col]
                    if type(text) is str:
                        text = self.word_tokenize_string(text)
                        self._database.iloc[row[0], real_col] = text
                        tmp_text.append(text)
                # merge texts of same company
                tmp_text = '    '.join(tmp_text)
                raw_texts.append(tmp_text)

        # drop all the rows that do not have essential data
        self._database.drop(drop_list, inplace=True)
        # create similarity col for similarity search use
        self._database = self._database.assign(similarity=np.zeros(len(self._database)))

        # use the raw_texts to generate tfidf model
        self._tfidf, self._dictionary = self.get_tfidf_and_dictionary(raw_texts)

    def crawl_database(self):
        """crawl a database with all the target url for each company provided"""
        for row in self._database.itertuples():
            if not (not type(row[1]) is str or (not type(row[2]) is str and not type(row[3]) is str)):
                # process each website and replace web address with texts crawled
                url = row[2]
                texts = self.get_text_from_url_and_its_children(url)
                if not texts:
                    # if cannot access url, replace url with Nan
                    self._database.iloc[row[0], 1] = np.nan
                else:
                    # replace the url with the crawled texts
                    texts = '   '.join(texts)
                    self._database.iloc[row[0], 1] = texts

    def save_database(self):
        """save database to current directory"""
        self._database.to_csv('crawled_database.csv')
        print("database save successful")

    def update_similarity(self, input_text, col=2):
        """use document vector to calculate the cosine similarity of the text data of each company compared to the input_text
        input:
            input_text: the target startup description
            col: choose to compare similarity against manually summarized info or crawled info of each company"""
        # get input text vector
        input_text_vector = self.get_doc_vector(input_text)
        i = 0
        for row in self._database.itertuples():
            row_text_vector = self.get_doc_vector(row[col])
            similarity = input_text_vector.dot(row_text_vector)
            self._database.iloc[i, -1] = similarity
            i += 1
        self._database = self._database.sort_values(by='similarity', ascending=False)
        return self._database

    def get_doc_vector(self, text):
        """For the given text input, this function can generate a corresponding document vector
        input:
            text: a string
        algorithm details:
            This function output a simple tfidf weighted sum of each word vector in the given text"""
        if not text == text:
            return self._w2v['happy'] * 0
        tokens = list(self._dictionary.token2id)
        # convert any unknown word to known word
        new_text = []
        for word in text.split():
            if word in tokens:
                new_text.append(word)
            elif word in self._w2v:  # replace the unknow word with the most similar word in tokens of dictionary
                new_text.append(self._w2v.most_similar_to_given(word_list=tokens, w1=word))

        # start to calculate vector using tfidf weighted word vector sum
        # get tfidf weight
        tokenized_text = [self._dictionary.doc2bow(new_text)]
        tfidf_text = self._tfidf[tokenized_text][0]
        # sum weighted word vectors
        sum_vector = self._w2v['happy'] * 0  # get the size of the word vector
        for word_id, weight in tfidf_text:
            word = self._dictionary[word_id]
            sum_vector += self._w2v[word] * weight
        if sum_vector.any():
            sum_vector /= np.sqrt(sum_vector.dot(sum_vector))  # normalize the vector

        return sum_vector

    def word_tokenize_string(self, text):
        """This fucntion essentially does the preprocessing of string, it will remove stop_words and unknown word in _w2v,
        and then convert all letters to lower letters
        input:
            text: a string"""
        stop_words = get_stop_words('en')  # get too frequent word
        text = text.replace('\r', ' ').replace('\n', ' ')  # remove symbols
        text = re.sub(r"http\S+", "", text)  # remove urls
        # remove any word that present too frequently or cannot be converted to word vector
        text = [word for sent in sent_tokenize(text.lower()) for word in word_tokenize(sent) \
                if not word in stop_words and word in self._w2v]
        return ' '.join(text)

    @staticmethod
    def get_tfidf_and_dictionary(texts):
        """generate dictionary and tfidf model for a given batch of texts
        input:
            texts: a list of string"""

        # get dictionary of texts
        texts = [text.split() for text in texts]
        dictionary = corpora.Dictionary(texts)

        # get tfidf ranking model
        tokenized_texts = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(tokenized_texts)

        return tfidf, dictionary

    def get_text_from_url_and_its_children(self, main_url):
        """This function crawls the given url, and all the url on the webpage of the given url, and collect all the useful text data
        input:
            main_url: the target url
        algorithm details:
            multiprocessing is used to simutaneously craw all the children urls on the main url webpage"""
        print("starting to crawl main url: ", main_url)
        # check validity of main_url
        resp = url_is_valid(main_url)
        if not resp:
            print("main_url is not valid")
            return False

        print("\nstarting to crawl all its children")
        # grab all urls in this web page
        urls = [main_url]
        urls.extend(get_urls_from_url(main_url))
        urls = list(set(urls))  # remove duplicated urls
        print("\n\nthese are the children links we crawled")
        print(urls, "\n")
        # grab all texts in each urls asynchronously
        # argmumentize urls
        urls = [(url, main_url) for url in urls]
        with multiprocessing.Pool(processes=24) as pool:
            text_data = pool.starmap(get_text_from_url_with_check, urls)
            # try terminating hung jobs
        text_data = [text for text in text_data if len(text_data) > 0]  # remove empty returns
        text_data = [text for text_list in text_data for text in text_list]  # get list elements to str
        return text_data