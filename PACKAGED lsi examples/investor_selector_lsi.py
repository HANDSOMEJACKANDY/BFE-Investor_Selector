from bs4 import BeautifulSoup
import requests
import multiprocessing
import numpy as np
import pandas as pd
from ast import literal_eval
from gensim import corpora, models, similarities
from fuzzywuzzy import fuzz
from nltk.tokenize import sent_tokenize, word_tokenize
from stop_words import get_stop_words
from random import shuffle
import stop_words
from gensim.parsing.porter import PorterStemmer
import time
import re
import pylab as pl
from ipywidgets import FloatProgress
from IPython import display
import matplotlib.pyplot as plt

"""
The following part contains all the crawling methods
These functions are not packaged as a class to enable parallization
"""
def get_urls_from_url(main_url):
    resp = requests.get(main_url)
    soup = BeautifulSoup(resp.content, 'html.parser')
    urls = []
    links = soup.find_all('a')
    for url in links:
        try:
            url = url.attrs['href']
            if len(url) > 5:
                urls.append(url)
        except:
            pass
    return urls

def get_texts_from_resp(resp):
    # parse the web response
    soup = BeautifulSoup(resp.content, 'html.parser')
    # find and filter texts
    print("These are texts under", resp.url)
    texts = soup.find_all('p')
    print("number of items grabed are", len(texts))
    texts = [text for text in texts if len(text.text) > 100]
    print("number of items after filtering", len(texts))
    # output texts
    for text in texts:
        #print(text.text)
        yield text.text

def url_is_valid(url):
    try:
        resp = requests.get(url, timeout=10)
        assert resp.status_code == 200
        return resp
    except:
        return False

def url_compare(url1, url2, thresh=70):
    """
    Based on the similarity between roots of two url, return whether these two url are smiliary or not
    """
    # extract pattern in () http(s)://()/???/???
    url1 = re.sub("(https?://)?", "", url1).split('/')[0]
    url2 = re.sub("(https?://)?", "", url2).split('/')[0]
    
    # find similarity between roots
    root_sim = fuzz.partial_ratio(url1, url2)
    
    if root_sim >= thresh:
        return True
    else:
        print(url1, " and ", url2, " may not be relevent")
        return False

def get_text_from_url_with_check(url, main_url):
    """
    The bottom function that extract text from url
    """
    # avoid url ends with .pdf
    if url.split(".")[-1] == "pdf":
        return []
    
    # check if url is valid
    resp = url_is_valid(url)
    # if the url is not valid, it is possible that it is in the form of 
    if not resp:
        if not "http" in url:
            url = main_url + url
            resp = url_is_valid(url)
            if not resp:
                print("url:", url, "invalid")
                return []
        else:
            print("url:", url, "invalid")
            return []
        
    # double check if the url is actually visited
    if resp.url != url: # meaning its redirected, which means an error happened
        # in many cases, the redirection is due to website has prefix https instead of http
        url = url[:4] + 's' + url[4:]
        resp = url_is_valid(url)
        if resp:
            if resp.url == url:
                print('try succeeded')
        else:
            return []
        
    # check if url is the child or sibling of main_url
    # sometimes, the url is directed to same irrelevent sites such as www.twitter.com etc.
    if not url_compare(main_url, resp.url):
        return []
    
    # get text from url
    text_data = []
    for text in get_texts_from_resp(resp):
        text_data.append(text)
    return text_data

def get_text_from_url_and_its_children(main_url):
    """
    Parallalize the text extraction process from given main url
    """
    # preprocess main url
    # remove space in url
    main_url = main_url.replace(" ", "")
    # force https:// prefix to the main url
    main_url = "https://" + re.sub("(https?://)?", "", main_url)
    # remove last "/" if there is one
    if main_url[-1] == "/":
        main_url = main_url[:-1]
    
    print("starting to crawl main url: ", main_url)
    
    # check validity of main_url
    resp = url_is_valid(main_url)
    if not resp:
        print("main_url: ", main_url, " is not valid")
        return "Main site not accessible"

    # grab all urls in this web page
    urls = [main_url]
    urls.extend(get_urls_from_url(main_url))
    # remove duplicated urls
    urls = list(set(urls)) 
    print("\n\nthese are the children links we crawled")
    print(urls, "\n")
    # grab all texts in each urls asynchronously
    # argmumentize urls
    urls = [(url, main_url) for url in urls]
    with multiprocessing.Pool(processes=24) as pool:
        text_data = pool.starmap(get_text_from_url_with_check, urls) 
    
    # collect output text data
    text_data = [text for text in text_data if len(text_data) > 0] # remove empty returns
    text_data = [text for text_list in text_data for text in text_list] # get list elements to str
    return " ".join(text_data)

"""
The following part is for investor selector lsi part
"""
class TextCleaner():
    """
    A class that cleans up text data
    """
    def __init__(self):
        # load stop words
        self.stop_words = stop_words.get_stop_words("en")

        # prepare stemmer
        self.stemmer = PorterStemmer()
    
    def clean(self, text):
        # remove non-alphanumerical letters
        text = re.sub("[^a-zA-Z]+", " ", text)
    
        # stem words
        text = self.stemmer.stem_sentence(text)
        
        return text
    
class TopicModelTrainer():
    """
    The class for training lsi model
    """
    def __init__(self, database_dir="crawled_database.csv"):
        """
        Initialize
        """
        
        # load database
        self.db = pd.read_csv(database_dir)
        
        # load string cleaner
        self.cl = TextCleaner()
        
    def train(self, model_dir="index_models"):
        """
        Train lsi model, including training of Dictionary, Tfidf and Lsi
        And Save them
        """
        # create index model folder
        try:
            os.mkdir(model_dir)
        except:
            pass
        
        # preprocess training documents
        # exclude all invalid text data
        training_docs = [row for row in self.db.iloc[:, -1] if row not in (np.nan, "Main site not accessible")]
        # clean the text
        training_docs = [self.cl.clean(doc) for doc in training_docs]
        
        # get dictionary
        self.dictionary = corpora.Dictionary([doc.split() for doc in training_docs])
        
        # remove organization specific words
        self.dictionary.filter_extremes(no_below=2, no_above=1)
        self.dictionary.compactify()
        
        # prepare tfidf
        training_bows = [self.dictionary.doc2bow(doc.split()) for doc in training_docs]
        self.tfidf = models.TfidfModel(training_bows)
        
        # prepare lsi model
        self.lsi = models.LsiModel(self.tfidf[training_bows], 300, id2word=self.dictionary)
        
        # save to file
        self.dictionary.save(model_dir + "/" + "dictionary")
        self.tfidf.save(model_dir + "/" + "tfidf")
        self.lsi.save(model_dir + "/" + "lsi")
        
    def index_text(self, text):
        """
        Index a given string use lsi model
        """
        if text not in (np.nan, "Main site not accessible"):
            # clean the text
            text = self.cl.clean(text)
            
            # get lsi index
            lsi_index = self.lsi[self.tfidf[self.dictionary.doc2bow(text.split())]]
            
            # get np.array of index
            lsi_index = np.array(list(zip(*lsi_index))[1])
            
            return lsi_index
            
        else: # discard empty texts
            return np.array([])
    
    def index_database(self, model_dir="index_models"):
        """
        Use topic model to index each investor
        And Save it
        """
        # index the database and append it to a new column
        self.db["Index"] = np.vectorize(self.index_text, otypes=[np.ndarray])(self.db["Crawled"])
        
        # save to file
        self.db.to_csv(model_dir + "/indexed_database.csv")
        
class TopicModelIndexer():
    """
    Class for find similarity of each orgnization to a given startup description
    """
    def __init__(self, folder_dir="index_models"):
        """
        Initialize
        """
        # load indexed database
        self.db = pd.read_csv(folder_dir + "/indexed_database.csv")
        print(self.db.columns)
        self.db["Index"] = self.db["Index"].apply(self.literal_eval)
        
        # load string cleaner
        self.cl = TextCleaner()
        
        # load lsi modules: dictionary, tfidf, lsi
        self.dictionary = corpora.Dictionary.load(folder_dir + "/dictionary")
        self.tfidf = models.TfidfModel.load(folder_dir + "/tfidf")
        self.lsi = models.LsiModel.load(folder_dir + "/lsi")
        
    def literal_eval(self, list_string):
        """
        convert string of list to list
        """
        # remove square bracket
        list_string = re.sub("[\[\]]", ' ', list_string)
        
        # convert str to list
        out = [np.float32(number) for number in list_string.split()]
        
        return np.array(out)

    def index_text(self, text):
        """
        Index a given string use lsi model
        """
        # clean the text
        text = self.cl.clean(text)

        # get lsi index
        lsi_index = self.lsi[self.tfidf[self.dictionary.doc2bow(text.split())]]

        # get np.array of index
        lsi_index = np.array(list(zip(*lsi_index))[1])

        return lsi_index
    
    def cos_sim(self, vector, text):
        """
        Dot to find cos similarity
        """
        if len(vector) == len(self.index_text(text)):
            return np.dot(vector, self.index_text(text))
        else:
            return 0
    
    def index_database(self, text, db_dir="indexed_database.csv"):
        """
        Use topic model to index each investor
        And Save it
        """
        # index the database and append it to a new column
        self.db["Similarity"] = np.vectorize(self.cos_sim)(self.db["Index"], text)
        
        # sort the database by its similarity value
        self.db.sort_values(by=["Similarity"], ascending=False)