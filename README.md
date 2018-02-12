# BFE-Investor_Selector

A selector that select best match investors for a startup

Dear members of BFE Data Analytics Team, Good Evening~

The files in this repo is a prototype for our investor selector, implemented in Python with Jupyter Notebook.

There are three files in this repo, including 1. search-engine-on-crawled-data 2. search-engine-on-manual-data 3. try-beautiful-soup

And also a directory called "Py version"

# "Py version"
There are two .py files in this file, including Search.py and crawler.py

The Search.py transplanted the Searcher class in the jupyter notebook into .py files

The crawler.py transplanted the crawler related functions in the jupyther notebokk into .py file

Both of then are clearly documented, enjoy~

# File 1:

## Searcher initialization
this file is the final product, which has all the functionalities implemented, all included in class Searcher. 
When this class is initiated, it will first try to load the word2vec model, which will be helpful in the latter stage of similarity matching. 
Then it will read the database from a csv file, and start to crawl data from the urls following each investor company, and then replace the url
with all the crawled data. When all the text data is prepared, it will further build a tfidf model on all the text data (including manual 
investor summary), which is helpful in calculating the sentence vector in the latter similarity matching stage.

## class variables of Searcher
In Searcher, you can find the following __class variables__:
1. Searcher._database: a pandas dataform that store all the relevent data for the investors including name, website url, crawled data, manual
summary etc.
2. Searcher._tfidf, ._dictionary: these are two gensim objects, which can give a priority value for each word you give it, according to the 
text data we have fed it, this priority value is latter used as a weight for each word.
3. Searcher._w2v: this is also a gensim object, which is, in this case, a fasttext word model pretrained from wiki_news. This model can give
a 300d word vector for the word you feed it, and it has the ability to infer word vectors for unseen words. This word vector can represent
the semantic similarity of each word, and is a great way of quantifying text data.

After the instanciation of class Searcher, you can call Searcher.updata_similarity() to sort the database of investors according to a startup
description you input. Then you can print the Searcher._database to check the best few matched investor for the startup.

## automatic crawling
Regarding the __crawling method__, in Searcher, it will crawl all the text data on the website of the input url, and it will do the same thing to
all the urls found on the website. It also has some simple function of excluding irrelevent ursl, automatically filling incomplete urls, 
automatically adjusting illegal urls, etc. Overall, the crawling is achieved with BS4(beautiful soup)

## similarity matching
Regarding the algorithm used to find the __similarity__ of between texts, in Searcher, sentence vector is used. Normally, to compare the similarity
of two word, all that you need to do is to calculate the cosine similarity of the two words' word vector. If we can get a sentence vector for
each text, then we can do exactly the same thing. To get the sentence vector, Searcher used the following equation: 
      
      Sentence vector = Sum of (tfidf priority value of the word * word vector of the word) over all the words of the sentence.

We can calculate the sentence vector for text of arbitrary length, and finally, we can find the similarity by calculate the cosine similarity
of two sentence vectors.
It might seems simplistic, but it turned out to be enough in this case

# File 2
This is the experiment file for the similarity function, there is no class in it, but there are several results of similarity searching displayed.

# File 3
This is the experiment file for the crawling function.


