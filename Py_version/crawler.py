from bs4 import BeautifulSoup
import requests

def get_urls_from_url(main_url):
    """collect all the urls presented on the webpage of the given url
    input:
        main_url: the target url, a string"""
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
    """generate texts from web response
    input:
        resp: the object returned by requests.get(a url)
    output:
        generate string"""
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
    """verify whether the given url is leagal and accessible
    input:
        url: interested url, a string

    output:
        if url invalid, return bool: false
        else: return the resp (to save the effort of accessing the url again)"""
    try:
        resp = requests.get(url, timeout=10)
        assert resp.status_code == 200
        return resp
    except:
        return False

def url_compare(url_target, url_income):
    """compare how many consecutive letters are identical from the beginning of the input two urls
    Notice: http and https are considered the same here
    input:
        url_target, url_income: string, the urls to be compared
    output:
        the number of consecutive letters identical from the beginning"""
    n_same_letter = 0.0
    # delete all http or https
    if url_target[4] == 's':
        url_target = url_target[5:]
    else:
        url_target = url_target[4:]
    if url_income[4] == 's':
        url_income = url_income[5:]
    else:
        url_income = url_income[4:]
    # check similarity
    min_len = min(len(url_target), len(url_income))
    for i in range(min_len-1):
        if url_target[i] == url_income[i]:
            n_same_letter += 1
        else:
            break
    return n_same_letter

def get_text_from_url_with_check(url, main_url):
    """crawl text data from the given url
    input:
        main_url: a string, interested url
    output:
        a list of strings"""
    resp = url_is_valid(url)
    if not resp:
        url = main_url + url
        resp = url_is_valid(url)
        if not resp:
            print("url:", url, "invalid")
            return []
    # double check if the url is visited
    if resp.url != url: # meaning its redirected
        print('the url is redirected, try https\n')
        # try https
        url = url[:4] + 's' + url[4:]
        resp = url_is_valid(url)
        if resp:
            if resp.url == url:
                print('try succeeded')
        else:
            return []
    # check if url is the child or sibling of main_url
    if url_compare(main_url, resp.url) < 10: # to avoid http://www.
        print('\nurl:', resp.url, 'might be irrelevent to', main_url, 'quit visiting\n')
        return []
    text_data = []
    for text in get_texts_from_resp(resp):
        text_data.append(text)
    return text_data
