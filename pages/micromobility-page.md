## Micromobility Text Analysis

**Overall Context** Micromobility is a trendy catch-all term for small modes of transportation that are coming into streets around the world. As a general definition, micromobility is a category of modes of transport that are provided by very light vehicles such as electric scooters, electric skateboards, shared bicycles and electric pedal assisted, pedelec, bicycles. The primary condition for inclusion in the category is a gross vehicle weight of less than 500 kg. This is still a developing field with new trends every day (see: [pogo sticks](https://www.autonews.com/mobility-report/pogo-sticks-join-micromobility-field)).



**Project description:** This is a simple webscraper that builds a word cloud and makes bigrams & trigrams. All code can be found in jupyter notebook file [here](https://github.com/ericenglin/Micromobility-Text-Analysis/blob/master/Micromobility%20Text%20Analysis.ipynb).

As an initial step, articles have been compiled throughout the summer of 2019. This analysis uses a subset of these articles as a quick attempt to find useful information. All articles can be found in the xlsx file or in the jupyter notebook.



### 1. Data Collection

This analysis requires a few packages to work correctly. Pandas and Matplotlib are broadly applicable to most data science projects and are also needed in this case. BeautifulSoup and Requests are needed to scrape the content of the different article sites.

This snippet of code also uses Pandas to upload our Excel. I've been tracking micromobility news articles in a separate excel document that has the name and link for about 50 articles. These articles are fairly broad but all refer to something that I found interesting in the micromobility field.

```python

# import the necessary libraries
from bs4 import BeautifulSoup
import requests
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

#add our micromobility news articles into a dataframe
df = pd.read_excel("Micromobility news links.xlsx")

```

Once we have our packages downloaded, we'll need a script to scrape their content. This is where Requests and BeautifulSoup come in. We have about 50 links that we want to go through, so we'll build a loop to go through each link in our Excel file (now called df). We'll use requests to pull each page and BeautifulSoup to parse the page content.

Once we have this content, we'll want to specifically grab the words in each paragraph. In HTML webpages, this section is usually marked as a "p". We tell BeautifulSoup that we want to find all of these (rather than find the first one), and then we turn it into read-able words by stripping the HTML filler and getting the actual text.

Once we have our words as a big group of text, we want to create tokens, or individual words in a list, because we want to make everything lower case. From here, we take the tokens and put them right back into a big string of words called "comment words".

```python


comment_words = ' '
token_list = []

for x in df['Link']: #loop through each link
    link = x
    my_page = requests.get(link)
    soup = BeautifulSoup(my_page.content, 'html.parser')
    for val in soup.find_all("p"):

        # typecaste each val to string
        val = str(val.get_text().strip())

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        for words in tokens:
            comment_words = comment_words + words + " "
            token_list.append(words)

```


### 2. Analysis

#### a. Word Cloud

Generally, the easiest first step is to create a WordCloud once you have a big group of text. To do this, we want to remove our stopwords -- these are words like "and", "also","the","a","of", etc. These words essentially provide no information for our analysis and should be removed. After this, there are other words that I think will show up a lot in the articles that I don't think are useful. It really isn't helpful if I get a big word cloud that has "Scooters" and "Cities" as my biggest words. I want words that are unique to these articles and a bit unexpected.

```python

stopwords = set(STOPWORDS)

#want to limit the words that can be in WordCloud
#So adding in more stop words
for x in ['div','national', 'cities','data',
          'transportation','scooter','one',
         'people','scooters','city','car','app',
         'rider','will','mobility','place','say','new',
         'information','way','vehicle','bicycle','street',
         'time', 'cycling', 'cyclist','bike',
         'said','cyclists']:
    stopwords.add(x)

#create WordCloud
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10,
                     collocations = True).generate(comment_words)

# plot the WordCloud image                        
plt.figure(figsize = (16, 16), facecolor = None)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()
```

This is the immage that is generated:
<img src="./../images/micromobility-word-cloud.png?raw=true"/>



#### b. N-bigrams

Word clouds offer some high-level information on the text, but they are more helpful for a very high-level first view. Another first step that is a bit more detailed is to create n-grams. These are words that show up next to each other. Two words together are bigrams, three words are trigrams, etc. The bigger the n-gram, generally the more processing power is necessary.

```python
import nltk
bigrams = nltk.collocations.BigramAssocMeasures()
trigrams = nltk.collocations.TrigramAssocMeasures()


bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(token_list)
trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(token_list)

#bigrams
bigram_freq = bigramFinder.ngram_fd.items()
bigramFreqTable = pd.DataFrame(list(bigram_freq),
  columns=['bigram','freq']).sort_values(by='freq',
                                ascending=False)

#trigrams
trigram_freq = trigramFinder.ngram_fd.items()
trigramFreqTable = pd.DataFrame(list(trigram_freq),
        columns=['trigram','freq']).sort_values(by='freq',
                                        ascending=False)
```

Once we've created these functions, we can call on them to create our ngrams and make some tables showing the frequency that each n-gram appears in all of our articles.


```python

#function to filter for ADJ/NN bigrams
def rightTypes(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in stopwords or word.isspace():
            return False
    acceptable_types = ('JJ', 'JJR',
                        'JJS', 'NN',
                        'NNS', 'NNP',
                        'NNPS')

    second_type = ('NN', 'NNS',
                  'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False

#filter bigrams
filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]

#function to filter for trigrams
def rightTypesTri(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in stopwords or word.isspace():
            return False
    first_type = ('JJ', 'JJR',
                  'JJS', 'NN',
                  'NNS', 'NNP',
                  'NNPS')

    third_type = ('JJ', 'JJR',
                  'JJS', 'NN',
                  'NNS', 'NNP',
                  'NNPS')

    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[2][1] in third_type:
        return True
    else:
        return False
#filter trigrams
filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]
```

Trigram Table:

| Trigram | # of Times Seen |
|-------|--------|---------|
| split, phase | 30 |
| crash rates | 21 |
| camden, nj | 14 |
| crash rate | 13 |
| red line | 13 |
| pbl installation | 12 |
| comfort, safety | 11 |
| transit feeder | 10 |
| linear regression | 9 |
| los angeles | 8 |
| auto lanes | 8 |
| higher turn | 7 |
| main effects | 7 |
| sample size | 7 |
| pogo sticks | 7 |
| united states | 7 |
| masoud, et | 6 |
| electric bikes | 6 |


Bigram Table:

| Bigram | # of Times Seen |
|-------|--------|---------|
| camden,, nj, — | 14 |
| metro, red, line | 7  |
| pbl, installation, table | 6  |
| appropriate, software, installed | 6  |
| lean, library, here | 6  |
| download, article, citation | 6  |
| masoud, et, al. | 5  |
| study, sample, size | 4  |
| split, phase, intersections | 4  |
| higher, turn, volume | 4  |
| crashes, per, bicyclist | 4  |
| international, municipal, lawyers | 4 |
| bicyclist, injury, crashes | 4 |
| split, phase, locations | 4 |
| energy,, green, building | 3 |
| building,, transportation,, waste | 3 |




### 3. Collect more data

#### A. Google News API

The Bigram and Trigrams give a little bit more information, but this definitely isn't sufficient to have an understanding of micromobility news happening in any given quarter. Now it is clear that we'll need collect more data. We'll do this by calling on the Google News API and trying to grab as many articles as we can -- the Google News API only allows information for the last 30 days, so we will want to grab as many articles as possible in this timeframe.


```python

from newsapi.newsapi_client import NewsApiClient
# Init
newsapi = NewsApiClient(api_key=api_key['App Key'].iloc[0])

from datetime import datetime, timedelta
from_time = datetime.strftime(datetime.now() - timedelta(31), '%Y-%m-%d')
to_time = datetime.strftime(datetime.now(), '%Y-%m-%d')

# Free plan only allows up to 100 articles taken at one time
from datetime import datetime, timedelta

title = []
url = []
author = []
publish_date = []
content = []

for num in range(1,30):

    from_time = datetime.strftime(datetime.now() - timedelta(num+1),
                                                  '%Y-%m-%d')
    to_time = datetime.strftime(datetime.now() - timedelta(num),
                                                  '%Y-%m-%d')

    for page_num in range(1,5):
        top_headlines = newsapi.get_everything(q='scooter',
                                           from_param=from_time,
                                          to=to_time,
                                           page=page_num,
                                              language='en')       
    for x in top_headlines['articles']:
        title.append(x['title'])
        url.append(x['url'])
        author.append(x['source']['id'])
        publish_date.append(x['publishedAt'])
        content.append(x['content'])

#make dataframe
df = pd.DataFrame({
    'Title':title,
    'URL':url,
    'Author':author,
    'Date Published':publish_date,
    'Content':content})

```

### 4. Next Steps: Machine Learning Classification

This Google News API allowed us to get 497 articles for the month of October. We will want to recreate our scraper from above and grab our contents again. However, rather than jumping into a WordCloud and N-gram tokens, we'll want to go a step further. For a subsection of these articles, I want to create classifications of the articles based on content area. From here, we can use these to create more precise NLP analyses that will allow us to look at new text being used in each specific area. For example, we'll want to know what terms are newly being used in safety articles compared to technology articles.

More to come on this front!
