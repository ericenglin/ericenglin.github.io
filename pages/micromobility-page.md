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
bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)

#trigrams
trigram_freq = trigramFinder.ngram_fd.items()
trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram','freq']).sort_values(by='freq', ascending=False)
```


```python

#function to filter for ADJ/NN bigrams
def rightTypes(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in stopwords or word.isspace():
            return False
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
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
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[2][1] in third_type:
        return True
    else:
        return False
#filter trigrams
filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]
```

| Priority apples | Second priority | Third priority |
|-------|--------|---------|
| ambrosia | gala | red delicious |
| pink lady | jazz | macintosh |
| honeycrisp | granny smith | fuji |


```html
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trigram</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3614</th>
      <td>(camden,, nj, —)</td>
      <td>14</td>
    </tr>
    <tr>
      <th>14748</th>
      <td>(metro, red, line)</td>
      <td>7</td>
    </tr>
    <tr>
      <th>21801</th>
      <td>(pbl, installation, table)</td>
      <td>6</td>
    </tr>
    <tr>
      <th>14533</th>
      <td>(appropriate, software, installed,)</td>
      <td>6</td>
    </tr>
    <tr>
      <th>14429</th>
      <td>(lean, library, here.)</td>
      <td>6</td>
    </tr>
    <tr>
      <th>14538</th>
      <td>(download, article, citation)</td>
      <td>6</td>
    </tr>
    <tr>
      <th>14548</th>
      <td>(choice., simply, select)</td>
      <td>6</td>
    </tr>
    <tr>
      <th>15023</th>
      <td>(masoud, et, al.)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>21546</th>
      <td>(=, average, annual)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4194</th>
      <td>(vincent, deblasio, camden,)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4195</th>
      <td>(deblasio, camden,, nj)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>21150</th>
      <td>(study, sample, size)</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19716</th>
      <td>(split, phase, intersections)</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19688</th>
      <td>(higher, turn, volume)</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19659</th>
      <td>(crashes, per, bicyclist)</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7905</th>
      <td>(international, municipal, lawyers)</td>
      <td>4</td>
    </tr>
    <tr>
      <th>21976</th>
      <td>(split, phase, locations)</td>
      <td>4</td>
    </tr>
    <tr>
      <th>21161</th>
      <td>(bicyclist, injury, crashes)</td>
      <td>4</td>
    </tr>
    <tr>
      <th>15028</th>
      <td>(transit, feeder, system)</td>
      <td>4</td>
    </tr>
    <tr>
      <th>14514</th>
      <td>(can:, need, help?)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14516</th>
      <td>(help?, contact, sage)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14596</th>
      <td>(content:, sharing, links)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9135</th>
      <td>(energy,, green, building,)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9136</th>
      <td>(green, building,, transportation,)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9137</th>
      <td>(building,, transportation,, waste)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9138</th>
      <td>(transportation,, waste, solutions,)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9139</th>
      <td>(waste, solutions,, connectivity,)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14527</th>
      <td>(email, alerts, contents)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9140</th>
      <td>(solutions,, connectivity,, policy)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14566</th>
      <td>(article, via, social)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14609</th>
      <td>(sage, journals, article)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14611</th>
      <td>(article, sharing, page.)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>20390</th>
      <td>(current-generation, mixing, zone)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5852</th>
      <td>(traffic, safety, administration)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5851</th>
      <td>(highway, traffic, safety)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14631</th>
      <td>(journals, sharing, page.)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14654</th>
      <td>(conditions, view, permissions)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19530</th>
      <td>(draft, manuscript, preparation:)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19501</th>
      <td>(authors, confirm, contribution)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4214</th>
      <td>(tapinto, camden, staff)</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
```

### 3. Support the selection of appropriate statistical tools and techniques


### 4. Provide a basis for further data collection through surveys or experiments


For more details see [Full Github Repo](https://github.com/ericenglin/Micromobility-Text-Analysis).
