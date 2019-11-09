## Micromobility Text Analysis

**Project description:** This is a simple webscraper that builds a word cloud and makes bigrams & trigrams. All code can be found in jupyter notebook file. Articles have been compiled throughout the summer of 2019. This analysis uses a subset of these articles as a quick attempt to find useful information. All articles can be found in the xlsx file or in the jupyter notebook.



### 1. Data Collection


```python

# import the necessary libraries
from bs4 import BeautifulSoup
import requests
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

```

```python

df = pd.read_excel("Micromobility news links.xlsx") #add our micromobility news articles into a dataframe

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

```javascript

from bs4 import BeautifulSoup
import requests
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

```

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
