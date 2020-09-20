# **Internet Movie Database (IMDB): NLP**

**Fastai Library or API**
- [Fast.ai](https://www.fast.ai/about/) is the first Deep learning library to provide a single consistent interface to all the most commonly used Deep learning applications for Vision, Text, Tabular Data, Time series, and Collaborative filtering. [Fast.ai](https://www.fast.ai/about/) is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains, and provides researchers with low-level components that can be mixed and matched to build new approaches.
- Setting up Fastai Environment in Google Colab.

```javascript
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```
**Downloading Libraries and Dependencies**

```javascript
from fastbook import *                                        
from fastai.text.all import *
from IPython.display import display                           
from IPython.display import HTML
```

**Getting the Data**
- Fastai has a number of [Dataset](https://course.fast.ai/datasets) which makes easy to download and to use. I will be using the [IMDB Dataset](https://course.fast.ai/datasets) for this Project available in Fastai. I am using Google Colab for this Project so the process of reading Data might be different in different platforms.

```javascript
path = untar_data(URLs.IMDB)
```

**Word Tokenization**
- I will use Fastai Tokenizer for the process of Word Tokenization. Then, I will use Fastai coll_repr function to display the results. It displays the first n items of the collection. The collections of text documents should be wrap into list. The tokens starting with xx are the special tokens which is not a common word prefix in English.

**Subword Tokenization**
- In Chinese and Japanese languages there are no spaces in the sentences. Similarly Turkish Languages add many subwords together without spaces creating very long words. In such problems the Subword Tokenization plays the key role.

**Preparing the Text Classifier Model**

```javascript
get_imdb = partial(get_text_files, folders=["train", "test", "unsup"])
dls_lm = DataBlock(
    blocks = TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=129, seq_len=80)
dls_clas = DataBlock(
    blocks = (TextBlock.from_folder(path, vocab=dls_lm.vocab), CategoryBlock),
    get_y = parent_label, 
    get_items = partial(get_text_files, folders=["train", "test"]),
    splitter = GrandparentSplitter(valid_name="test")
).dataloaders(path, path=path, bs=128, seq_len=72)
```

**Model Evaluation**
- I have prepared a Text Classifier Model using Fastai API which has the accuracy of 94%. The Final Model can classify the Sentiment of the Internet Movie Database reviews (IMDB) which mean it can classify the Positive or Negative Sentiment of the reviews.
