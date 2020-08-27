# **IMDbase-NLP**

**Fastai Library or API**
- [Fast.ai](https://www.fast.ai/about/) is the first deep learning library to provide a single consistent interface to all the most commonly used deep learning applications for vision, text, tabular data, time series, and collaborative filtering.
- [Fast.ai](https://www.fast.ai/about/) is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains, and provides researchers with low-level components that can be mixed and matched to build new approaches.

**IMDB**
- IMDb is an online database of information related to films, television programs, home videos, video games, and streaming content online – including cast, production crew and personal biographies, plot summaries, trivia, ratings, and fan and critical reviews.

**Preparing the Model**
- I have used [Fastai](https://www.fast.ai/about/) API to train the Model. It seems quite challenging to understand the code if you have never encountered with Fast.ai API before.
One important note for anyone who has never used Fastai API before is to go through [Fastai Documentation](https://docs.fast.ai/). And if you are using Fastai in Jupyter Notebook then you can use doc(function_name) to get the documentation instantly.

**Dataset**
- Fastai has its own [Dataset](https://docs.fast.ai/datasets.html).I have used [Fastai IMDB_SAMPLE](https://course.fast.ai/datasets) using the following lines of codes:

```javascript
untar_data(URLs.IMDB_SAMPLE)
```

**Creating TextDataBunch**

```javascript
TextDataBunch.from_csv(path, "...csv")
```

**Training the Model**

```javascript
language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
```

**Auto-Completion with Model**
- The Model can complete the sentence.

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1596720349/Auto_rrxfiw.png)

**Text Classifier with Fastai**

```javascript
(TextList.from_folder(path, vocab=data_lm.vocab)
            .split_by_folder(valid="test")
            .label_from_folder(classes=['neg', 'pos'])
            .databunch(bs=bs))
```

**Accuracy of the Model**
- The Model is very accurate in Text Classification.


![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1596720699/Ac_w6ecjd.png)
