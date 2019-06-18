import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#  load data into dataframe
fileDirectory = "D:\sentiment-labelled-sentences-data-set\sentiment labelled sentences\sentiment labelled sentences"
filepath_dict = {'yelp':   fileDirectory + '\yelp_labelled.txt',
                 'amazon': fileDirectory + '\\amazon_cells_labelled.txt',
                 'imdb':   fileDirectory + '\imdb_labelled.txt'}
df_list = pd.DataFrame(columns=['sentence', 'label', 'source'])

for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list = df_list.append(df, ignore_index=True)

# split data into test and train
sentences_train, sentences_test, y_train, y_test = train_test_split(df_list['sentence'], df_list['label'], test_size=0.20, random_state=42)

y_test = y_test.astype('int')
y_train = y_train.astype('int')

#  tokenize sentences
vectorizer = CountVectorizer(lowercase=False)
vectorizer.fit(sentences_train)
print(vectorizer.vocabulary_)
vectorizer.transform(df_list['sentence']).toarray()


X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
scoreval = np.asarray(score).astype(float)
print('Accuracy of model: {}'.format(round(score, 4)))

print('0 indicates negative review.')
print('1 indicates positive review')

sentences = ["The US military is building a case against Iran with clearer images from the latest tanker attacks",
             "Pentagon sending 1,000 U.S. troops to Middle East after oil tanker attack",
             "Infowars host Alex Jones accused of threatening Sandy Hook lawyers after child porn is found in his electronic files, court document says",
             "AMD says its Ryzen 3000 isn’t just cheaper—it’s better",
             "Grandview: Burglar gets away with cash"]


Xtest = vectorizer.transform(sentences)
print(classifier.predict(Xtest))