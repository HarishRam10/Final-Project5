#%%
import pandas as pd

df_train = pd.read_csv(r'F:\Github\NLP_FinalProject\SST-2\train.tsv', sep='\t')
df_val = pd.read_csv(r'F:\Github\NLP_FinalProject\SST-2\dev.tsv', sep='\t')
df_test = pd.read_csv(r'F:\Github\NLP_FinalProject\SST-2\test.tsv', sep='\t')
#%%
# save label
train_label = df_train['label']
val_label = df_val['label']
#%%
# concat data
df = pd.concat([df_train.iloc[:,0:1], df_val.iloc[:,0:1], df_test.iloc[:,1:]], sort=False)
df.reset_index(drop = True, inplace = True)
#%%

# Create stopwords
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
nltk_stop_words = nltk.corpus.stopwords.words('english')

stop_words = set(nltk_stop_words).union(sklearn_stop_words)
stop_words.remove('no') # well
print(len(stop_words))


# convert pos_tag
from nltk.corpus import wordnet

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# handle the 'string list', remove '[ ]'
import re
def replace_s(text):
    text = re.sub(r'[\[\]\,\']','' ,text)
    return text
# %%
# NLP Normlization
import nltk
from nltk import TweetTokenizer
import neuspell
from neuspell import BertChecker

Tt_Tokenizer = TweetTokenizer()
wnl = nltk.WordNetLemmatizer()

""" select spell checkers & load """
checker = BertChecker()
checker.from_pretrained()
# %%

# set minimum words number
length_lim = 0

x=0
normal_reviews = []
length =[]

for i in range(len(df)):

    txt  = df.iloc[i]['sentence']
    # correct missspelling
    checker.correct(txt)
    # tokenize
    tokens = Tt_Tokenizer.tokenize(txt)

    normal_review = []
    # lemma
    # lower
    # stopwords
    for t in nltk.pos_tag(tokens):
        pos_tag = get_wordnet_pos(t[1])    
        if (t[0].isalpha()) and (pos_tag != None):
            lemma_word = wnl.lemmatize(t[0].lower(), pos=pos_tag)
            if lemma_word not in stop_words: 
                normal_review.append(lemma_word)

        if (t[0].isalpha()) and (pos_tag == None) and (t[0].lower() not in stop_words):
            normal_review.append(t[0].lower())

    # length
    if len(normal_review) >= length_lim:
        normal_reviews.append(replace_s(str(normal_review)))

    length.append(len(normal_review))

    x += 1
    print(x)
#%%
# add new cols
df['Normalized Review'] = normal_reviews
df['length'] = length
#%%
# remove empty col
# df = df[df['length'] != 0]   # 68988 rows remain
#%%
df
#%%
# Separating the training data
df_train = df.iloc[:df_train.shape[0], :]

# Separating the validation data
df_val = df.iloc[df_train.shape[0]:df_train.shape[0] + df_val.shape[0], :]

# Separating the test data
df_test = df.iloc[df_train.shape[0] + df_val.shape[0]:, :]
#%%
df_train['label'] = train_label
df_val['label'] = val_label

df_val.reset_index(drop = True, inplace = True)
df_test.reset_index(drop = True, inplace = True)
# %%
df_train.to_csv(r'F:\Github\NLP_FinalProject\Data\df_corrected\df_train.csv', encoding="'utf-8-sig'")
df_val.to_csv(r'F:\Github\NLP_FinalProject\Data\df_corrected\df_val.csv', encoding="'utf-8-sig'")
df_test.to_csv(r'F:\Github\NLP_FinalProject\Data\df_corrected\df_test.csv', encoding="'utf-8-sig'")


# %%

# %%
