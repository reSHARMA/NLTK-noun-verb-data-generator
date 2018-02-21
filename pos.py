import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()
# nltk.download()
porter = PorterStemmer()
table = str.maketrans('', '', string.punctuation)
df = pd.read_csv("./data.txt")
df = df.drop(['id'],axis = 1)
output = []
for row in df.itertuples(index = False,name = 'Pandas'):
    text = getattr(row,'sentence')
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stemmed = [wordnet_lemmatizer.lemmatize(word) for word in words]
    out = pos_tag(stemmed)
    noun = []
    verb = []
    flg = 0
    for w in out:
        if(w[1] == "NN"):
            noun.append(w[0])
        if(w[1] == "PRP"):
            flg = 1
        if(w[1] == "VB"):
            verb.append(w[0])
    if(flg == 1):
        noun.append("person")
    if(len(noun) > 0 and len(verb) > 0):
        output.append((noun,verb))
res = pd.DataFrame.from_dict(output)
res.to_csv("out.csv",sep = ':')
