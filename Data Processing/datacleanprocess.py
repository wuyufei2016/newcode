import time
import os
import re
import numpy as np
import pandas as pd
import pickle as pickle

#from nltk.stem.porter import *
#stemmer = PorterStemmer()
from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
stemmer = SnowballStemmer('english')
  
from sklearn.feature_extraction.text import HashingVectorizer

df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")
df_prod_desc = pd.read_csv('product_descriptions.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('attributes.csv', encoding="ISO-8859-1")
def clean_str(s):
    if isinstance(s, str):
        segment={"&nbsp;":" ",",":"","$":" ","?":" ","!":" ","#":" ",":":" ",";":" ","'":" ",
                "+":" ", "_":" ","\"":" ,","-":" ","//":"/","..":"."," / ":" "," \\ ":" ",
                "(":" ",")":" ",".":" . "}
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)        #Split words with a.A to a A
        s = re.sub(r"([a-z]{1})([A-Z]{1})", r"\1 \2", s) #Split words with aA to a A
        s = s.lower()                                    #lower case
        s = re.sub(r"(\s+)", r" ", s)                    #replace any number of whitespaces with one whitespace 
        
        #Replace the following characters:
        for item in list(segment.keys()):
            s=replace(item, segment[item])
        
        
        s = re.sub(r"(\s+)", r" ", s)                    #replace any number of whitespaces with one whitespace
        
        #?? Seems to remove forward slashes around words
        s = re.sub(r"(^\.|/)", r" ", s)
        s = re.sub(r"(\.|/)$", r" ", s) 
        
        #Replace spelled out numbers with digits
        strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")]) 

        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s) #whitespace between number and character 
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s) #whitespace between character and number
        
        s = re.sub(r"([0-9])( *)x( *)([0-9])", r"\1 xbi \4", s)  #Replace various spellings of 'by' to 'xbi'
        s = re.sub(r"([0-9])( *)\*( *)([0-9])", r"\1 xbi \4", s)
        s = re.sub(r"([0-9])( *)by( *)([0-9])", r"\1 xbi \4", s)
        
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)      #Remove forward slashes between characters
        
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)     #Remove whitespaces around decimal seperator, 
                                                                 #has problems with '102  . 400 . 10'
        
        #Standardize spelling of the following:
        s = re.sub(r"([0-9]+)( *)(inches|inch|in)(\.|\s)+", r"\1 in. ", s)                
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft)(\.|\s)+", r"\1 ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)(\.|\s)+", r"\1 lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)(\.|\s)+", r"\1 sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)(\.|\s)+", r"\1 cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)(\.|\s)+", r"\1 gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)(\.|\s)+", r"\1 oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)(\.|\s)+", r"\1 cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)(\.|\s)+", r"\1 mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)(\.|\s)+", r"\1 deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)(\.|\s)+", r"\1 volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)(\.|\s)+", r"\1 watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)(\.|\s)+", r"\1 amp. ", s)
        
        s = s.replace(" . "," ") #Remove full stops 
        
        return s
    else:
        return "null"

def cleaner_str(s):
    cleaner = HashingVectorizer(decode_error = 'ignore',
                           analyzer = 'word',
                           ngram_range = (1,1),
                           stop_words = 'english')
    c = cleaner.build_analyzer()
    s = (" ").join(c(s))
    return s

def stem_str(s):
    if isinstance(s, str):
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        return s
    else:
        return "null"

df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

df_attr['value'] = df_attr['value'].apply(lambda x: str(x))
df_attr = df_attr.groupby(['product_uid'])['value'].apply(lambda x: '. '.join(x)).reset_index()

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_prod_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
df_all = pd.merge(df_all, df_attr, how='left', on='product_uid')

df_all = df_all.fillna('')

n_train = df_train.shape[0]
print("Number of first rows containing training data: " + str(n_train))

spell_check_dict = {}
with open('dic.txt', 'r') as df:
    for kv in [d.strip().split(' ') for d in df]:
        spell_check_dict[kv[0]] = kv[1]

df_all['search_term'] = [spell_check_dict[x] if x in spell_check_dict else x for x in df_all['search_term'] ]



start_time = time.time()

df_all['search_term'] = [clean_str(x) for x in df_all['search_term']]
df_all['product_title'] = [clean_str(x) for x in df_all['product_title']]
df_all['product_description'] = [clean_str(x) for x in df_all['product_description']]
df_all['brand'] = [clean_str(x) for x in df_all['brand']]
df_all['value'] = [clean_str(x) for x in df_all['value']]

df_all = df_all.fillna('')

with open('df_all_clean.pkl', 'wb') as outfile:
    pickle.dump(df_all, outfile, pickle.HIGHEST_PROTOCOL)
    
print("--- Cleaning of text: %s minutes ---" % round(((time.time() - start_time)/60),2))
