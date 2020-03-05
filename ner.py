'''
Description: Script that extracts percentages, CEO names and Companies from 2 years
of Business Insider articles (2013-2014)

Process: Read in articles, run spaCy nlp on articles
Extract percentages tagged by spaCy. 

Read in CEO, company training data (from data/labels). Deduplicate data. 
Filter articles for sentences containing people's names for CEO train data
Filter articles for sentences with org namies for company training data
Turn sentences into two separate dataframes. 
Assign labels to rows based on whether sentence contains name of a label, 
for ceo and company training data respectively.

Engineer Features:
count number of capital letters
count number of words in sentence. 
contains quote
contains 'top 10' ceo or company name from 2013
contians word 'CEO' or 'company'

Write company data and ceo data to csvs in data folder.

MODELING:
Train test split
TF-IDF vectorization of sentences with top 10k features

Companies: 
Train LR model, make predictions
Extract orgs from True positive sentences, write to csv
Extract orgs from high confidence False Positive Sentences, write to csv

CEOs:
Train GBC, make predictions
Extract persons from True positive sentences, write to csv
Extract persons from high confidence false positive sentences, write to csv 
'''

import pandas as pd
import re
import spacy
import os
import string
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
import nltk
from tqdm import tqdm

files = os.listdir('data/articles/')
# filter hidden files
files = [file for file in files if file != ".DS_Store"]

text_dat = []
for file in files:
    fname = 'data/articles/' + file
    f = open(fname, encoding='utf8', errors='replace')
    article = f.read()
    # removing nonsense chars
    printable = set(string.printable)
    article = ''.join(filter(lambda x: x in printable, article))
    text_dat.append(article)

f.close()

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

articles_processed = [nlp(article) for article in text_dat]

### Percentage Extraction

def return_percentages(article_spacified):
	'''
	iterates through entities in article and returns those tagged as 
	a percentage
	'''
    pcts = [ent.text for ent in article_spacified.ents if ent.label_ == "PERCENT"]
    return pcts

pct_lst = []
for article in articles_processed:
    pcts = return_percentages(article)
    pct_lst.extend(pcts)

pct_df = pd.DataFrame({'percentages' : pct_uniq})
pct_df.to_csv("results/percentage_matches.csv")


### Labeling and Feature Engineering
ceos = pd.read_csv("data/labels/ceo.csv", header=None, encoding="ISO-8859-1")
companies = pd.read_csv("data/labels/companies.csv", header=None)
# some ceos have two spaces between f name l name
ceos[2] = [' '.join(s.split()) for s in ceos[2]]
# deduplication
ceo_lst = list(set(ceos[2]))

# deduplication of companies
company_lst = list(set(companies[0]))

# removing nonsense overlap
not_ceos = ['American Apparel','Interpublic Group', 'Office Depot',
            'Pershing Square', 'Third Point', 'Swanson']
not_companies = ['Murray Kessler', 'Steve Blumenthal', 'Garland', 
                 'Paul Singer', 'Laurent Potdevin']
ceo_lst = [ceo for ceo in ceo_lst if ceo not in not_ceos]
company_lst = [company for company in company_lst if company not in not_companies]

def filter_sentences(space_article, entity):
	'''
	Filters a spaCy doc for sentences containing a specific entity
	in this case it will be 'PERSON' for the CEO data, and 'ORG' for the 
	company data. 
	'''
    sentences = [sent.text for sent in space_article.sents]
    entities = [ent.text for ent in space_article.ents if ent.label_ == entity]
    sent_w_ent = []
    for s in sentences:
        if any(ent in s for ent in entities):
            sent_w_ent.append(s)
    
    return sent_w_ent

org_sentences = []
for article in tqdm(articles_processed):
	'''
	Creates list of sentences containing ORGs 
	from the corpora
	'''
    sent_w_org = filter_sentences(article, "ORG")
    org_sentences.extend(sent_w_org)


person_sentences = []
for article in tqdm(articles_processed):
	'''
	Creates list of sentences containing PERSONs 
	from the corpora
	'''
    sent_w_person = filter_sentences(article, "PERSON")
    person_sentences.extend(sent_w_person)


## DataFrames
org_df = pd.DataFrame({'sentences': org_sentences})
person_df = pd.DataFrame({'sentences': person_sentences})
def match(sentence, labels=None):
	'''
	Check if any item from a list 
	is contained in a sentence. 
	'''
    if any(lbl in sentence for lbl in labels):
        return 1
    else:
        return 0

# constructing labels for each dataset
org_df['y'] = org_df.sentences.apply(lambda x: match(x, labels=company_lst))
person_df['y'] = person_df.sentences.apply(lambda x: match(x, labels=ceo_lst))


def count_caps(sent):
	'''
	Count capital letters in a sentence
	'''
    return sum([1 for s in sent if s.isupper()])

org_df['num_cap'] = org_df.sentences.apply(lambda x: count_caps(x))
person_df['num_cap'] = person_df.sentences.apply(lambda x: count_caps(x))

def num_words(sent):
	'''
	Count number of words in a sentence. 
	'''
    tokes = sent.split(" ")
    return len(tokes)


org_df['num_words'] = org_df.sentences.apply(lambda x: num_words(x))
person_df['num_words'] = person_df.sentences.apply(lambda x: num_words(x))

def contains_quote(text):
	'''
	Check if sentence contains a quote
	'''
    if '\'' in text or '\"' in text:
        return 1
    else:
        return 0


person_df['has_quote'] = person_df.sentences.apply(lambda x: contains_quote(x))
org_df['has_quote'] = org_df.sentences.apply(lambda x: contains_quote(x))

# using an article from 2013 about highly rated ceos
best_ceos = ['Mark Zuckerberg', 'Zuckerberg', 'Bill McDermott', 'McDermott', 'Jim Hagemann Snabe',
                      'Snabe', 'Jim Snabe', 'Dominic Barton', 'Barton', 'Jim Turley', 'Turley', 'John Schlifske',
                     'Schlifske', 'Jeff Bezos', 'Bezos', 'Page', 'Larry Page', 'Musk', 'Elon Musk']
# same thing but for companies
best_companies = ['Google, Inc.', 'Google', 'SAS', 'Facebook', 
                  'Guidewire', 'Workday', 'Intel', 'YouTube',
                 'Amazon', 'Tesla']


org_df['best_company'] = org_df.sentences.apply(lambda x: match(x, best_companies))
person_df['best_ceo'] = person_df.sentences.apply(lambda x: match(x, best_ceos))


org_df['contain_company'] = org_df.sentences.apply(lambda x: match(x, ['company','Company']))
person_df['contain_ceo'] = person_df.sentences.apply(lambda x: 
                                                     match(x, ['CEO', 'Chief executive', 'chief executive']))


# writing dfs to csv
org_df.to_csv('data/labeled_org_data.csv', index=False)
person_df.to_csv('data/labeled_person_data.csv', index=False)

from sklearn.model_selection import train_test_split
# train test split org data
X_org_train, X_org_test, y_org_train, y_org_test = train_test_split(org_df.drop('y', axis=1), 
                                                                    org_df.y, test_size=0.25)

# train test split person data
X_per_train, X_per_test, y_per_train, y_per_test = train_test_split(person_df.drop('y', axis=1), person_df.y, 
                                                                    test_size=0.25)


### COMPANY MODELING  ####
from sklearn.feature_extraction.text import TfidfVectorizer
# fitting on the train set because I'm a good data scientist :)
tfidf_org = TfidfVectorizer(ngram_range=(1,3), 
                           stop_words='english',
                           max_features=10000).fit(X_org_train.sentences)

from scipy.sparse import hstack
# transforming test and train data
org_text_train = tfidf_org.transform(X_org_train.sentences)
org_text_test = tfidf_org.transform(X_org_test.sentences)


# final train and test data for orgs
X_org_train = hstack([X_org_train.drop("sentences", axis=1), org_text_train])
X_org_test = hstack([X_org_test.drop("sentences",axis=1), org_text_test])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# fitting logistic regression model, class weight is important
# because we are dealing with some mad class imbalance
lr_org = LogisticRegression(max_iter=5000, class_weight='balanced')
lr_org.fit(X_org_train, y_org_train)

# train preds
y_train_pred = lr_org.predict(X_org_train)

# test predictions
y_test_pred = lr_org.predict(X_org_test)
print("Company test set results")
print(classification_report(y_org_test, y_test_pred))

### Company Extraction 
companies_train = y_org_train[(y_org_train == y_train_pred) & (y_org_train ==1)]
companies_test = y_org_test[(y_org_test == y_test_pred) & (y_org_test == 1)]

# indices
true_pos_indices = companies_train.index.append(companies_test.index).to_list()
# selecting rows by index
company_df = org_df.iloc[true_pos_indices]

company_sentences = company_df.sentences.to_list()

# Extracting TRUE POSITIVES
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
company_processed = [nlp(sentence) for sentence in company_sentences]

def return_companies(sentence_spacified):
	'''
	Company version of return_percentages
	'''
    company = [ent.text for ent in sentence_spacified.ents if ent.label_ == "ORG"]
    return company


company_lst = []
for sent in company_processed:
    companies = return_companies(sent)
    company_lst.extend(companies)

unique_companies = list(set(company_lst))

companies_final = pd.DataFrame({'companies': unique_companies})
companies_final.to_csv("results/company_matches.csv",index=False)

# Extracting HIGH CONFIDENCE FALSE POSITIVES
# same process as the previous extraction
conf_train = lr_org.decision_function(X_org_train)
compFP_train = y_org_train[(y_org_train != y_train_pred) & (y_train_pred ==1) & (conf_train >= 2.0)]

conf_test = lr_org.decision_function(X_org_test)
compFP_test = y_org_test[(y_org_test != y_test_pred) & (y_test_pred == 1) & (conf_test > 2.0)]

company_FP = compFP_train.index.append(compFP_test.index).to_list()
company_FP_df = org_df.iloc[company_FP]
company_FP_processed = [nlp(sentence) for sentence in company_FP_df.sentences.to_list()]

company_FP_lst = []
for sent in company_FP_processed:
    company_FP = return_companies(sent)
    company_FP_lst.extend(company_FP)

company_FP_lst = list(set(company_FP_lst))

companyDF_fp = pd.DataFrame({'company': company_FP_lst})
companyDF_fp.to_csv('results/companies_hiconf_FP.csv', index=False)

### CEO MODELING ####
# tfidf time
tfidf_per =  TfidfVectorizer(ngram_range=(1,3), 
                           stop_words='english',
                            max_features=10000).fit(X_per_train.sentences)

per_text_train = tfidf_per.transform(X_per_train.sentences)
per_text_test = tfidf_per.transform(X_per_test.sentences)
X_per_train = hstack([X_per_train.drop('sentences',axis=1), per_text_train])
X_per_test = hstack([X_per_test.drop('sentences',axis=1), per_text_test])

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(verbose=1, learning_rate=0.6, n_estimators=100)
gbc.fit(X_per_train, y_per_train)

y_train_gbm = gbc.predict(X_per_train)
y_test_gbm = gbc.predict(X_per_test)
print("CEO test set results")
print(classification_report(y_per_test, y_test_gbm))

ceos_train = y_per_train[(y_per_train == y_train_gbm) & (y_per_train == 1)]
ceos_test= y_per_test[(y_per_test == y_test_gbm) & (y_per_test == 1)]
ceos_idx = ceos_train.index.append(ceos_test.index).to_list()

ceo_df = person_df.iloc[ceos_idx]
ceo_sentences = [nlp(sent) for sent in ceo_df.sentences.to_list()]
def return_ceos(sentence_spacified):
	'''
	returns the person from a sentence containing a ceo
	'''
    ceo = [ent.text for ent in sentence_spacified.ents if ent.label_ == "PERSON"]
    return ceo

ceo_lst = []
for sent in ceo_sentences:
    ceo = return_ceos(sent)
    ceo_lst.extend(ceo)

ceo_lst = list(set(ceo_lst))
ceo_df = pd.DataFrame({'ceo': ceo_lst})
ceo_df.to_csv('results/ceo_matches.csv')

## CEO HIGH CONFIDENCE FALSE POSITIVES
# including high confidence false positives as these may be 
# ceos not included in the training set

ceo_conf_train = gbc.decision_function(X_per_train)
ceosFP_train = y_per_train[(y_per_train != y_train_gbm) & (y_train_gbm==1) & (ceo_conf_train >= 2.0)]

ceo_conf_test = gbc.decision_function(X_per_test)
ceosFP_test = y_per_test[(y_per_test != y_test_gbm) & (y_test_gbm==1) & (ceo_conf_test >= 2.0)]

ceosFP_idx = ceosFP_train.index.append(ceosFP_test.index).to_list()
ceoFP_df = person_df.iloc[ceosFP_idx]
ceoFP_sent = [nlp(sent) for sent in ceoFP_df.sentences.to_list()]

ceoFP_lst =[]
for sent in ceoFP_sent:
    ceoFP = return_ceos(sent)
    ceoFP_lst.extend(ceoFP)

ceoFP_lst = list(set(ceoFP_lst))

ceo_hiconf_FP = pd.DataFrame({'ceo': ceoFP_lst})
ceo_hiconf_FP.to_csv('results/ceo_hiconf_FP.csv', index=False)

