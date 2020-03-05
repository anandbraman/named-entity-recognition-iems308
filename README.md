# named-entity-recognition-iems308

The task was to extract every company, CEO name and percentage from 2 years worth of Business Insider Articles. 

The data is available here: 
<br>
2013: https://www.dropbox.com/s/g43k1qzhpx3bwf5/2013.zip?dl=0
<br>
2014: https://www.dropbox.com/s/siuc4loq2fxsr5y/2014.zip?dl=0
<br><br>
The training labels are available here:
<br>
https://www.dropbox.com/sh/4tur0275szlyj2j/AADGgt4_9yOXaiwNPpCgWoF8a?dl=0

## File descriptions
`clean_final.ipynb`: the notebook where I did all my data cleaning and feature engineering. I also extracted percentages in this NB.
<br>
`modeling.ipynb`: the notebook where I did all my modeling and extraction of CEOs and company
<br>
`ner.py`: is the reduced and combined versions of these two files. If you run ner.py, provided you have the directory of articles and labels saved, you should be able to replicate the entire results folder! Make sure you have some time because I think it will take about 45 min to run. 

### Results Folder
`ceo_matches.csv` contains all the CEOs extracted from sentences correctly identified as containing a CEO by my gradient boosting classifier model.
<br>
`ceo_hiconf_FP.csv`: The training labels were incomplete so these are the names extracted from sentences where the model was highly confident that the sentence contained a CEO, but the true label was 0. It is likely some or most of these names are CEOs not included in the training labels.
<br>
`company_matches.csv` is the same as CEO matches but for companies. 
<br>
`company_hiconf_FP.csv` is the same as ceo_hiconf_FP but for companies.
<br>
`percentage_matches.csv`: same deal as `company_matches.csv` and `ceo_matches.csv` but all these percentages were identified by built in `spaCy` NER. 
<br>
`Text_Analytics_Writeup.pdf`: My writeup. Includes an executive summary and details of all my methods. 

### Running this code
Store the 2013 and 2014 articles in 'data/articles/' in your directory.
<br>
Store labels in 'data/labels/' in your directory. 
<br>
You should be able to run `clean_final.ipynb`, `modeling.ipynb`, and `ner.py` after this. 
