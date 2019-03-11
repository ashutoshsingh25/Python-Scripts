
# coding: utf-8

# In[ ]:


import pandas as pd
import xlrd
import csv
import fasttext
import re
import string
import numpy as np
from nltk.corpus import stopwords
from collections import defaultdict
from os import listdir
from os.path import isfile, join
import os
import random
import nltk
import csv

nltk.download('stopwords')
stop_words = stopwords.words('english')


# In[ ]:


input_file_name = "/home/divyansh/excavator_subcat_155_new.xlsx"
subcat_name = "155"
base_directory="./FastText_Waterfall_Model/"


# In[ ]:


if not os.path.exists(str(base_directory)+"subcat_"+subcat_name):
    os.mkdir(str(base_directory)+"subcat_"+subcat_name)
    print("Directory " , str(base_directory)+"subcat_"+subcat_name ,  " Created ")
else:    
    print("Directory " , str(base_directory)+"subcat_"+subcat_name ,  " already exists")


# In[ ]:


#path where all xmls are kept
base_directory = base_directory+"subcat_"+subcat_name+"/"

os.mkdir(str(base_directory)+"pmcat/")
os.mkdir(base_directory+"child/")

def create_folders(name):
    input_path= base_directory+name+"/"+"input_"+subcat_name+"/"
    os.mkdir(input_path)
    output_path= base_directory+name+"/"+"output_"+subcat_name+"/"
    os.mkdir(output_path)
    KFold_Validation_Files_path= base_directory+name+"/"+"kFold_validation_"+subcat_name+"/"
    os.mkdir(KFold_Validation_Files_path)
    model_path=base_directory+name+"/"+"model_"+subcat_name+"/"
    os.mkdir(model_path)
    return True

create_folders("pmcat")
create_folders("child")

combined_parsed_file_name="combined.txt"
combined_shuffled_file_name="combined_shuffled.txt"
train_file_name="singleLabel_train_"+subcat_name
test_file_name="single_label_test_"+subcat_name
model_name="supervised_classifier_model_CATID_"+subcat_name
split=.9


# In[ ]:


url = "http://fts-master.intermesh.net:8020/solr/mcat/select?q.alt=*:*&fl=id,name,catid,catname,parentmcat&wt=csv&csv.separator=%09&rows=200000"
mcat_data = pd.read_csv("mcat_data.csv" ,sep='\t')


# In[ ]:


with open('../complete_product_data/mcat_data/mcat_data.csv', mode='r') as infile:
    reader = csv.reader(infile, delimiter='\t')
    mcat_data_id_name = {rows[0]:rows[1] for rows in reader}


# In[ ]:


from collections import OrderedDict

def remove_stopwords_and_duplicate_words(df, fieldName):
    df[fieldName] = df[fieldName].apply(lambda x: ' '.join([item for item in str(x).split() if item not in stop_words]))
    df[fieldName] = df[fieldName].str.split().apply(lambda x: OrderedDict.fromkeys(str(x).split()).keys()).str.join(' ')
    df[fieldName] = df[fieldName].replace('[^\w\s]','', regex=True).replace(' +',' ', regex=True)
    return df

def remove_punctuation(df, fieldName):
    excludePunctuation = set(string.punctuation)
    df[fieldName] = df[fieldName].apply(lambda x: ' '.join([item for item in str(x).split() if item not in excludePunctuation]))
    return df

def get_bulleted_data(df):
    regex = re.compile("<ul>(.*?)</ul>")
    df['REFINED_SMALL_DESC_BULLET'] = df.SMALL_DESC.apply(lambda x: [' '.join(re.findall(regex,str(x))) if len(re.findall(regex,str(x))) > 0 else '']).str.join(' ')
#     df['REFINED_SMALL_DESC'] = df.SMALL_DESC.apply(lambda x: print(x))

    return df

def get_bold_data(df):
    regex = re.compile("<b>(.*?)</b>")
    df['REFINED_SMALL_DESC_BOLD'] = df.SMALL_DESC.apply(lambda x: [' '.join(re.findall(regex,str(x))) if len(re.findall(regex,str(x))) > 0 else '']).str.join(' ')
#     df['REFINED_SMALL_DESC'] = df.SMALL_DESC.apply(lambda x: print(x))

    return df

def get_table_data(df):
    regex = re.compile("<td>(.*?)</td>")
    df['REFINED_SMALL_DESC_TABLE'] = df.SMALL_DESC.apply(lambda x: [' '.join(re.findall(regex,str(x))) if len(re.findall(regex,str(x))) > 0 else '']).str.join(' ')
#     df['REFINED_SMALL_DESC'] = df.SMALL_DESC.apply(lambda x: print(x))

    return df

def refine_dataframe_title(df):
    df['REFINED_TITLE'] = df['TITLE'].str.lower().replace(' +',' ', regex=True).replace('[^\w\s]','', regex=True).replace(' +',' ', regex=True)
    df = remove_stopwords_and_duplicate_words(df,'REFINED_TITLE')
    
#     display(df[['TITLE','REFINED_TITLE']])
    return df

def refine_dataframe_description(df):
#     df.SMALL_DESC.fillna(value=pd.np.nan, inplace=True)
    df = get_bulleted_data(df)
#     df = get_bold_data(df)
#     df = get_table_data(df)
#     df['REFINED_SMALL_DESC'] = df['REFINED_SMALL_DESC_BULLET']+ " " + df['REFINED_SMALL_DESC_BOLD']+ " " + df['REFINED_SMALL_DESC_TABLE']
    df['REFINED_SMALL_DESC'] = df['REFINED_SMALL_DESC_BULLET']
    df['REFINED_SMALL_DESC'] = df['REFINED_SMALL_DESC'].replace('<li>',' ',regex=True).replace('</li>',' ',regex=True).replace(' +', ' ',regex=True).replace('</li li>',' ',regex=True).replace('<ol>',' ',regex=True).replace('</ol/>',' ',regex=True).replace('</ul>',' ',regex=True).replace('<ul>',' ',regex=True).replace('<p>',' ',regex=True).replace('</p>',' ',regex=True).replace('<sup>',' ',regex=True).replace('</sup>',' ',regex=True).replace('<b>',' ',regex=True).replace('</b>',' ',regex=True).replace('<br/>',' ',regex=True).replace('<br>',' ',regex=True).replace('<br />',' ',regex=True).replace('[[:punct:]]',' ',regex=True).replace('[0-9] \\w+ *', '',regex=True).replace('[0-9]\\w+ *', '',regex=True).replace('[0-9]', '',regex=True).replace('\\s+',' ',regex=True).replace(' +', ' ',regex=True).replace(np.nan, '', regex=True)
    df = remove_punctuation(df, 'REFINED_SMALL_DESC')
    df['REFINED_SMALL_DESC'] = df['REFINED_SMALL_DESC'].str.lower()
    df = remove_stopwords_and_duplicate_words(df,'REFINED_SMALL_DESC')
    
#     display(df[['SMALL_DESC','REFINED_SMALL_DESC']])
    return df

def refine_dataframe_isq(df):
    df['REFINED_ISQ'] = df['ISQ_DETAILS4_INDEX'].str.lower().replace(' +',' ', regex=True).str.replace('[^\w\s]','', regex=True).str.replace('[0-9] \\w+ *','', regex=True).str.replace('[0-9]\\w+ *', '', regex=True).str.replace('[0-9]', '', regex=True).str.replace(' +',' ', regex=True).replace(np.nan, '', regex=True)
    df = remove_punctuation(df, 'REFINED_ISQ')
    df = remove_stopwords_and_duplicate_words(df,'REFINED_ISQ')

#     display(df[['ISQ_DETAILS4_INDEX','REFINED_ISQ']])
    return df

def refine_dataframe_combinedCorpus(df):
    df['REFINED_COMBINED_DATA'] = df['COMBINED_DATA'].replace(' +',' ', regex=True).replace('NA','', regex=True).str.lower().str.replace('[^\w\s]','', regex=True).str.replace(' +',' ', regex=True)
    df = remove_punctuation(df, 'REFINED_COMBINED_DATA')
    df = remove_stopwords_and_duplicate_words(df,'REFINED_COMBINED_DATA')

#     display(df[['COMBINED_DATA','REFINED_COMBINED_DATA']])
    return df

def clean_and_preProcess_data(df_data):
    df_refined = refine_dataframe_title(df_data)
    df_refined = refine_dataframe_description(df_refined)
    df_refined = refine_dataframe_isq(df_refined)
    
    df_refined['COMBINED_DATA'] = df_refined['REFINED_TITLE']+ " " + df_refined['REFINED_SMALL_DESC']+ " " + df_refined['REFINED_ISQ']
    
    df_refined = refine_dataframe_combinedCorpus(df_refined)
    df_refined['PRIME_MCAT_NAME'] = df_refined['PRIME_MCAT_NAME'].replace(' ','_', regex=True)
    df_refined['SEARCH_TOPMOST_MCAT'] = df_refined['SEARCH_TOPMOST_MCAT'].str.lower()
    df_refined['FASTTEXT_CORPUS'] = "__label__" + df_refined['PRIME_MCAT_NAME']+ " " + df_refined['REFINED_COMBINED_DATA'] + " " + df_refined['SEARCH_TOPMOST_MCAT']

#     display(df_refined[['FASTTEXT_CORPUS','PRIME_MCAT_NAME','REFINED_COMBINED_DATA']])
    df_refined['FASTTEXT_CORPUS'].replace("", np.nan, inplace=True)
    df_refined.FASTTEXT_CORPUS.dropna()
    return df_refined


# In[ ]:


stop_words.extend(("we","are","dealing","quality","manufacturers","manufacturer","exporters","supplier","dealer",
                 "good","topmost","business","trusted","finest","offer","offering","involved","provide","reputed",
                   "company","organization","trader","trading","inr","indian","rupees",
                    "rupee","features","specifications","material","feature","specification","materials",
                   "size","li","pvt.","ltd","pvt","ltd."))

# title,mcatid,mcatname,catid,catname,smalldesc,isq

def get_file_names_to_parse(input_path):
    files_to_parse = [join(input_path,f) for f in listdir(input_path) if isfile(join(input_path, f))]
    print("Total Files to Parse="+str(len(files_to_parse)))
    return files_to_parse

def generateLabel(mcatNames,combinedTitleDescIsq,prime_mcat_name):
    label_prefix="__label__"
    parsed_strings=[]
    labels=""
    fastTextLabel=""
#     print(prime_mcat_name)
    if prime_mcat_name != "" and combinedTitleDescIsq != "" and combinedTitleDescIsq not in ['\n', '\r\n']:
        labels = label_prefix+prime_mcat_name.replace(" ","_")
#         for eachMcat in mcatNames:
#             if labels == "":
#                 labels=labels+label_prefix+eachMcat.replace(" ","_")
#             else:
#                 labels=labels+" "+label_prefix+eachMcat.replace(" ","_")
    
        if labels!="":
            mcatName_str = ' '.join(mcatNames)            
            fastTextLabel= labels+" "+combinedTitleDescIsq+" "+mcatName_str
    return fastTextLabel

def refined_combined_title_desc_isq(combinedWord):
    combinedWord = re.sub(' +', ' ',combinedWord)
    combinedWord = re.sub(r'NA','',combinedWord)
    combinedWord = combinedWord.lower()
    combinedWord = (' '.join(remove_duplicates(combinedWord.split()))).lstrip()
    combinedWord = re.sub(r'[0-9] \\w+ *','',combinedWord)
    combinedWord = re.sub(r'[0-9]\\w+ *','',combinedWord)
    combinedWord = re.sub(r'[0-9]','',combinedWord)
    exclude = set(string.punctuation)
    combinedWord = ''.join(ch for ch in combinedWord if ch not in exclude)
    combinedWord = re.sub(r':','',combinedWord)
    combinedWord = removeStopWords(combinedWord)
    combinedWord = re.sub(' +', ' ',combinedWord)
    
    return combinedWord

def preProcess(file):
    parsed_strings=[]
    df_data = pd.read_csv(file, sep='\t', engine='python')
    df_refined = refine_dataframe_title(df_data)
    df_refined = refine_dataframe_description(df_refined)
    df_refined = refine_dataframe_isq(df_refined)
    
    df_refined['COMBINED_DATA'] = df_refined['REFINED_TITLE']+ " " + df_refined['REFINED_SMALL_DESC']+ " " + df_refined['REFINED_ISQ']
    
    df_refined = refine_dataframe_combinedCorpus(df_refined)
    
    df_refined['FASTTEXT_CORPUS'] = "__label__" + df_refined['PRIME_MCAT_NAME']+ " " + df_refined['REFINED_COMBINED_DATA']

#     display(df_refined[['FASTTEXT_CORPUS','PRIME_MCAT_NAME','REFINED_COMBINED_DATA']])
    
    df_refined.FASTTEXT_CORPUS.dropna()
    
    parsed_strings = df_refined['FASTTEXT_CORPUS'].tolist()
    
#     with open(file, mode='r') as csvfile:
#         readCSV = csv.reader(csvfile, delimiter='\t')
#         for column in readCSV:
#             combinedTitleDescIsq = ""
#             mcatNames = []
#             prime_mcat_name = ""
#             refinedTitle = ""
#             refinedIsq = ""
#             refinedDescription = ""
    
#             if len(column) == 7 and column[0] != "":
#                 title = column[0]
#                 refinedTitle = refineTitle(title)
# #                 print(refinedTitle)
#                 if column[1] != "":
#                     smallDesc = column[1]
#                     regex = re.compile("<ul>(.*?)</ul>")
# #                     regex = "<ul>" + '(.*?)' + "</ul>"
#                     impDesc = re.findall(regex,smallDesc)
#                     if len(impDesc) > 0:
#                         smallDesc = " ".join(impDesc)
#                         refinedDescription = refine_description(smallDesc)
# #                         print(refinedDescription)
#                     else:
#                         refinedDescription = ""
# #                     print(refinedDescription)
#                 if column[5] != "":
#                     isq = column[5]
# #                     allisqs = re.split(r'\t', isq)
#                     refinedIsq = refineIsq(isq)

#                 if column[2] != "":
#                     prime_mcat_name = column[2]
#             else:
#                 next
            
#             if refinedTitle != "" and refinedTitle != "title":
#                 combinedTitleDescIsq = combinedTitleDescIsq + refinedTitle
#                 if refinedDescription != "" and refinedDescription != "smalldesc":
#                     combinedTitleDescIsq = combinedTitleDescIsq + " " + refinedDescription
#                 if refinedIsq != "" and refinedIsq != "isq":
#                     combinedTitleDescIsq = combinedTitleDescIsq + " " + refinedIsq
                
#                 combinedTitleDescIsq = refined_combined_title_desc_isq(combinedTitleDescIsq)
#                 fastTextLabel = generateLabel(mcatNames,combinedTitleDescIsq,prime_mcat_name)
#                 parsed_strings.append(fastTextLabel)
#             else:
#                 next
    return parsed_strings

def remove_duplicates(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist

def refineTitle(title):
    title = re.sub(' +', ' ',title)
    title = title.lower()
    title = re.sub(r'[[:punct:]]','',title)
    exclude = set(string.punctuation)
    title = ''.join(ch for ch in title if ch not in exclude)
    title = re.sub(r'[0-9] \\w+ *', '',title)
    title = re.sub(r'[0-9]\\w+ *', '',title)
    title = re.sub(r'[[:digit:]]','',title)
    title = re.sub(' +', ' ',title)
    
    return title
    
def refineIsq(allisqs):
    concatinatedIsq = re.sub(' +', ' ',allisqs)
    concatinatedIsq = concatinatedIsq.lower()
    concatinatedIsq = (' '.join(remove_duplicates(concatinatedIsq.split()))).lstrip()
    concatinatedIsq = re.sub(r'[^\w\s]','',concatinatedIsq)
    concatinatedIsq = re.sub(r'[[:punct:]]',' ',concatinatedIsq)
    exclude = set(string.punctuation)
    concatinatedIsq = ''.join(ch for ch in concatinatedIsq if ch not in exclude)
    concatinatedIsq = re.sub(r'[0-9] \\w+ *','', concatinatedIsq)
    concatinatedIsq = re.sub(r'[0-9]\\w+ *','', concatinatedIsq)
    concatinatedIsq = re.sub(r'[0-9]','', concatinatedIsq)
    concatinatedIsq = re.sub(' +', ' ',concatinatedIsq)
    concatinatedIsq = concatinatedIsq.lstrip()
    concatinatedIsq = concatinatedIsq.rstrip()
    concatinatedIsq = removeStopWords(concatinatedIsq)
    concatinatedIsq = re.sub(' +', ' ',concatinatedIsq)
    
    return concatinatedIsq

def refine_description(smallDesc):
    
    smallDesc = smallDesc.lower()
    smallDesc = re.sub(r',',' ',smallDesc)
    smallDesc = re.sub(' +', ' ',smallDesc)
    smallDesc = re.sub(r'</li li>',' ',smallDesc)
    smallDesc = re.sub(r'</li>',' ',smallDesc)
    smallDesc = re.sub(r'<li>',' ',smallDesc)
    smallDesc = re.sub(r'<ol>',' ',smallDesc)
    smallDesc = re.sub(r'</ul>',' ',smallDesc)
    smallDesc = re.sub(r'<ul>',' ',smallDesc)
    smallDesc = re.sub(r'<p>',' ',smallDesc)
    smallDesc = re.sub(r'</p>',' ',smallDesc)
    smallDesc = re.sub(r'<sup>',' ',smallDesc)
    smallDesc = re.sub(r'</sup>',' ',smallDesc)
    smallDesc = re.sub(r'<b>',' ',smallDesc)
    smallDesc = re.sub(r'<br />',' ',smallDesc)
    smallDesc = re.sub(r'[[:punct:]]',' ',smallDesc)
    exclude = set(string.punctuation)
    smallDesc = ''.join(ch for ch in smallDesc if ch not in exclude)
    smallDesc = re.sub(r'[0-9] \\w+ *', '',smallDesc)
    smallDesc = re.sub(r'[0-9]\\w+ *', '',smallDesc)
    smallDesc = re.sub(r'[0-9]', '',smallDesc)
    smallDesc = re.sub(r'\\s+',' ',smallDesc.strip())
    smallDesc = (' '.join(remove_duplicates(smallDesc.split()))).lstrip()
    smallDesc = re.sub(' +', ' ',smallDesc)
    smallDesc = removeStopWords(smallDesc)
    smallDesc = re.sub(' +', ' ',smallDesc)
    
    return smallDesc

def removeStopWords(word):
    tokenized_word = word.split()
    list_of_word = []
    [list_of_word.append(word) for word in tokenized_word if word not in stop_words]
    finalWord = " ".join(list_of_word)
    
    return finalWord

def write_to_file(parsed_strings,output_path,output_file_name,new_line_flag):
    train_data = open(output_path+output_file_name, 'w')
    for item in parsed_strings:
        if new_line_flag == True:
            train_data.write(str(item)+"\n")
        else:
            train_data.write(str(item))
    train_data.close()
    
def randomly_shuffle_file(input_file,output_file):
    data_array=[]
    with open(input_file,'r') as source:
        data = [(random.random(), line) for line in source ]
    data.sort()
    with open(output_file,'w') as target:
        for _, line in data:
            target.write( line )
            data_array.append(line)
    return data_array
            
def merge_files_in_folder(folder_path,output_filename):
    filenames = get_file_names_to_parse(folder_path)
    with open(folder_path+output_filename, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


# In[ ]:



def readCsvOrExcel(file):
#     df = pd.read_csv(file,sep='\t')
    df = pd.read_excel(file,sheet_name=None)
    return df

def getPmcatData(complete_dataFrame):
    # Get Good and Super PMCAT Data
    df_pmcat_data = df[(df.FK_MCAT_TYPE_ID == 1) | (df.FK_MCAT_TYPE_ID == 2)]
    df_pmcat_data_fields = df_pmcat_data[['TITLE','SMALL_DESC','PRIME_MCAT_NAME','FK_MCAT_TYPE_ID','PRIME_SUBCAT_ID','ISQ_DETAILS4_INDEX','MAP_MCAT_NAMELIST','SEARCH_TOPMOST_MCAT']].copy()
    
    # Get Thin PMCAT data with their NAME if its PMCAT is not Good PMCAT
    df_pmcat_thin = df[(df.FK_MCAT_TYPE_ID == 3) & (df.PMCAT_MCAT_TYPE != 2)]
    df_pmcat_thin_fields = df_pmcat_thin[['TITLE','SMALL_DESC','PRIME_MCAT_NAME','FK_MCAT_TYPE_ID','PRIME_SUBCAT_ID','ISQ_DETAILS4_INDEX','MAP_MCAT_NAMELIST','SEARCH_TOPMOST_MCAT']].copy()
    
    df_pmcat_thinAndChild = df[(df.PMCAT_MCAT_TYPE == 2)]
    df_pmcat_thinAndChild_fields = df_pmcat_thinAndChild[['TITLE','SMALL_DESC','PMCAT_NAME','PMCAT_MCAT_TYPE','PRIME_SUBCAT_ID','ISQ_DETAILS4_INDEX','MAP_MCAT_NAMELIST','SEARCH_TOPMOST_MCAT']].copy()
    df_pmcat_thinAndChild_fields.rename(columns={'PMCAT_NAME': 'PRIME_MCAT_NAME','PMCAT_MCAT_TYPE': 'FK_MCAT_TYPE_ID'}, inplace=True)
    
    # Loose Mcats directly mapped in subcats
    df_looseMcats = df[df.PMCAT_NAME.isna()]
    df_looseMcats_fields = df_looseMcats[['TITLE','SMALL_DESC','PMCAT_NAME','PMCAT_MCAT_TYPE','PRIME_SUBCAT_ID','ISQ_DETAILS4_INDEX','MAP_MCAT_NAMELIST','SEARCH_TOPMOST_MCAT']].copy()
    df_looseMcats_fields.rename(columns={'PMCAT_NAME': 'PRIME_MCAT_NAME','PMCAT_MCAT_TYPE': 'FK_MCAT_TYPE_ID'}, inplace=True)
    
    combined_df_pmcat = pd.concat([df_pmcat_data_fields, df_pmcat_thinAndChild_fields, df_looseMcats_fields, df_pmcat_thin_fields])
    
    return combined_df_pmcat

def refine_pmcat_data(pmcat_training_data):
    for index, row in pmcat_training_data.iterrows():
        if len(mcat_data.loc[(mcat_data['name'] == row.PRIME_MCAT_NAME) & (mcat_data['catid'] == int(subcat_name))]) > 0:
            next
        else:
            pmcat_training_data.drop(index, inplace=True)
    return pmcat_training_data

def get_child_data(df):
    df_child_data_1 = df[(df.FK_MCAT_TYPE_ID != 1) & (df.FK_MCAT_TYPE_ID != 2)]
#     df_child_data_3 = df[df.FK_MCAT_TYPE_ID.isna()]
    df_child_data = pd.concat([df_child_data_1])
    return df_child_data

def get_childData_groups(df_child_data):
    df_child_groups = df_child_data.groupby(["PMCAT_NAME"])
    return df_child_groups

def refined_child_groups(df_child_data):
    count=1
    df_child_groups = get_childData_groups(df_child_data)
    for name,group in df_child_groups:
        count = count+1
#         if (len(mcat_data.loc[(mcat_data['name'] == name) & (mcat_data['catid'] == int(subcat_name))]) > 0) and (len(df_child_data.loc[(df_child_data['PMCAT_NAME'] == name) & ((df_child_data['PMCAT_MCAT_TYPE'] == 2))]) > 0):
        if (len(mcat_data.loc[(mcat_data['name'] == name) & (mcat_data['catid'] == int(subcat_name))]) > 0):
            if (len(df_child_data.loc[(df_child_data['PMCAT_NAME'] == name) & (df_child_data['PMCAT_MCAT_TYPE'] == 2)]) > 0):
                next
            elif (len(df_child_data.loc[(df_child_data['PMCAT_NAME'] == name) & (df_child_data['PMCAT_MCAT_TYPE'] == 3)]) > 0):
                next
            else:
                df_child_data = df_child_data.drop(df_child_groups.get_group(name).index)
        else:
            df_child_data = df_child_data.drop(df_child_groups.get_group(name).index)
       
    df_child_groups = get_childData_groups(df_child_data)

    return df_child_groups

def remove_newline_character(df):
    df = df.replace('\\n','', regex=True)
    df = df.replace('\n','', regex=True)
    return df

def write_dataframe_to_csv(base_directory, file_name, df):
    df.to_csv(base_directory + file_name, index=False, sep='\t', encoding='utf-8')



# In[ ]:


# df = preProcess("./FastText_POC/input_155/product_data_excavator_155_1.csv")
df_complete = readCsvOrExcel(input_file_name)


# In[ ]:


df = df_complete.get("subcat_"+subcat_name)


# In[ ]:


search_data = pd.read_csv("/home/divyansh/Downloads/subcat_155_title_search - Sheet2.tsv", sep="\t")


# In[ ]:


df["SEARCH_TOPMOST_MCAT"] = search_data['SEARCH_TOPMOST_MCAT']


# In[ ]:


display(df)


# In[ ]:


len(df)


# In[ ]:


# df = df[df.MAP_SUBCAT_IDLIST == str(","+subcat_name+",")]

pmcat_training_data = getPmcatData(df)
len(pmcat_training_data)


# In[ ]:


display(pmcat_training_data)


# In[ ]:


pmcat_training_data = refine_pmcat_data(pmcat_training_data)
len(pmcat_training_data)


# In[ ]:


list(pmcat_training_data.PRIME_MCAT_NAME.unique())


# In[ ]:


pmcat_training_data = remove_newline_character(pmcat_training_data)


# In[ ]:


base_directory = "./FastText_Waterfall_Model/subcat_"+str(subcat_name)+"/pmcat/input_"+str(subcat_name)+"/"
write_dataframe_to_csv(base_directory, str("pmcat_data_"+ subcat_name +".csv"), pmcat_training_data)
# pmcat_training_data.to_csv(base_directory + "pmcat_data_"+ subcat_name +".csv", index=False,header=False, sep='\t', encoding='utf-8')


# In[ ]:


base_directory = "./FastText_Waterfall_Model/subcat_"+str(subcat_name)+"/pmcat/output_"+str(subcat_name)+"/"
df_cleaned = clean_and_preProcess_data(pmcat_training_data)
df_cleaned_updated = df_cleaned.FASTTEXT_CORPUS.dropna()
df_cleaned_updated.to_csv(base_directory+str("pmcat_data_"+ subcat_name +".txt"), index=None, header=False)


# In[ ]:


df_child_data = get_child_data(df)


# In[ ]:


df_child_data.head()
print(len(df_child_data))


# In[ ]:


df_child_groups = refined_child_groups(df_child_data)


# In[ ]:


print(df_child_groups.PMCAT_NAME.count())


# In[ ]:


base_directory = "./FastText_Waterfall_Model/subcat_"+str(subcat_name)+"/child/input_"+str(subcat_name)+"/"
out_directory = "./FastText_Waterfall_Model/subcat_"+str(subcat_name)+"/child/output_"+str(subcat_name)+"/"
for name, group in df_child_groups:
    group = remove_newline_character(group)
    group_fields = group[['TITLE','SMALL_DESC','PRIME_MCAT_NAME','FK_MCAT_TYPE_ID','PRIME_SUBCAT_ID','ISQ_DETAILS4_INDEX','MAP_MCAT_NAMELIST','SEARCH_TOPMOST_MCAT']].copy()
    
    write_dataframe_to_csv(base_directory, str(str(name.replace(" ","_"))+".csv"), group_fields)
    
    df_cleaned = clean_and_preProcess_data(group_fields)
    df_cleaned_updated = df_cleaned.FASTTEXT_CORPUS.dropna()
    df_cleaned_updated.to_csv(out_directory+str(str(name.replace(" ","_"))+".txt"), index=None, header=False)

#     group_fields.to_csv("./FastText_Waterfall_Model/subcat_155/child/input_155/"+str(name.replace(" ","_"))+".csv", index=False, sep='\t', encoding='utf-8')


# In[ ]:


combined_parsed_file_name="combined.txt"
combined_shuffled_file_name="combined_shuffled.txt"
train_file_name="singleLabel_train_"+subcat_name
test_file_name="single_label_test_"+subcat_name
model_name="supervised_classifier_model_CATID_"+subcat_name
split=.9


# In[ ]:


def generate_files_and_trainModel(modelType, modelName):
    
    base_directory = "./FastText_Waterfall_Model/subcat_"+subcat_name+"/"
    input_path= base_directory+modelType+"/"+"input_"+subcat_name+"/"
    output_path= base_directory+modelType+"/"+"output_"+subcat_name+"/"
    KFold_Validation_Files_path= base_directory+modelType+"/"+"kFold_validation_"+subcat_name+"/"
    model_path=base_directory+modelType+"/"+"model_"+subcat_name+"/"
    
#     file = input_path+modelName+".csv"
#     print("Parsing File:"+file)
#     parsed_items=preProcess(file)
#     write_to_file(parsed_items,output_path,str(str(modelName)+".txt"),True)
#     print("Total Lines Parsed ="+str(len(parsed_items))) 
#     print("File Parsing Completed")
    
    print("Shuffling File..")
    total_data=randomly_shuffle_file(output_path+str(str(modelName)+".txt"),output_path+modelName+"_shuffled.txt")
    print("Total Data Len"+str(len(total_data)))
    
    if modelType == "child":
        split = 1.0
    else:
        split = 0.9
        
    print("Splitting Files..")
    train_data=total_data[0:int(len(total_data)*split)]
    print("Train Data Len"+str(len(train_data)))
    test_data =total_data[int(len(total_data)*split):len(total_data)-1]
    print("Test Data Len"+str(len(test_data)))
    
    if modelType == "child":
        
        print("Saving Files...")
        write_to_file(train_data,output_path,modelName,False)
#         write_to_file(test_data,output_path,test_file_name,False)
    
    print(total_data[0:10])
    
    if modelType == "pmcat":
        from sklearn.model_selection import KFold # import KFold
        import numpy as np
        
        kf = KFold(n_splits=10) # Define the split - into 2 folds 
        kf.get_n_splits(total_data) # returns the number of splitting iterations in the cross-validator
        print(kf) 
        KFold(n_splits=10, random_state=None, shuffle=False)
        file_number = 1
        precision=0
        recall=0
        for train_index, test_index in kf.split(total_data):
            train_data = np.array(total_data)[train_index]
            test_data = np.array(total_data)[test_index]
            write_to_file(train_data,KFold_Validation_Files_path,train_file_name+"_"+str(file_number)+".txt",False)
            write_to_file(test_data,KFold_Validation_Files_path,test_file_name+"_"+str(file_number)+".txt",False)
            
            print("Iteration:- " + str(file_number))
            file_number=file_number+1
        
        i=1
        final_precision=0
        final_recall=0
        while i < 11:
            classifier = fasttext.supervised(KFold_Validation_Files_path+train_file_name+"_"+str(i)+".txt",
                                             model_path+model_name+"_"+str(i), label_prefix='__label__',
                                             epoch=50,lr=1.0,word_ngrams=1,bucket=200000,dim=50,loss='hs')
    #     ,lr_update_rate=100,thread=4,minn=4
        
            classifier_test_result=classifier.test(KFold_Validation_Files_path+test_file_name+"_"+str(i)+".txt")
            precision=classifier_test_result.precision
            recall=classifier_test_result.recall
            final_precision=final_precision+precision
            final_recall=final_recall+recall
            print("Precision "+ str(i)+ " is:   " + str(precision))
            print("Recall "+ str(i)+ " is:   " + str(recall))
        #     print("Iteration:- " + str(i))
            i=i+1
    
        print("Average Precision is:-" + str(final_precision/10))
        print("Average Recall is:-" + str(final_recall/10))
        
        pmcat_classifier = fasttext.supervised(output_path+modelName+"_shuffled.txt",
                                             model_path+model_name+"_pmcat_data", label_prefix='__label__',
                                             epoch=50,lr=1.0,word_ngrams=1,bucket=200000,dim=50,loss='hs')
    
        def getTestSetForValidation(file_name,no_of_cases):
            texts=[]
            num_tests=int(no_of_cases)
            with open(file_name) as infile:
                num_line=0
                for line in infile:
                    num_line=num_line+1
                    texts.append(line)
                    if num_line >num_tests:
                        break
            return texts
        
        texts = getTestSetForValidation(KFold_Validation_Files_path+test_file_name+"_1.txt",1000)
        
        classifier = fasttext.load_model(model_path+model_name+"_1.bin")
        
        predicted_labels = classifier.predict_proba(texts,k=1)
        i=0
        for labels in predicted_labels:
            count=0
            lab=""
            while count < len(labels):
                if lab=="":
                    lab = str(labels[count]).replace("('","").replace("'","").replace(")","")
                else:
                    lab = str(lab)+"="+str(labels[count]).replace("('","").replace("'","").replace(")","")
                count +=1
                lab = str(lab).replace("__label__","")
            finalString = str(lab).split(",")[0].replace("_"," ")+"\t"+str(lab).split(",")[1]+"\t"+str(texts[i]).split(" ")[0].replace("__label__","").replace("_"," ")+"\t"+ " ".join(str(texts[i]).split(" ")[1:])
#             finalString =  str(" ".join(str(texts[i]).split(" ")[1:]))+"\t"+str(texts[i]).split(" ")[0].replace("__label__","").replace("_"," ")+"\t"+str(lab).split(",")[0].replace("_"," ")+"\t"+str(lab).split(",")[1]
            print(finalString)
        #     ML_Single_label.write(str(lab)+"?"+str(texts[i]))
            i=i+1
    else:
        classifier = fasttext.supervised(output_path+modelName,
                                             model_path+model_name+"_"+modelName, label_prefix='__label__',
                                             epoch=50,lr=1.0,word_ngrams=1,bucket=200000,dim=50,loss='hs')
        print("Model Successfully Loaded for model name:- " + modelName)


# In[ ]:


subcat_name = "155"


# In[ ]:


generate_files_and_trainModel("pmcat","pmcat_data_"+str(subcat_name)+"")


# In[ ]:


type(df_child_groups.groups.keys())
for key in df_child_groups.groups.keys():
    print(key)


# In[ ]:


def get_file_names_to_parse(input_path):
    files_to_parse = [str(f).replace(".txt","") for f in listdir(input_path) if isfile(join(input_path, f))]
    print("Total Child Models to load="+str(len(files_to_parse)))
    return files_to_parse


# In[ ]:


childModelsName = get_file_names_to_parse("FastText_Waterfall_Model/subcat_155/child/output_155/")
childModelsName


# In[ ]:


i = 0
while i < len(childModelsName):
    generate_files_and_trainModel("child",str(childModelsName[i]))
    i+=1


# In[ ]:


# for key in df_child_groups.groups.keys():
#     generate_files_and_trainModel("child",key.replace(" ","_"))


# In[ ]:


# df.TITLE.to_csv("./subcat_title_updated_products.csv", index=False, header=None)

