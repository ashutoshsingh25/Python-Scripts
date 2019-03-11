# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 10:07:07 2018

@author: prachi
"""

from selenium import webdriver
import time as t
import pandas as pd

#Loading additional functionalities for running the web browser

'''  For Google Chrome browser'''
options = webdriver.ChromeOptions()

''' For Firefox browser'''
#options = webdriver.FirefoxOptions()

options.add_argument('--ignore-certificate-errors')
options.add_argument("--test-type")
#options.add_argument("--headless")

#Reading the input file having video urls

df = pd.read_excel('C:/Users/prachi/Downloads/Video POC MCAT.xlsx')
listofmcats = df['MCAT NAME'].tolist()
finaldf = pd.DataFrame(columns = ['Keyword','VideoURL','Title','Description','Number of views'])
row = 0
for mcat in listofmcats:
    keyword = mcat
    
    
    
    
    driver = webdriver.Chrome(executable_path='C:/Users/prachi/Downloads/chromedriver.exe',chrome_options=options)
    
    
    #Go to the youtube page
    driver.get('http://www.youtube.com')
    
    #Search for the keywords in the youtube page
    driver.find_element_by_id('search').send_keys(keyword)
    driver.find_element_by_id('search-icon-legacy').click()
    t.sleep(3)
    
    #Store the results in a list and find the top 10 links
    listofresults=driver.find_element_by_id('contents').find_elements_by_id('video-title')
    
    listofresults = listofresults[:10]
    toptenresults = [x.get_attribute('href') for x in listofresults]
    
    
    
    '''The structure of the final output required'''
    
    
    
    
    
    
     
    
    for i in range(0,len(toptenresults)):
        
        #The top 10 links taken up one by one
        
        finallink = toptenresults[i] 
        
        
        ''' For Firefox browser'''
        #webdriver.Firefox(executable_path='C:/Users/IMART/Downloads/geckodriver-v0.21.0-win64/geckodriver.exe')
        
        driver.get(finallink)
        
        t.sleep(5)
        
        
        #Finding the title and description of youtube links
        try:
            title = driver.find_elements_by_xpath('//*[@id="container"]/h1/yt-formatted-string')[0].text
            
            
        except:
            title = "No video available"
            
        try:
            
            description = driver.find_elements_by_xpath('//*[@id="description"]/yt-formatted-string')[0].text
            
        except:
            
            description = "No video available"
            
        
            
        finally:
            views = driver.find_element_by_class_name('view-count').text
            finaldf.set_value(row,'Keyword',keyword)
            finaldf.set_value(row,'VideoURL',toptenresults[i])
            finaldf.set_value(row,'Title',title)
            finaldf.set_value(row,'Description',description)
            finaldf.set_value(row,'Number of views',views)
            row = row + 1
        
        t.sleep(5)
        
    driver.quit()
    
    


#Saving the final dataframe information into a csv file
finaldf.to_csv("YoutubeVideosTitleDescriptionViews.csv")