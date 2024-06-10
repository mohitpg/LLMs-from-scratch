import requests 
from bs4 import BeautifulSoup 
import time

NUMBER_OF_FILES=10 #Max value 40

listurl = "https://archiveofourown.org/works?work_search%5Bsort_column%5D=hits&include_work_search%5Brating_ids%5D%5B%5D=10&work_search%5Bother_tag_names%5D=&work_search%5Bexcluded_tag_names%5D=&work_search%5Bcrossover%5D=&work_search%5Bcomplete%5D=&work_search%5Bwords_from%5D=&work_search%5Bwords_to%5D=&work_search%5Bdate_from%5D=&work_search%5Bdate_to%5D=&work_search%5Bquery%5D=&work_search%5Blanguage_id%5D=&commit=Sort+and+Filter&tag_id=Harry+Potter+-+J*d*+K*d*+Rowling" 
html = requests.get(listurl).content 
data = BeautifulSoup(html, 'html.parser') 

ol=data.find("ol", {"class": "work index group"})
work_name=[]
litags=ol.find_all('li',recursive=False)
for x in litags:
    work_name.append(x.get('id'))

baseurl="https://archiveofourown.org/"
for i in range(NUMBER_OF_FILES):   
    s=work_name[i].split('_')
    url=baseurl+'works/'+s[1]+"?view_full_work=true"
    time.sleep(5)
    html = requests.get(url).content 
    data = BeautifulSoup(html, 'html.parser') 
    chapters=data.find("div",{"id": "chapters"})
    texts = chapters.findAll(text=True)
    filename=f'data{i}.txt'
    with open(filename,'w',encoding="utf8") as f:
        f.write(''.join(texts))