import requests 
from bs4 import BeautifulSoup 
# providing url 
url = "https://archiveofourown.org/works?work_search%5Bsort_column%5D=hits&include_work_search%5Brating_ids%5D%5B%5D=10&work_search%5Bother_tag_names%5D=&work_search%5Bexcluded_tag_names%5D=&work_search%5Bcrossover%5D=&work_search%5Bcomplete%5D=&work_search%5Bwords_from%5D=&work_search%5Bwords_to%5D=&work_search%5Bdate_from%5D=&work_search%5Bdate_to%5D=&work_search%5Bquery%5D=&work_search%5Blanguage_id%5D=&commit=Sort+and+Filter&tag_id=Harry+Potter+-+J*d*+K*d*+Rowling"
  
# creating requests object 
html = requests.get(url).content 
  
# creating soup object 
data = BeautifulSoup(html, 'html.parser') 

ol=data.find("ol", {"class": "work index group"})
l=[]
litags=ol.find_all('li',recursive=False)
for x in litags:
    l.append(x.get('id'))

# for litag in ol.find_all('li',{"class": "work blurb group"}):
#     l.append(litag.find('li'))
# print(l[0:3])