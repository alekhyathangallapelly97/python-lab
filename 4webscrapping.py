from bs4 import BeautifulSoup
import requests

url="https://catalog.umkc.edu/course-offerings/graduate/comp-sci/"
html_content = requests.get(url).text
soup = BeautifulSoup(html_content,"html.parser")
text=soup.find_all(text = True)
output = ''
blacklist = [
'[document]',
'noscript',
'header',
'html',
'meta',
'head',
'input',
'script',
]

for t in text:
 if t.parent.name not in blacklist:
  output += '{} '.format(t)
print(output)




