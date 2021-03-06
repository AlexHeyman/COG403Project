from bs4 import BeautifulSoup
import urllib3
#import urllib
#import urllib.parse
import re
import sys

if len(sys.argv) < 3:
    sys.exit('Usage: %s [media category] [min size] (e.g.: %s "TV Shows" 500 )' % (sys.argv[0],sys.argv[0]))

threshold = int(sys.argv[2])
category = re.sub(r' ', r'%20', sys.argv[1])
category = re.sub(r'&', r'*a*', category)

#url = "http://archiveofourown.org/media/TV%20Shows/fandoms"
url = "http://archiveofourown.org/media/" + category + "/fandoms"

try:
    http = urllib3.PoolManager()
    r = http.request('GET', url)
except:
    print("Bad URL: " + url)

soup = BeautifulSoup(r.data)

soup.prettify()

fandoms = []

showinfo = soup.find_all('li')
#print str(showinfo[0])
for si in showinfo:
#    matchObj = re.match( r'<a.*>(.*)</a> 
    matchObj = re.search( r'<a class=\"tag\".*>(.*)</a>.*\(([0-9]+)\)', str(si), re.I|re.S)
    if matchObj:
        fandom = matchObj.group(1)
        if not re.search(r'All Media Types', fandom, re.I):
            if not re.search(r'Related Fandoms', fandom, re.I):
                numworks = int(matchObj.group(2))
                if numworks >= threshold:
                    fandoms.append((fandom, numworks))
#        print matchObj.group(1)
#        print matchObj.group(2)

fandoms.sort(key=lambda x: -x[1])
for i in range(len(fandoms)):
    fandom, numworks = fandoms[i]
    print(str(i + 1) + ". " + fandom + ", " + str(numworks))

#showtags = soup.find_all('a', class_="tag")
#showtags = soup.find_all('a', class_="tag")
#print shows[0]

#for showtag in showtags:
#    show = showtag.getText()
#    print show
