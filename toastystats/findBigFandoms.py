from bs4 import BeautifulSoup
import urllib3
#import urllib
#import urllib.parse
import re
import sys

# basic error checking for right number of arguments
if len(sys.argv) < 3:
    sys.exit('Usage: %s [media category] [min num works] (e.g.: %s "TV Shows" 500 )\n Possible media categories listed here: http://archiveofourown.org/media' % (sys.argv[0],sys.argv[0]))

# reformat the media type string into the AO3 URL format 
threshold = int(sys.argv[2])
category = re.sub(r' ', r'%20', sys.argv[1])
category = re.sub(r'&', r'*a*', category)
url = "http://archiveofourown.org/media/" + category + "/fandoms"

# get the web page listing the fandoms for this Media Type category
try:
    http = urllib3.PoolManager()
    r = http.request('GET', url)
except:
    print("Bad URL: " + url)
soup = BeautifulSoup(r.data)
soup.prettify()

fandoms = []

# grab all the fandoms and numbers of fanworks
fandominfo = soup.find_all('li')
for fi in fandominfo:
    # separate out the fandom name and the number of fanworks
    matchObj = re.search( r'<a class=\"tag\".*>(.*)</a>.*\(([0-9]+)\)', str(fi), re.I|re.S)
    if matchObj:
        fandom = matchObj.group(1)
        numworks = int(matchObj.group(2))
        # print out the fandom and num works if they pass the threshold
        if numworks >= threshold:
            #reformat special characters in fandom name
            #fandom = re.sub(r'&amp;', r'&', fandom)
            fandoms.append((fandom, numworks))

fandoms.sort(key=lambda x: -x[1])
for i in range(len(fandoms)):
    fandom, numworks = fandoms[i]
    print(str(i + 1) + ". " + fandom + ", " + str(numworks))
