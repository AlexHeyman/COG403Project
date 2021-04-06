These scripts find the current largest fandoms on AO3, broken down by
the categories that AO3 divides them into.  They also output the
number of fanworks in each fandom.  For a list of all media
categories, see the headers on this page:
http://archiveofourown.org/media 

USAGE:
findBigFandoms[Filtered].py <media category> <min num fanworks>

(The "Filtered" version excludes "& Related Fandoms" and "All Media Types.")


*** Example 1a: Find TV shows on AO3 with over 50,000 fanworks:

> python findBigFandoms.py "TV Shows" 50000

*** Example 1b: Find all TV shows on AO3, excluding umbrella fandoms:

> python findBigFandomsFiltered.py "TV Shows" 0