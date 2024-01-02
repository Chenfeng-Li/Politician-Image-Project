"""
Extract names of politicians and corresponding image urls from the following three Wikimedia pages:

1. Category:21st-century female politicians of the United States
https://commons.wikimedia.org/w/index.php?title=Category:21st-century_male_politicians_of_the_United_States&oldid=746862837

2. Category:21st-century female politicians of the United States
https://commons.wikimedia.org/w/index.php?title=Category:21st-century_female_politicians_of_the_United_States&oldid=522514415

3. Category: 21st-century businesspeople from the United States
https://commons.wikimedia.org/w/index.php?title=Category:21st-century_businesspeople_from_the_United_States&oldid=527515279


and two Wikipedia page:

4. Category:21st-century American politicians
https://en.wikipedia.org/w/index.php?title=Category:21st-century_American_politicians&oldid=1015022478

5. Category:21st-century American businesspeople
https://en.wikipedia.org/w/index.php?title=Category:21st-century_American_businesspeople&oldid=1110690935
"""

import torch
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from tools import get_ith_a_tag_from_id



def all_page_media(url, element_id='mw-subcategories', isFirst=False, first_range=[1,-1], middle_range=[2,-2], last_range=[1,-1]):
    """
    The super-category of Wikimedia page has several pages of "subcategory", where each page contains several links to personal pages.
    This function recursively tranverses all pages of "subcategory",
    and in each page, extracts all urls of personal pages.
    
    isFirst (bool): Whether the current page is the first page of "subcategory".
    first_range: range of <a> objects to extract in the first page.
    middle_range: range of <a> objects to extract in the pages except first and last ones.
    last_range: range of <a> objects to extract in the last page.
    """
    urls_media.append(url)
    # Last Page
    if get_ith_a_tag_from_id(0, url, element_id).get_text()!='next page':
        l, r = last_range
        people_media.extend(get_ith_a_tag_from_id("all", url, element_id)[l:r])
        return
    # First Page
    if isFirst:
        l, r = first_range
        people_media.extend(get_ith_a_tag_from_id("all", url, element_id)[l:r])

    
    # Pages other than first or last
    else:
        l, r = middle_range
        people_media.extend(get_ith_a_tag_from_id("all", url, element_id)[l:r])
    
    url = "https://commons.wikimedia.org/" + get_ith_a_tag_from_id(0, url, element_id)['href']
    all_page_media(url, middle_range=middle_range, last_range=last_range)
    


def all_page_pedia(url, element_id='mw-pages', isFirst=False, first_range=[-1,1], middle_range=[2,-2], last_range=[1,-1]):
    """
    The super-category of Wikipedia has several pages of "subcategory", where each page contains several links to personal pages.
    This function recursively tranverses all pages of "subcategory",
    and in each page, extracts all urls of personal pages.
    
    isFirst (bool): Whether the current page is the first page of "subcategory".
    first_range: range of <a> objects to extract in the first page.
    middle_range: range of <a> objects to extract in the pages except first and last ones.
    last_range: range of <a> objects to extract in the last page.
    """

    urls_pedia.append(url)
    # Last Page
    if get_ith_a_tag_from_id(0, url, element_id).get_text()!='next page':
        l, r = last_range
        people_pedia.extend(get_ith_a_tag_from_id("all", url, element_id)[l:r])
        return
    # First Page
    if isFirst:
        l, r = first_range
        people_pedia.extend(get_ith_a_tag_from_id("all", url, element_id)[l:r])
    
    # Pages other than first or last
    else:
        l, r = middle_range
        people_pedia.extend(get_ith_a_tag_from_id("all", url, element_id)[l:r])
    
    url = "https://en.wikipedia.org/" + get_ith_a_tag_from_id(0, url, element_id)['href']
    all_page_pedia(url, middle_range=middle_range, last_range=last_range)




def person_img_media(url, name=None, original_image=False, depth=0):
    """
    Get the name and all images from a person's Wikimedia page.
    original_image (bool): True if get original images, but will be much slower.
                           False if get preview images, fast but with low resolution.
    depth (int): Maximum depth of sub-category to reach.
    """
    
    # Some politicians are famous with deep sub-sub-...subcategory. 
    # (e.g. Joe Biden https://commons.wikimedia.org/wiki/Category:Joe_Biden)
    # We don't need so many images, so stop when the depth of subcategory is 3.
    if depth == 3:
        return    
    
    url = "https://commons.wikimedia.org/" + url
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    

    # Initialize the dictionary key
    if name is None:
        name = soup.find(class_="mw-page-title-main").get_text()
        if name in name_img_url.keys():
            return
        name_img_url[name] = []
        subcategory_name.clear()
    
    # Get all image urls from current page
    imgs = soup.find_all(class_="mw-file-description")
    # Remark: First image in the imgs is the standard one
    
    for img in imgs:
        if original_image:
            img_url = "https://commons.wikimedia.org/" + img['href']
            response_img = requests.get(img_url)
            soup_img = BeautifulSoup(response_img.content, 'html.parser')
            original_img_url = soup_img.find(id="file").find('img')['src']
            name_img_url[name].append(original_img_url)
        else:
            name_img_url[name].append(img.find('img')['src'])
            
    # Recursively get images from subpages
    subpages = soup.find(id="mw-pages")
    if subpages:
        sps = subpages.find_all('a')
        for sp in sps:
            sp_url = sp['href']
            person_img_media(sp_url, name, original_image, depth+1)
    
    # Recursively get images from subcategories
    subcategories = soup.find(id="mw-subcategories")
    if not subcategories:
        return
    scs = subcategories.find_all('a')
    for sc in scs:
        # Avoid circular subcategories (A -> B -> A ...).
        # (e.g. Rocky Anderson https://commons.wikimedia.org/wiki/Category:Rocky_Anderson)
        if sc in subcategory_name:
            continue
        else:
            subcategory_name.add(sc)
        sc_url = sc['href']
        person_img_media(sc_url, name, original_image, depth+1)


def person_img_pedia(url):
    """
    Get the name and portrait from the Wikipedia page.
    First check if this person has wikimedia page, and get images from wikimedia page if possible.
    Otherwise, take the portrait link from the Wikipedia page if there is.
    """
    
    url = "https://en.wikipedia.org/" + url
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    url = urllib.parse.unquote(url) # convert, e.g. "%C3%A9" to "é", in url
    name_from_url = " ".join(url.split("/")[-1].split("_"))
    
    # Name
    try:
        name_from_page = soup.find(class_="mw-page-title-main").get_text()
    except:
        # Inconsistant title, very rare
        # e.g. https://en.wikipedia.org/wiki/Nick_Wilson_(Survivor_contestant)
        return
    
    if name_from_url != name_from_page:
        # The page has been redirected
        # e.g. https://en.wikipedia.org/wiki/Becky_Ames         
        return
    
    name = name_from_url
    # Omit if name_img_url contains the images
    if name in name_img_url.keys():
        return

    # Check if this person has wikimedia page
    media_url = "wiki/Category:" + url.split('/')[-1]
    person_img_media(media_url, original_image=True) # This will initialize name_img_url[name] = []
    if name_img_url[name]: # non-empty:
        return
    
    # Get portrait from the person's infobox
    infobox = soup.find(class_="infobox")
    
    if infobox:
        img = infobox.find(class_="mw-file-description")
        if img:
            img_url = "https://en.wikipedia.org/" + img['href']
            response_img = requests.get(img_url)
            soup_img = BeautifulSoup(response_img.content, 'html.parser')
            original_img_url = "https:" + soup_img.find(id="file").find('img')['src']
            name_img_url[name].append(original_img_url)
            




if __name__ == "__main__":
    # Map: name -> list of image urls
    try:
        name_img_url = torch.load("name_img_url.pt")
    except:
        name_img_url = {}

    # Wikimedia
    # Urls of page of "subcategory"
    urls_media = [] 
    # Urls of person pages.
    people_media = []

    # Get personal pages from the two websites.
    url_male = "https://commons.wikimedia.org/w/index.php?title=Category:21st-century_male_politicians_of_the_United_States&oldid=746862837"
    all_page_media(url_male, isFirst=True)
    url_female = "https://commons.wikimedia.org/w/index.php?title=Category:21st-century_female_politicians_of_the_United_States&oldid=522514415"
    all_page_media(url_female, isFirst=True)
    url_businesspeople = "https://commons.wikimedia.org/w/index.php?title=Category:21st-century_businesspeople_from_the_United_States&oldid=527515279"
    all_page_media(url_businesspeople, isFirst=True)

    # Extract all images urls for each politician.
    subcategory_name = set()

    # This takes around 24 hours.
    for person in tqdm(people_media):
        person_url = person['href']
        person_img_media(person_url, original_image=True)
        torch.save(name_img_url,"name_img_url.pt")

        
    # Wikipedia
    urls_pedia = [] 
    people_pedia = []
        
    url_politicians = "https://en.wikipedia.org/w/index.php?title=Category:21st-century_American_politicians&oldid=1015022478"
    all_page_pedia(url_politicians, isFirst=True, first_range=[3,-1], middle_range=[3,-2], last_range=[2,-1])
    url_businesspeople = "https://en.wikipedia.org/w/index.php?title=Category:21st-century_American_businesspeople&oldid=1110690935"
    all_page_pedia(url_businesspeople, isFirst=True, first_range=[4,-1], middle_range=[3,-2], last_range=[2,-1])

    # This takes around 24 hours.
    for person in tqdm(people_pedia):
        person_url = person['href']
        person_img_pedia(person_url)
        torch.save(name_img_url,"name_img_url.pt")

    # Save the name_img_url dictionary
    # Size around 60.9 MB
    torch.save(name_img_url,"name_img_url.pt")