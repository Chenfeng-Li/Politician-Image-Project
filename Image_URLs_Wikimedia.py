"""
Extract names of politicians and corresponding image urls from the following two Wikimedia pages:

1. Category:21st-century female politicians of the United States
https://commons.wikimedia.org/w/index.php?title=Category:21st-century_male_politicians_of_the_United_States&oldid=746862837

2. Category:21st-century female politicians of the United States
https://commons.wikimedia.org/w/index.php?title=Category:21st-century_female_politicians_of_the_United_States&oldid=522514415
"""

import torch
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from tools import get_ith_a_tag_from_id



def all_page_media(url, element_id='mw-subcategories', isFirst=False):
    """
    The two website has several pages of "subcategory", where each page contains several links to personal pages.
    
    This function recursively tranverses all pages of "subcategory",
    and in each page, extracts all urls of personal pages.
    """
    urls_media.append(url)
    if get_ith_a_tag_from_id(0, url, element_id).get_text()!='next page':
        people_media.extend(get_ith_a_tag_from_id("all", url, element_id)[2:-1])
        return
    if isFirst:
        people_media.extend(get_ith_a_tag_from_id("all", url, element_id)[1:-1])
        
    else:
        people_media.extend(get_ith_a_tag_from_id("all", url, element_id)[2:-2])
    url = "https://commons.wikimedia.org/" + get_ith_a_tag_from_id(0, url, element_id)['href']
    all_page_media(url)




def person_img_media(url, name=None, original_image=False, depth=0):
    """
    Get the name and all images from a person's Wikimedia page.
    original_image (bool): True if get original images, but will be much slower.
                           False if get preview images, fast but with low resolution.
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
        name_img_media[name] = []
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
            name_img_media[name].append(original_img_url)
        else:
            name_img_media[name].append(img.find('img')['src'])
            
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




if __name__ == "__main__":
    # Urls of page of "subcategory"
    urls_media = [] 
    # Urls of person pages.
    people_media = []

    # Get personal pages from the two websites.
    url_male = "https://commons.wikimedia.org/w/index.php?title=Category:21st-century_male_politicians_of_the_United_States&oldid=746862837"
    all_page_media(url_male, isFirst=True)
    url_female = "https://commons.wikimedia.org/w/index.php?title=Category:21st-century_female_politicians_of_the_United_States&oldid=522514415"
    all_page_media(url_female, isFirst=True)

    # Extract all images urls for each politician.
    # Map: name -> list of image urls
    name_img_media = {}
    subcategory_name = set()

    # This takes around 24 hours.
    for person in tqdm(people_media):
        person_url = person['href']
        person_img_media(person_url, original_image=True)

    # Save the name_img_media dictionary
    # Size around 32 MB
    torch.save(name_img_media,"name_img_media.pt")