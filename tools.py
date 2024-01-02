import requests
from io import BytesIO
from bs4 import BeautifulSoup
from PIL import Image

def img_from_url(url):
    """
    Get image from url.
    """
    headers = {'User-Agent': 'Face recognition project (jmlichenfeng@gmail.com)'}
    response = requests.get(url, headers=headers)
    img = Image.open(BytesIO(response.content))
    return img


def get_ith_a_tag_from_id(i, url, element_id='mw-pages'):
    """
    Input an url, get i-th <a> tag (or all <a> tags) under an element id.
    i: Integer or "all". If 0, then get the last <a> tag.
    """
    # Fetch the webpage content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Find the element by ID
    element = soup.find(id=element_id)
    all_a_tags = element.find_all('a')
    if i=="all":
        return all_a_tags
    
    return all_a_tags[i-1]