"""
Automatically scrape paintings by specified artists from wikiart.

Jesse Mu
"""

import requests
from bs4 import BeautifulSoup
import os
from collections import defaultdict
from fake_useragent import UserAgent  # Hehe

# Global constants
ARTIST_URL = 'http://wikiart.org/en/{artist}/mode/all-paintings/{page}'
IMG_URL = 'http://uploads4.wikiart.org/images/{artist}/{painting}/jpg'
ARTISTS = ['vincent-van-gogh']
FAKE_HEADERS = {'User-Agent': UserAgent().random}
IMG_DIR = '../wikiart/{artist}'
BASE_DIR = IMG_DIR.format(artist='')


def total_pages(soup):
    """
    Given an artist's parsed BeautifulSoup page, determine how many
    pages there are (there's a predictable pattern).
    """
    pager_items = soup.find('div', {'class': 'pager-items'})
    pager_links = pager_items.find_all('a')
    for pager, next_pager in zip(pager_links, pager_links[1:]):
        # There's always a "next" link before the last page
        if next_pager.text == 'Next':
            return int(pager.text)

    # If here, we haven't found a last page
    canonical = soup.find('link', {'rel': 'canonical'})['href']
    raise ValueError("Couldn't find last page for {}".format(canonical))


def raise_if_bad(request_obj, url='undef'):
    """Throw a helpful error message when a request fails."""
    try:
        request_obj.raise_for_status()
    except requests.exceptions.HTTPError as e:
            print "wikiart-scraper.py: Error trying to retrieve {}".format(
                request_obj.url
            )
            raise e


def clean_painting_url(painting_url):
    """
    Clean the painting url by removing the size specification.

    Might be other things later.
    """
    splitted = painting_url.split('!')
    assert len(splitted) == 2, 'url {} has more than one !'.format(
        painting_url
    )
    return splitted[0]


def save_painting(link, directory):
    """
    Actually request the url and save the painting into the directory.
    """
    r_img = requests.get(link, stream=True)
    raise_if_bad(r_img, link)

    # Get name by splitting on slash and getting the last element
    img_name = link.split('/')[-1]
    print u"Saving img {} in directory {}".format(img_name, directory + '/')
    with open(directory + '/' + img_name, 'wb') as fout:
        fout.write(r_img.content)


def scrape_paintings(soup, directory=BASE_DIR):
    """
    Scrape the given artist page and save images into the specified
    directory.

    ADDITIONALLY, return a list of names of paintings scraped.
    """
    # Make artist directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Class "Paintings" contains all links
    soup_paintings = soup.find('div', {'class': 'Painting'})
    # Just get all links - they should all be paintings
    soup_imgs = [s['src'] for s in soup_paintings.find_all('img')]
    # But double check just to make sure
    cleaned_links = [clean_painting_url(s) for s in soup_imgs
                     if 'jpg' in s and 'uploads' in s]
    for link in cleaned_links:
        save_painting(link, directory)

    return cleaned_links


def main():
    # Create the img directory if it doesn't exist.
    if not os.getcwd().endswith('util'):
        print "You ought to run this script from the util directory for " + \
              "accurate save locations (see IMG_DIR)"
        return
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    artist_paintings = defaultdict(list)
    for artist in ARTISTS:
        r_artist = requests.get(
            ARTIST_URL.format(artist=artist, page=1), headers=FAKE_HEADERS
        )
        raise_if_bad(r_artist, url=r_artist.url)
        soup = BeautifulSoup(r_artist.text, 'lxml')  # Default to lmxl parser
        for i in xrange(1, total_pages(soup)):
            r_artist_page = requests.get(
                ARTIST_URL.format(artist=artist, page=i),
                headers=FAKE_HEADERS
            )
            raise_if_bad(r_artist_page, url=r_artist_page.url)
            soup_page = BeautifulSoup(r_artist_page.text, 'lxml')

            # Download the paintings!
            paintings = scrape_paintings(
                soup_page,
                directory=IMG_DIR.format(artist=artist)
            )

            # Add paintings to the dictionary
            artist_paintings[artist].extend(paintings)

if __name__ == '__main__':
    main()
