# Not another neural style implementation!

Recreate photos in in the style of famous artists with machine learning.
CSCI3341 Artificial Intelligence Final Project.

## Usage

Make sure to download models with the `util/download_models.sh` script first. Thanks @jcjohnson!

    python art.py --help

## Examples

The best [Gassongram](https://www.instagram.com/explore/tags/gassongram/) ever:

## Where do I get all this art?

`util/wikiart-scraper.py` contains a neat script which will automatically scrape **every** painting for a given artist from [wikiart](http://wikiart.org). Simply specify the URL component of the artist's name (e.g. `pablo-picasso`) in the `ARTISTS` array in the script.
