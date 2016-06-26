# Not another neural style implementation!

Recreate photos in in the style of famous artists with machine learning.
CSCI3341 Artificial Intelligence Final Project.

## Usage

Make sure to download models with the `util/download_models.sh` script first. Thanks @jcjohnson!

    python art.py --help

## Examples

Create a (low quality) version of the image below. The first positional
argument is the content image and the image(s) after are the style
images. The further options instruct the algorithm to run for 200 iterations
and output an image with a 256px maximum width, which should be doable
relatively quickly on a personal computer.

    python art.py examples/gasson.jpg examples/starry.jpg -n 200 -w 256

Further options include using the entire oeuvre of an artist as a stylistic
target (which never works well), scaling the stylistic features by a factor,
specifying whether to initialize the candidate image from random noise or the
content image, and more. See the script's help for details.

The best [Gassongram](https://www.instagram.com/explore/tags/gassongram/) ever:

![gasson](https://raw.githubusercontent.com/jayelm/neural-art/master/examples/gasson.jpg)
![starry](https://raw.githubusercontent.com/jayelm/neural-art/master/examples/starry.jpg)
![gasson_final](https://raw.githubusercontent.com/jayelm/neural-art/master/examples/gasson_final.jpg)

## Where do I get all this art?

`util/wikiart-scraper.py` contains a neat script which will automatically scrape **every** painting for a given artist from [wikiart](http://wikiart.org). Simply specify the URL component of the artist's name (e.g. `pablo-picasso`) in the `ARTISTS` array in the script.
