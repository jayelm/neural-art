"""
Implementation of A Neural Algorithm of Artistic Style

authors: Jesse Mu, Andrew Francl
"""

# Make matplotlib not use X11 (not necessary)
import matplotlib
matplotlib.use('Agg')

import numpy as np

import caffe
# GPU mode
caffe.set_mode_gpu()

# Numerical computation
from scipy import optimize
# np.dot is too slow. Use blas.sgemm instead
from scipy.linalg import blas

# IO
import skimage
from skimage.io import imsave
from skimage import transform

# Util
from itertools import izip_longest
from datetime import datetime
import os
import glob

# Constants
VGG_MODEL = './models/VGG_ILSVRC_19_layers.caffemodel'
VGG_PROTOTXT = './models/VGG_ILSVRC_19_layers_deploy.prototxt'

MEAN_PIXEL = np.array([104.00698793, 116.66876762, 122.67891434])

SC_RATIO = 1e4

# NOTE: We use the blob name instead of the layer name according to the model
# prototxt. Layers are specified in the paper.
CONTENT_LAYERS = ['conv4_2']
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
# TODO: Make these weights configurable
CONTENT_WEIGHTS = [1. / len(CONTENT_LAYERS)] * len(CONTENT_LAYERS)
STYLE_WEIGHTS = [1. / len(STYLE_LAYERS)] * len(STYLE_LAYERS)

class Art(object):
    """
    Your one-stop shop for all things neural style
    """
    def __init__(self, net, args):
        self.net = net

        self.transformer = self.create_transformer()

        # These values will be initialized when
        # network is setup (set_style_targets, set_content_targets)
        self.style_targets = None
        self.content_target = None

        # Get reversed list of layer pairs for backprop.
        # Placed in init to save running time in optimization fn
        # Blobs is an ordereddict, so keys are in order
        reversed_layers = []
        for layer in self.net.blobs.keys():
            if layer in STYLE_LAYERS or layer in CONTENT_LAYERS:
                reversed_layers.append(layer)

        reversed_layers.reverse()

        self.reversed_pairs = list(izip_longest(reversed_layers,
                                                reversed_layers[1:]))

        self.max_width = args.width

        self.init = args.init

        self.print_rate = args.print_rate

        self.style_scale = args.style_scale

        # TODO: Get rid of this when you've kept things separate.
        self.args = args

        # Set counter for printing progress in graddesc
        self.iter = 0

    def create_transformer(self):
        """
        Create the preprocessor and deprocessor using the default settings for
        the VGG-19 network.
        """
        # Give transformer necessary imput shape. Should be specified from
        # argparse arguments when creating the net
        transformer = caffe.io.Transformer(
            {'data': self.net.blobs['data'].data.shape}
        )
        # Order of the channels in the input data (not sure why necessary)
        transformer.set_transpose('data', (2, 0, 1))
        # Use BGR rather than RGB
        transformer.set_channel_swap('data', (2, 1, 0))
        # Subtract mean pixel
        transformer.set_mean('data', MEAN_PIXEL)
        # Use 8bit image values
        transformer.set_raw_scale('data', 255)

        return transformer


    def resize_image(self, img, scale=1.0):
        """
        Resize image to self.max_width, with varying height.
        """
        assert img.shape[2] == 3

        oldwidth = float(img.shape[1])
        oldheight = float(img.shape[0])
        newheight = int(((self.max_width / oldwidth) * oldheight) * scale)
        newwidth = int(self.max_width * scale)
        return transform.resize(img, (newheight, newwidth, 3))

    def resize_caffes(self, img):
        """
        Resize the caffe net and transformer input blobs to accept the scaled
        image.
        """
        new_size = (1, img.shape[2], img.shape[0], img.shape[1])
        self.net.blobs['data'].reshape(*new_size)  # Unpack for mult args
        self.transformer.inputs['data'] = new_size

    def set_style_targets(self, imgs, weights):
        """
        Params
        ======
        imgs : List<str>
            Filename of the image to load in.
        """
        target_sl_list = []
        for sl, _ in zip(STYLE_LAYERS, STYLE_WEIGHTS):

            sl_dim0 = self.net.blobs[sl].data[0].shape[0]
            target_sl = np.zeros((sl_dim0, sl_dim0))

            for img, imgweight in zip(imgs, weights):
                # Preprocess image, load into net
                stylei = caffe.io.load_image(img)
                # Resize image, set net and transformer shapes accordingly
                scaled = self.resize_image(stylei, self.style_scale)
                self.resize_caffes(scaled)

                stylei_pp = self.transformer.preprocess('data', scaled)
                self.net.blobs['data'].data[...] = stylei_pp

                self.net.forward()

                layer = self.net.blobs[sl].data[0].copy()  # Get one batch?
                # Expand style layer to 2d array
                layer = np.reshape(layer,
                                   (layer.shape[0],
                                    layer.shape[1] * layer.shape[2])
                                  )

                gram = self._gram(layer)

                target_sl += gram * imgweight

                target_sl_list.append(gram)

        self.style_targets = target_sl_list

    def set_content_target(self, img):
        """
        Create content representation of image and set as the content target.
        """
        # XXX: Assume only one content layer
        cl = CONTENT_LAYERS[0]
        contenti = caffe.io.load_image(img)
        # Resize image, set net and transformer shapes accordingly
        scaled = self.resize_image(contenti)
        self.resize_caffes(scaled)

        contenti_pp = self.transformer.preprocess('data', scaled)
        self.net.blobs['data'].data[...] = contenti_pp
        self.net.forward()

        self.content_target = self.net.blobs[cl].data[0].copy()
        # Get contenti_pp (after transformer)
        self.content_target = (
            np.reshape(self.content_target,
                       (self.content_target.shape[0],
                        self.content_target.shape[1] * self.content_target.shape[2]))
        )

    def random_image(self):
        """
        Compute a random multicolor noise image.

        We assume that the user has called set_content_target
        because we obtain the content representation from the
        net input blob.
        """
        content_shape = self.net.blobs['data'].data.shape[1:]
        randi = (np.random.rand(*content_shape) * 255)
        return (randi.transpose() - MEAN_PIXEL).transpose()

    def _gram(self, layer):
        """
        Compute gram matrix; just the dot product of the layer and its
        transform
        """
        gram = blas.sgemm(1.0, layer, layer.T)
        return gram

    def _mse(self, A, B):
        return ((A - B) ** 2).mean()

    def style_lag(self, noisies, grams, i, compute_grad=False):
        """
        Compute style losses and gradients for all gram matrices

        This is compressed into one function to save intermediate computations.

        Is assumed that gram matrices and self.style_targets correspond to
        identical layers.
        """
        # Get everything.
        style_noisy = noisies[i]
        style_gram = grams[i]
        style_target = self.style_targets[i]
        weight = STYLE_WEIGHTS[i]

        diff = (style_gram - style_target)
        size_c = 1. / ((style_noisy.shape[0] ** 2) * (style_noisy.shape[1] ** 2))
        loss = (size_c / 4) * (diff**2).sum() * weight

        if compute_grad:
            gradient = size_c * blas.sgemm(1.0, diff, style_noisy) * (style_noisy > 0) * weight
            return loss, gradient

        return loss, None

    def content_lag(self, content_noisy, compute_grad=False):
        """
        Compute content loss and gradient.

        This is compressed into one function to save intermediate computations.
        """
        diff = (content_noisy - self.content_target)
        loss = .5 * (diff ** 2).sum()
        if compute_grad:
            gradient = diff * (content_noisy > 0)
            return loss, gradient
        return loss, None

    def loss_and_gradient(self, x):
        debug_print("Running loss and gradient")
        x_reshaped = np.reshape(x, self.net.blobs['data'].data.shape[1:])
        print x

        # Run the net on the candidate
        self.net.blobs['data'].data[...] = x_reshaped.copy()
        self.net.forward()

        content_noisy = self.net.blobs[CONTENT_LAYERS[0]].data[0].copy()
        content_noisy = np.reshape(content_noisy, (content_noisy.shape[0],
                                                   content_noisy.shape[1] * content_noisy.shape[2]))

        # COMPUTE LOSSES
        # For the first iteration, we don't care about the gradients.

        # Compute content losses.
        content_loss, _ = self.content_lag(content_noisy)
        loss = content_loss

        # Collect style layers and gram matrices
        style_noisies = map(
            lambda layer: self.net.blobs[layer].data[0].copy(),
            STYLE_LAYERS
        )
        style_reshaped = map(
            lambda n: np.reshape(n, (n.shape[0], n.shape[1] * n.shape[2])),
            style_noisies
        )
        style_grams = [self._gram(m) for m in style_reshaped]

        # Compute style losses and weight by their ratio
        total_style_loss = 0
        for i in xrange(len(STYLE_WEIGHTS)):
            total_style_loss += self.style_lag(
                style_reshaped, style_grams, i, compute_grad=False
            )[0]

        loss += total_style_loss * SC_RATIO

        # Compute backprop layer by layer to obtain gradients.
        # self.net.blobs is an ordered dict, so reversed makes sense
        # Initialize net to empty
        self.net.blobs[self.reversed_pairs[-1][0]].diff[:] = 0
        for curr, prev in self.reversed_pairs:
            # Alias this for sanity
            curr_grad = self.net.blobs[curr].diff[0]

            try:
                style_index = STYLE_LAYERS.index(curr)
            except ValueError:
                # Nope, not in style layers
                style_index = -1
            if style_index > -1:
                gradient = self.style_lag(
                    style_reshaped, style_grams, style_index, compute_grad=True
                )[1]
                curr_grad += np.reshape(gradient, curr_grad.shape) * SC_RATIO
            else:
                try:
                    content_index = CONTENT_LAYERS.index(curr)
                except ValueError:
                    # Not in style layers
                    content_index = -1
                if content_index > -1:
                    # We assume weight is 1 since we're not changing this model
                    gradient = self.content_lag(
                        content_noisy, compute_grad=True
                    )[1]
                    gradient = np.reshape(gradient, curr_grad.shape)
                    curr_grad += gradient

            # Compute the gradient
            self.net.backward(start=curr, end=prev)

        final_grad = self.net.blobs['data'].diff[0]

        # Flatten for optimization
        return loss, final_grad.flatten().astype(np.float64)

    def print_prog(self, x):
        """
        Save and print progress every self.print_rate iterations.
        """
        if (self.iter % self.print_rate) == 0:
            debug_print("gdesc iteration {}".format(str(self.iter)))
            new_img = self.transformer.deprocess(
                'data',
                x.reshape(self.net.blobs['data'].data.shape)
            )
            imsave(
                '{}/iter-{}.jpg'.format(self.dirname, self.iter),
                skimage.img_as_ubyte(new_img)
            )
        self.iter += 1

    def go(self, maxiter=512):
        """
        This is where the magic happens.

        Return the image resulting from gradient descent for maxiter
        iterations
        """
        # Init random noise image
        debug_print("Running go")
        if args.init == 'rand':
            img = self.random_image()
        else:
            default = caffe.io.load_image(self.args.content_image)
            scaled = self.resize_image(default)
            self.resize_caffes(scaled)
            img = self.transformer.preprocess('data', scaled)

        # TODO: compute bounds for gradient descent!
        debug_print("Starting grad descent")

        x, f, d = optimize.fmin_l_bfgs_b(
            self.loss_and_gradient,
            img.flatten(),
            fprime=None,  # We'll use loss_and_gradient
            maxiter=maxiter,
            callback=self.print_prog,
        )

        x = np.reshape(x, self.net.blobs['data'].data[0].shape)

        return self.transformer.deprocess('data', x)


def debug_print(msg, verbose=True):
    """
    Print msg only if verbose flag is True.
    """
    if verbose:
        print "{}: {}".format(datetime.now(), msg)


def main(args):
    """
    The main algorithm implementation function.
    """
    vgg = caffe.Net(
        VGG_PROTOTXT, VGG_MODEL, caffe.TEST,
    )

    style = Art(vgg, args)

    # Collect art from the wikiart folder
    if args.artist:
        try:
            args.style_images = glob.glob(
                './wikiart/{}/*'.format(args.artist)
            )
        except Exception as e:
            print "Couldn't get images for artist {}, check dir!".format(
                args.artist
            )
            raise e

    num_simages = len(args.style_images)

    if num_simages > 2:
        sample = args.style_images[:2]
    else:
        sample = args.style_images

    # Make the directory for this run
    raw_dirname = './img/{}__{}-w{}-{}'.format(
        args.content_image.split('.')[0],
        '_'.join(os.path.basename(s).split('.')[0] for s in sample),
        str(args.width),
        str(args.numiter)
    )

    # Get a unique dirname if the directory already exists
    dirname = raw_dirname
    unique = 1
    while os.path.exists(dirname):
        dirname = raw_dirname + '-' + str(unique)
        unique += 1

    debug_print("Starting {}".format(dirname))

    os.mkdir(dirname)

    # TODO this is bad, but whatever. save this dirname as an attr
    style.dirname = dirname

    # NOTE: For now, we assume style weights are equal
    style_image_weights = [1. / num_simages] * num_simages

    # Get style and content targets
    debug_print("Setting up content targets")
    style.set_content_target(args.content_image)
    debug_print("Setting up style targets")
    style.set_style_targets(args.style_images, style_image_weights)
    debug_print("Done initialization")

    # Get the candidate image
    debug_print("Running gradient descent...")
    new_img = style.go(maxiter=args.numiter)

    debug_print("Done gradient descent, saving image...")

    imsave(
        '{}/final.jpg'.format(dirname),
        skimage.img_as_ubyte(new_img)
    )

    debug_print("Done: Saved {}".format(dirname))

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()

    parser.add_argument('content_image', help="Content image")
    parser.add_argument('style_images', nargs='*',
                        help="Style image(s)")
    parser.add_argument('-A', '--artist', type=str, default=None,
                        help=("Artist to imitate, Script will randomly "
                              "choose an artwork from this artist. Artist's "
                              "work must be saved in wikiart/artist_name!"))
    parser.add_argument('-n', '--numiter', type=int, default=512,
                        help="Number of iterations")
    parser.add_argument('-w', '--width', type=int, default=512,
                        help="Max image width")
    parser.add_argument('-i', '--init', choices=['rand', 'content'],
                        default='content',
                        help=("Initialize image from noise (rand) or original "
                              "image"))
    parser.add_argument('-p', '--print_rate', type=int, default=10,
                        help="How often to save intermediate images")
    parser.add_argument('-s', '--style_scale', type=float, default=1.0,
                        help=("Resize style image - changes resolution of "
                              "features"))
    # parser.add_argument('-c' '--color-transfer', action='store_true',
                        # help=("Apply color transfer algorithm to attempt to "
                              # "change style image to match color of the "
                              # "content image."))
    # TODO: Output location options?

    args = parser.parse_args()

    if args.artist and args.style_images:
        sys.exit("art.py: can't use both individual style "
                 "images and artist flag")
    if not args.artist and not args.style_images:
        sys.exit("art.py: need to specify either an artist or "
                 "style images")

    main(args)
