"""
Implementation of A Neural Algorithm of Artistic Style

authors: Jesse Mu, Andrew Francl
"""

# Temporary for our caffe installation - matplotlib wants X11
import matplotlib
matplotlib.use('Agg')

import scipy.optimize as optimize
import numpy as np
import caffe
caffe.set_mode_gpu()
# XXX: We're loading pillow *and* image?
from PIL import Image
from itertools import izip_longest
# np.dot is too slow. Use blas.sgemm instead
import scipy.linalg.blas as blas
from math import sqrt

# Constants
# TODO: Make these user-specifiable parameters in argparse!
# VGG_MODEL = './models/VGG_ILSVRC_19_layers.caffemodel'
# VGG_PROTOTXT = './models/VGG_ILSVRC_19_layers_deploy.prototxt'
VGG_MODEL = './models2/vgg19/VGG_ILSVRC_19_layers.caffemodel'
VGG_PROTOTXT = './models2/vgg19/VGG_ILSVRC_19_layers_deploy.prototxt'

DIMX = 256
DIMY = 256

STYLE_IMAGE = './wikiart/vincent-van-gogh/the-starry-night-1889(1).jpg'
CONTENT_IMAGE = './images/content/sanfrancisco.jpg'

MEAN_PIXEL = np.array([103.939, 116.779, 123.68])

SC_RATIO = 1e4

PRINT_RATE = 1

# NOTE: We use the blob name instead of the layer name according to the model
# prototxt. Layers are specified in the paper.
CONTENT_LAYERS = ['conv4_2']
CONTENT_WEIGHTS = [1. / len(CONTENT_LAYERS)] * len(CONTENT_LAYERS)
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
STYLE_WEIGHTS = [1. / len(STYLE_LAYERS)] * len(STYLE_LAYERS)
# XXX only one style image weight supported?
STYLE_IMAGE_WEIGHTS = [1.]

class Art(object):
    """
    Your one-stop shop for all things neural style
    """
    def __init__(self, net):
        self.net = net
        # TODO: Allow preprocessing options
        self.transformer = self.create_transformer(None)
        
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

    def create_transformer(self, options):
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

    def set_style_targets(self, imgs):
        """
        Params
        ======
        imgs : List<str>
            Filename of the image to load in.
        """
        # TODO: Programatically figure out the desired shape
        # XXX: For now, just set style image weights equal

        # XXX: Is a list ok?
        target_sl_list = []
        for index, sl in enumerate(STYLE_LAYERS):

            sl_dim0 = self.net.blobs[sl].data[0].shape[0]
            target_sl = np.zeros((sl_dim0, sl_dim0))

            for img, weight in zip(imgs, STYLE_IMAGE_WEIGHTS):
                # Preproces image, load into net
                stylei = caffe.io.load_image(img)
                stylei_pp = self.transformer.preprocess('data', stylei)
                self.net.blobs['data'].data[...] = stylei_pp

                self.net.forward()

                layer = self.net.blobs[sl].data[0].copy()  # Get one batch?
                # Expand style layer to 2d array
                layer = np.reshape(layer, (layer.shape[0], layer.shape[1] ** 2))

                gram = self._gram(layer, weight)

                target_sl += gram

            target_sl_list.append(target_sl)

        self.style_targets = target_sl_list

        # Set counter for printing progress in graddesc
        self.iter = 0

    def set_content_target(self, img):
        # XXX: Assume only one content layer
        cl = CONTENT_LAYERS[0]
        contenti = caffe.io.load_image(img)
        contenti_pp = self.transformer.preprocess('data', contenti)
        self.net.blobs['data'].data[...] = contenti_pp
        self.net.forward()
        self.content_target = self.net.blobs[cl].data[0].copy()

    def random_image(self):
        # TODO: won't always be 224, fix this!!
        randi = (np.random.rand(3, 224, 224) * 255)
        return (randi.transpose() - MEAN_PIXEL).transpose()

    def _gram(self, layer, weight):
        # Compute averaged and weighted gram matrix.
        gram = blas.sgemm(1.0, layer, layer.T)
        # wut.
        # gram /= layer.size
        # gram *= weight
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
            # XXX: Bad things happened here
            try:
                gradient = size_c * blas.sgemm(1.0, style_noisy.T, diff).T * (style_noisy > 0) * weight
            except ValueError:
                print "BAD"
                import ipdb; ipdb.set_trace()
            return loss, gradient

        return loss, None

    def content_lag(self, content_noisy, compute_grad=False):
        """
        Compute content loss and gradient.

        This is compressed into one function to save intermediate computations.
        """
        # XXX: Assume only one content layer
        diff = (content_noisy - self.content_target)
        loss = .5 * (diff ** 2).sum()
        if compute_grad:
            try:
                gradient = diff * (content_noisy > 0)
            except ValueError:
                print "BAD"
                import ipdb; ipdb.set_trace()
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

        # COMPUTE LOSSES
        # For the first iteration, we don't care about the gradients.

        # Compute content losses.
        # XXX: Assume only one content layer
        content_loss, _ = self.content_lag(content_noisy)
        loss = content_loss

        # Collect style layers and gram matrices
        style_noisies = map(
            lambda layer: self.net.blobs[layer].data[0].copy(),
            STYLE_LAYERS
        )
        style_reshaped = map(
            lambda n: np.reshape(n, (n.shape[0], n.shape[1] ** 2)),
            style_noisies
        )
        style_grams = [self._gram(m, w) for m, w in
                       zip(style_reshaped, STYLE_WEIGHTS)]

        # Compute style losses and weight by their ratio
        # TODO: Add calculate_gradient flag to style_lag
        total_style_loss = 0
        for i in xrange(len(STYLE_WEIGHTS)):
            total_style_loss += self.style_lag(
                style_reshaped, style_grams, i, compute_grad=False
            )[0]

        loss += total_style_loss * SC_RATIO

        # Compute backprop LAYER BY LAYER to obtain gradients.
        # self.net.blobs is an ordered dict, so reversed makes sense
        # Initialize net to empty
        self.net.blobs[self.reversed_pairs[-1][0]].diff[:] = 0
        for curr, prev in self.reversed_pairs:
            # We only want to compute gradients for the style layers

            # Alias this for sanity
            curr_grad = self.net.blobs[curr].diff[0]

            # import ipdb; ipdb.set_trace()

            try:
                style_index = STYLE_LAYERS.index(curr)
            except ValueError:
                # Nope, not in style layers
                style_index = -1
            if style_index > -1:
                gradient = self.style_lag(
                    style_reshaped, style_grams, style_index, compute_grad=True
                )[1]
                sqrt_shape = int(sqrt(gradient.shape[1]))
                curr_grad += np.reshape(gradient, (gradient.shape[0], sqrt_shape, sqrt_shape))
            else:
                try:
                    content_index = CONTENT_LAYERS.index(curr)
                except ValueError:
                    # NOPE!
                    content_index = -1
                if content_index > -1:
                    # XXX: Assume weight is 1 since we're not changing this model
                    gradient = self.content_lag(
                        content_noisy, compute_grad=True
                    )[1]
                    # ValueError happens here
                    curr_grad += gradient

            # Compute the gradient from the current layer to the next (previous)
            # layer
            self.net.backward(start=curr, end=prev)

        final_grad = self.net.blobs['data'].diff[0]

        # Flatten for optimization
        print (loss, final_grad.flatten().astype(np.float64) * 100)
        return loss, final_grad.flatten().astype(np.float64) * 100


    def print_prog(self, something):
        self.iter += 1
        if (self.iter % PRINT_RATE) == 0:
            debug_print("gdesc iteration {}".format(str(iter)))
            self.iter = 0



    def go(self, maxiter=50):
        """
        This is where the magic happens.
        
        Return the image resulting from gradient descent for maxiter
        iterations
        """
        # Init random noise image
        debug_print("Running go")
        randi = self.random_image()
        debug_print("Generated random image")

        # TODO: Other optimization methods?
        debug_print("Starting grad descent")

        x, f, d = optimize.fmin_l_bfgs_b(
            self.loss_and_gradient,
            randi,
            fprime=None,  # We'll use loss_and_gradient
            maxiter=maxiter,
            callback=self.print_prog
        )

        px = sqrt(x.shape[0] / 3)
        x = np.reshape(x, (3, px, px))

        return self.transformer.deprocess('data', x)


def debug_print(msg, verbose=True):
    if verbose:
        print "{}: {}".format(datetime.now(), msg)


def main(args):
    """
    The main algorithm implementation function.
    """
    vgg = caffe.Net(
        VGG_PROTOTXT, VGG_MODEL, caffe.TEST
        # Mean pixel from internet.
        # mean=MEAN_PIXEL,
        # channel_swap=(2, 1, 0),
        # raw_scale=255,
        # image_dims=(DIMX, DIMY)
    )

    style = Art(vgg)

    # Get style and content targets
    debug_print("Setting up style targets")
    style.set_style_targets([STYLE_IMAGE])
    debug_print("Setting up content targets")
    style.set_content_target(CONTENT_IMAGE)
    debug_print("Done initialization")

    new_img = style.go()
    
    img = Image.fromarray(new_img, 'RGB')
    img.save('out.png')
    
    print "Done"


if __name__ == '__main__':
    from argparse import ArgumentParser
    from datetime import datetime

    parser = ArgumentParser()

    args = parser.parse_args()

    main(args)
