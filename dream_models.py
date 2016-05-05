#!/usr/bin/env python2

""" Model file """

from __future__ import division

# Standard lib
from cStringIO import StringIO
import sys
import os
import textwrap
import shutil
import subprocess
import random
import copy
import json

# This part is from py-faster-rcnn/tools/demo.py
# Mess with the path so we get the *correct* version of caffe
def _add_to_path(p):
    p = os.path.realpath(p)
    assert os.path.isdir(p)
    if p not in sys.path:
        sys.path.insert(0, p)

# Add caffe to PYTHONPATH
caffe_path = os.path.join('py-faster-rcnn', 'caffe-fast-rcnn', 'python')
_add_to_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = os.path.join('py-faster-rcnn', 'lib')
_add_to_path(lib_path)
        
# 3rd party
import numpy as np

import scipy.ndimage as nd
from scipy.interpolate import interp1d
from scipy.io import loadmat

from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import binary_dilation

from sklearn.cluster import KMeans

import PIL.Image

from IPython.display import clear_output, Image, display

from google.protobuf import text_format

import matplotlib.pyplot as plt

import cv2

from fast_rcnn.config import cfg as rcnn_cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms as rcnn_nms

import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

# Constants
ROOTDIR = '/home/david/Desktop/CompVis/Project'

IMAGENET_VAL_LABELS = os.path.join(ROOTDIR, 'devkit-1.0/data/ILSVRC2010_validation_ground_truth.txt')
IMAGENET_VAL_ROOT = os.path.join(ROOTDIR, 'images/val')
IMAGENET_CODES = os.path.join(ROOTDIR, 'devkit-1.0/data/meta.mat')

DEEPDREAM_ROOT = os.path.join(ROOTDIR, 'deepdream/models/bvlc_googlenet')

RCNN_ROOT = os.path.join(ROOTDIR, 'py-faster-rcnn')
RCNN_NETS = {
    'vgg16': ('VGG16', 'VGG16_faster_rcnn_final.caffemodel'),
    'zf': ('ZF', 'ZF_faster_rcnn_final.caffemodel'),
}
RCNN_CLASSES = (
    '__background__',
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# Classes

class FastRCNN(object):

    def __init__(self, p=RCNN_ROOT, net='vgg16', nms_thresh=0.3, conf_thresh=0.8):
        self.input_model_path = os.path.realpath(p)
        self.net_key = net
        
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        
        self.net = None
        
    def load(self):
        cfg = rcnn_cfg
        cfg.TEST.HAS_RPN = True
        
        net_name, net_path = RCNN_NETS[self.net_key]

        prototxt = os.path.join(
            cfg.MODELS_DIR, net_name, 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
        caffemodel = os.path.join(
            cfg.DATA_DIR, 'faster_rcnn_models', net_path)
        if not os.path.isfile(caffemodel):
            raise OSError('{:s} not found'.format(caffemodel))
        if not os.path.isfile(prototxt):
            raise OSError('{:s} not found'.format(prototxt))
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    def find_main_axis(self, image_fn):
        _, img_boxes = self.detect(image_fn)
        head_boxes = self.lookup(image_fn, 'head')
        
        img_center = []
        for boxes in img_boxes:
            for b0, b1, b2, b3 in boxes:
                img_center.append([(b0 + b2)/2.0, (b1+b3)/2.0])
        img_center = np.array(img_center)

        head_center = []
        for b0, b1, b2, b3 in head_boxes:
            head_center.append([(b0 + b2)/2.0, (b1+b3)/2.0])
        head_center = np.array(head_center)

        if img_center.shape[0] < 1:
            c0, c1 = 0, 0
        elif img_center.shape[0] == 1:
            c0, c1 = img_center[0, :]
        else:
            c0, c1 = np.mean(img_center, axis=0)
        
        if head_center.shape[0] < 1:
            c2, c3 = 0, 0
        elif head_center.shape[0] == 1:
            c2, c3 = head_center[0, :]
        else:
            c2, c3 = np.mean(head_center, axis=0)
        return img_boxes, head_boxes, np.array([c0, c1, c2, c3])
        
    def lookup(self, image_fn, tag):
        """ Lookup part locations for an image """
        tagfile = os.path.join(os.path.dirname(image_fn), 'tags.json')
        if not os.path.isfile(tagfile):
            return []
        with open(tagfile, 'rt') as fp:
            alltags = json.load(fp)
        tags = alltags.get(os.path.basename(image_fn), {})
        return tags.get(tag, [])
        
    def detect(self, image_fn):
        """ Detect object classes in an image using pre-computed object proposals."""
        image_fn = os.path.realpath(image_fn)
        if not os.path.isfile(image_fn):
            raise OSError('Image not found: {}'.format(image_fn))
            
        im = cv2.imread(image_fn)
        scores, boxes = im_detect(self.net, im)
        
        # Drop the background class
        scores = scores[:, 1:]
        boxes = boxes[:, 4:]
        
        # Filter out dumb detections
        final_scores = []
        final_boxes = []
        for cls_ind, cls in enumerate(RCNN_CLASSES[1:]):
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = rcnn_nms(dets, self.nms_thresh)
            dets = dets[keep, :]
            
            inds = np.where(dets[:, -1] >= self.conf_thresh)[0]
            if len(inds) < 1:
                continue
            final_scores.append(dets[inds, -1])
            final_boxes.append(dets[inds, :4])

        return final_scores, final_boxes

        
class DeepDream(object):
    
    dream_patch = textwrap.dedent(r"""
    diff --git a/deploy.prototxt b/deploy.prototxt
    index 50b54a9..1781a46 100644
    --- a/deploy.prototxt
    +++ b/deploy.prototxt
    @@ -1,10 +1,14 @@
     name: "GoogleNet"
    -layer {
    -  name: "data"
    -  type: "Input"
    -  top: "data"
    -  input_param { shape: { dim: 10 dim: 3 dim: 224 dim: 224 } }
    -}
    +
    +input: "data"
    +input_shape {
    +  dim: 10
    +  dim: 3
    +  dim: 224
    +  dim: 224
    +}
    +force_backward: true
    +
     layer {
       name: "conv1/7x7_s2"
       type: "Convolution"
    """)
    
    def __init__(self, p=DEEPDREAM_ROOT):
        self.input_model_path = os.path.realpath(p)
        self.net_fn = os.path.join(self.input_model_path, 'deploy.prototxt')
        self.param_fn = os.path.join(self.input_model_path,  'bvlc_googlenet.caffemodel')
        
        self.temp_model_fn = os.path.realpath(os.path.join('model', 'deploy.prototxt'))
        self.temp_patch_fn = os.path.realpath(os.path.join('model', 'dream_tmp.patch'))

        self.net = None
        self.guide_features = None
        self.objective = None
        self.end = None
    
    @property
    def dream_layers(self):
        return [k for k in self.net.blobs.keys() if k.endswith('output')]
    
    def set_guide(self, guide_fn):
        
        # Load the guide image
        guide = np.float32(PIL.Image.open(guide_fn))
        
        # Fiddle with the inputs
        h, w = guide.shape[:2]
        src, dst = self.net.blobs['data'], self.net.blobs[self.end]
        src.reshape(1,3,h,w)
        src.data[0] = self.preprocess(guide)
        self.net.forward(end=self.end)
        
        # Stash the features we need
        self.guide_features = dst.data[0].copy()
        self.objective = self.objective_guide
    
    def load(self):
        """ Patching model to be able to compute gradients.
        
        Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
        """

        # Patch the model file
        if os.path.isfile(self.temp_model_fn):
            os.remove(self.temp_model_fn)
        
        temp_dir = os.path.dirname(self.temp_model_fn)
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
            
        with open(self.temp_patch_fn, 'wt') as fp:
            fp.write(self.dream_patch)
        
        fp = open(self.temp_patch_fn, 'rt')
        try:
            shutil.copy2(self.net_fn, self.temp_model_fn)
            subprocess.check_call(['patch', '-p1', self.temp_model_fn], stdin=fp)
        finally:
            fp.close()

        # Load the model
        self.net = caffe.Classifier(self.temp_model_fn,
                                    self.param_fn,
                                    mean=np.float32([104.0, 116.0, 122.0]),
                                    channel_swap=(2,1,0))
    
    # a couple of utility functions for converting to and from Caffe's input image layout
    def preprocess(self, img):
        return np.float32(np.rollaxis(img, 2)[::-1]) - self.net.transformer.mean['data']

    def deprocess(self, img):
        return np.dstack((img + self.net.transformer.mean['data'])[::-1])
    
    def objective_L2(self, dst):
        dst.diff[:] = dst.data 
        
    def objective_guide(self, dst):
        x = dst.data[0].copy()
        y = self.guide_features
        ch = x.shape[0]
        x = x.reshape(ch,-1)
        y = y.reshape(ch,-1)
        A = x.T.dot(y) # compute the matrix of dot-products with guide features
        dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

    def make_step(self, step_size=1.5, jitter=32, clip=True):
        '''Basic gradient ascent step.'''

        src = self.net.blobs['data'] # input image is stored in Net's 'data' blob
        dst = self.net.blobs[self.end]

        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

        self.net.forward(end=self.end)
        self.objective(dst)  # specify the optimization objective
        self.net.backward(start=self.end)
        g = src.diff[0]
        # apply normalized ascent step to the input image
        src.data[:] += step_size/np.abs(g).mean() * g

        src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

        if clip:
            bias = self.net.transformer.mean['data']
            src.data[:] = np.clip(src.data, -bias, 255-bias)
    
    def deepdream(self, base_img, iter_n=10, octave_n=4, octave_scale=1.4, clip=True, showimg=True, **step_params):
        
        if self.objective is None:
            self.objective = self.objective_L2
        if self.end is None:
            self.end = 'inception_4c/output'

        # prepare base images for all octaves
        octaves = [self.preprocess(base_img)]
        for i in xrange(octave_n-1):
            octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

        src = self.net.blobs['data']
        detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                # upscale details from the previous octave
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

            src.reshape(1,3,h,w) # resize the network's input image size
            src.data[0] = octave_base+detail
            for i in xrange(iter_n):
                self.make_step(clip=clip, **step_params)

                # visualization
                vis = self.deprocess(src.data[0])
                if not clip: # adjust image contrast if clipping is disabled
                    vis = vis*(255.0/np.percentile(vis, 99.98))
                if showimg:
                    showarray(vis)
                print octave, i, self.end, vis.shape
                clear_output(wait=True)

            # extract details produced on the current octave
            detail = src.data[0]-octave_base
        # returning the resulting image
        return self.deprocess(src.data[0])


class PartDream(object):
    """ Do deep dreaming with overlayed part models """

    def __init__(self, dream_net, weight=0.3, sigma=64, net_iters=3, net_end='inception_5b/output',
                 global_dream_weight=0.2):
        self.dream_net = dream_net
        self.weight = weight
        self.sigma = sigma
        self.global_dream_weight = global_dream_weight

        self.baseimg = None
        self.net_end = net_end
        self.net_iters = net_iters
        self.tempimg_fn = os.path.realpath('temp_guide.jpg')
        self.guides = []
        
    def set_base(self, baseimg):
        if isinstance(baseimg, str):
            baseimg = np.float32(PIL.Image.open(baseimg))
        self.baseimg = baseimg
    
    def add_part(self, guideimg, base_roi, guide_roi):
        if isinstance(guideimg, str):
            guideimg = np.float32(PIL.Image.open(guideimg))
        self.guides.append((guideimg, base_roi, guide_roi))
    
    def guide_dream(self, img, guideimg, base_roi, guide_roi):
        dream_net = self.dream_net
        
        img, gimg = make_composite_image(img, guideimg, base_roi, guide_roi,
                                         sigma=self.sigma,
                                         weight=self.weight,
                                         global_dream_weight=self.global_dream_weight)
        savearray(gimg, self.tempimg_fn)
        
        # Run deep dream
        dream_net.end = self.net_end
        dream_net.set_guide(self.tempimg_fn)

        for _ in range(self.net_iters):
            # TODO: Add masking into the dream algo...
            img = dream_net.deepdream(img, iter_n=10, octave_n=4)
        return img, maskimg
    
    def dream(self):
        outimg = self.baseimg.copy()
        for guideimg, base_roi, guide_roi in self.guides:
            oimg, omask = self.guide_dream(outimg, guideimg, base_roi, guide_roi)
            outimg = oimg * omask + outimg * (1-omask)
        return outimg

    
class ImageNet(object):
    
    def __init__(self, imagedir=IMAGENET_VAL_ROOT, codefile=IMAGENET_CODES):
        self.imagedir = imagedir
        self.codefile = codefile
        
        self._index_to_label = None
        self._label_to_index = None
    
    def keys(self):
        self.read_codefile()
        return self._label_to_index.keys()
    
    def get_imagefile(self, label, index=None):
        imagepath = os.path.join(self.imagedir, label[0], label)
        if not os.path.isdir(imagepath):
            raise KeyError(label)
        imagefiles = sorted(os.listdir(imagepath))
        if index is None:
            index = random.randint(0, len(imagefiles)-1)
        imagefile = os.path.join(imagepath, imagefiles[index])
        return imagefile
    
    def load_image(self, label, index=None):
        imagefile = self.get_imagefile(label, index=index)
        return np.float32(PIL.Image.open(imagefile))
    
    def sort_validation(self, intext):
        
        indir = self.imagedir

        assert os.path.isdir(indir)
        assert os.path.isfile(intext)

        self.read_codefile()
        index_to_label = self._index_to_label

        # Cute, duplicate labels are a great idea..
        val_labels = []
        with open(intext, 'rt') as fp:
            for i, line in enumerate(fp):
                if i % 100 == 0:
                    print(i, line)
                filename = 'ILSVRC2010_val_{:08d}.JPEG'.format(i+1)
                filepath = os.path.join(indir, filename)
                assert os.path.isfile(filepath), filepath
                label = index_to_label[int(line.strip())]

                subdir = os.path.join(indir, label[0], label)
                if not os.path.isdir(subdir):
                    os.makedirs(subdir)
                subpath = os.path.join(subdir, filename)
                shutil.move(filepath, subpath)

    def read_codefile(self):
        if self._index_to_label is not None:
            return
        
        incode = self.codefile
        assert os.path.isfile(incode)
        
        # Decode the insufferable matlab dictionary
        raw = loadmat(incode)['synsets']

        index_to_label = {}
        label_to_index = {}
        for i, row in enumerate(raw):
            idx = i + 1  # MATLAB is dumb
            label = row[0][2][0].split(',', 1)[0].strip().lower()
            index_to_label[i] = label
            if label not in label_to_index:
                label_to_index[label] = i
        self._index_to_label = index_to_label
        self._label_to_index = label_to_index
    
    
# Functions
    
def showarray(a, fmt='jpeg'):
    if isinstance(a, str):
        a = np.float32(PIL.Image.open(a))
    
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def savearray(a, outfile, fmt='jpeg'):
    if isinstance(a, str):
        a = np.float32(PIL.Image.open(a))
    
    a = np.uint8(np.clip(a, 0, 255))
    outfile = os.path.realpath(outfile)
    if not os.path.isdir(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    with open(outfile, 'wb') as f:
        PIL.Image.fromarray(a).save(f, fmt)

        
def histogram_transfer(s, t, crange=None):
    """ Transfer histograms from s to t
    
    :param np.array s:
        The image to transfer the histogram from
    :param np.array t:
        The image to transfer the histogram to
    :param tuple crange:
        The pair of [cmin, cmax) for the range of this color channel
        (default: (0, 256) i.e RGB)
    """
    
    if crange is None:
        crange = (0, 256)
    
    if s.ndim == 3 and t.ndim == 3:
        timg = []
        for ci in range(s.shape[2]):
            timg.append(histogram_transfer(s[:, :, ci], t[:, :, ci], crange=crange))
        return np.dstack(timg)
    assert s.ndim == 2
    assert t.ndim == 2
    
    crange = np.round(np.array(crange))
    cmin = np.min(crange)
    cmax = np.max(crange)
    nbins = np.ceil(cmax - cmin).astype(np.int)
    
    shist, sbins = np.histogram(s, bins=nbins)
    sbins = sbins[:-1]
    shist = np.cumsum(shist)
    shist_min = np.min(shist)
    shist = (shist - shist_min)/(s.shape[0]*s.shape[1] - shist_min) * (cmax-1) + cmin
    
    thist, tbins = np.histogram(t, bins=nbins)
    tbins = tbins[:-1]
    thist = np.cumsum(thist)
    thist_min = np.min(thist)
    thist = (thist - thist_min)/(t.shape[0]*t.shape[1] - thist_min) * (cmax-1) + cmin

    # Look up the values for t in s
    f = interp1d(shist, sbins, kind='nearest')
    tfixed = f(t.flatten())
    return np.reshape(tfixed, t.shape)


def plot_detections(img_fn, img_boxes, head_boxes=None, guide_boxes=None, img_axis=None):
    """ Plot the detection fields """
    
    img = np.float32(PIL.Image.open(img_fn))/255.0
    rows, cols, _ = img.shape
    fig, axes = plt.subplots(1, 1, figsize=(12, 12), subplot_kw={'xticks': [], 'yticks': []})
    axes.imshow(img, aspect='equal')

    for boxes in img_boxes:
        for bbox in boxes:
            axes.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor='red', linewidth=3.5))
    if img_axis is not None:
        axes.plot([img_axis[0], img_axis[2]], [img_axis[1], img_axis[3]], 'r-o')
    
    if head_boxes is not None:
        for bbox in head_boxes:
            axes.plot((bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0, 'bo')
            axes.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='blue', linewidth=3.5))
    if guide_boxes is not None:
        for bbox in guide_boxes:
            axes.plot((bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0, 'go')
            axes.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='green', linewidth=3.5))
        

def segment_guide_parts(guideimg_fn, guide_parts, dilate=5):
    """ Create a segmentation mask for the guide parts """
    
    im = cv2.imread(guideimg_fn)
    mask = np.zeros(im.shape[:2])

    rows, cols, _ = im.shape

    for b0, b1, b2, b3 in guide_parts:
        b0 = int(max((0, np.floor(b0))))
        b1 = int(max((0, np.floor(b1))))
        b2 = int(min((rows, np.ceil(b2) + 1)))
        b3 = int(min((cols, np.ceil(b3) + 1)))

        part = im[b1:b3, b0:b2]
        cls = KMeans(n_clusters=2)
        labels = cls.fit_predict(np.reshape(part, (-1, 3)))
        labels = np.reshape(labels, part.shape[:2])

        border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
        border_labels, border_counts = np.unique(border, return_counts=True)

        bg_class = border_labels[np.argmax(border_counts)]
        fg_mask = labels != bg_class
        for _ in range(dilate):
            fg_mask = binary_dilation(fg_mask)
        mask[b1:b3, b0:b2] = fg_mask
    return mask


def find_box_correspondence(guide_bbox, part_bbox, base_bbox, expand=2.0):
    """ Find the relation between guide and part, then apply that relation to base """
    
    def _swap_dirs(bbox):
        x0, x1, x2, x3 = bbox
        if x0 > x2:
            x2, x0 = x0, x2
        if x1 > x3:
            x3, x1 = x1, x3
        return x0, x1, x2, x3
    
    g0, g1, g2, g3 = _swap_dirs(guide_bbox)
    p0, p1, p2, p3 = _swap_dirs(part_bbox)
    b0, b1, b2, b3 = _swap_dirs(base_bbox)
    
    gcx = (g2+g0)/2
    gcy = (g3+g1)/2
    
    pcx = (p2+p0)/2
    pcy = (p3+p1)/2
    
    part_x = abs(p2 - p0) / 2.0
    part_y = abs(p3 - p1) / 2.0
    
    # Vector from center of guide to center of part
    dx = pcx - gcx
    dy = pcy - gcy
    
    # Scale between the two boxes
    sdx = abs(b2 - b0) / abs(g2 - g0) * expand
    sdy = abs(b3 - b1) / abs(g3 - g1) * expand
    
    bcx = (b2+b0)/2
    bcy = (b3+b1)/2
    
    # New center for the box
    gbx = bcx + dx*sdx
    gby = bcy + dy*sdy
    
    gb0 = gbx - part_x*sdx
    gb1 = gby - part_y*sdy
    
    gb2 = gbx + part_x*sdx
    gb3 = gby + part_y*sdy
    return gb0, gb1, gb2, gb3 
    
def make_composite_image(baseimg, guideimg, base_roi, guide_roi, sigma=64, weight=0.5, global_dream_weight=0.2):
    
    img = baseimg.copy()
    
    # Composite the base and guide images
    bx0, bx1, by0, by1 = np.round(base_roi).astype(np.int)
    gx0, gx1, gy0, gy1 = np.round(guide_roi).astype(np.int) 
    
    # Save the guide image so caffe can use it
    gimg = histogram_transfer(baseimg, guideimg)
    rimg = resize(gimg[gx0:gx1, gy0:gy1]/255.0, (bx1-bx0, by1-by0))*255

    # Blend the two back together with a mask
    maskimg = np.zeros_like(baseimg)
    maskimg[bx0+sigma//2:bx1-sigma//2, by0+sigma//2:by1-sigma//2] = 1.0
    maskimg = gaussian(maskimg, sigma, multichannel=True)
    maskimg[maskimg < global_dream_weight] = global_dream_weight

    bimg = rimg * weight + baseimg[bx0:bx1, by0:by1] * (1-weight)
    mimg = maskimg[bx0:bx1, by0:by1]

    img[bx0:bx1, by0:by1] = mimg * bimg + baseimg[bx0:bx1, by0:by1] * (1-mimg)

    return img, gimg
