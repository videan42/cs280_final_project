#!/usr/bin/env python2

# Standard lib
import os
import json
import argparse

# 3rd party
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

# Constants

THISDIR = os.path.dirname(os.path.realpath(__file__))

# Class

class ImageTagger(object):

    def __init__(self, imgdir):
        self.imgdir = imgdir
        self.tagfile = os.path.join(imgdir, 'tags.json')

        self.tags = None

        self._fig = None
        self._ax = None
        self._mouse_down = None
        self._mouse_rects = []
        self._mouse_cur = None
        self._status = None
        self._bbox = []

    def load_tags(self):
        if self.tags is not None:
            return

        if os.path.isfile(self.tagfile):
            with open(self.tagfile, 'rt') as fp:
                tags = json.load(fp)
        else:
            tags = {}
        self.tags = tags

    def save_tags(self):
        if self.tags is None:
            return
        with open(self.tagfile, 'wt') as fp:
            tags = json.dump(self.tags, fp)

    def on_mouse_down(self, event):
        if self._mouse_down is None and event.button == 1 and event.inaxes:
            self._mouse_down = (event.xdata, event.ydata)
            b0, b1 = self._mouse_down
            self._mouse_cur = self._ax.add_patch(
                    plt.Rectangle((b0, b1), 1, 1, fill=False,
                                  edgecolor='red', linewidth=3.5))

    def on_mouse_up(self, event):
        if self._mouse_down is not None and event.inaxes:
            sx, sy = self._mouse_down
            ex, ey = (event.xdata, event.ydata)
            self._bbox.append((sx, sy, ex, ey))
            self._mouse_rects.append(self._mouse_cur)
            self._mouse_cur = None
        self._mouse_down = None
        self._mouse_cur = None

    def on_mouse_move(self, event):
        if self._mouse_down is not None and event.inaxes:
            b0, b1 = self._mouse_down
            b2, b3 = (event.xdata, event.ydata)

            self._mouse_cur.set_width(b2 - b0)
            self._mouse_cur.set_height(b3 - b1)
            self._mouse_cur.figure.canvas.draw()

    def on_key_press(self, event):
        if event.key in (' ', '\n', '\r', '\r\n'):
            plt.close()
        if event.key in ('d', ):
            plt.close()
            self._status = 'delete'

    def tag_img(self, imgfile, tag):
        print('tagging {}: {}'.format(tag, imgfile))

        imgname = os.path.basename(imgfile)
        img_tags = self.tags.get(imgname, {})
        bbox = img_tags.get(tag, [])
        self._bbox = bbox

        self._mouse_down = None
        self._mouse_rects = []
        self._mouse_cur = None
        self._ax = None
        self._fig = None
        self._status = None

        img = np.asarray(Image.open(imgfile))

        self._fig, self._ax = plt.subplots(1, 1, figsize=(16, 16))
        self._ax.imshow(img, aspect='equal')
        for b0, b1, b2, b3 in self._bbox:
            self._ax.add_patch(
                    plt.Rectangle((b0, b1), b2-b0, b3-b1, fill=False,
                                  edgecolor='red', linewidth=3.5))

        self._fig.canvas.mpl_connect('button_press_event', self.on_mouse_down)
        self._fig.canvas.mpl_connect('button_release_event', self.on_mouse_up)
        self._fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self._fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show()

        if self._status == 'delete':
            print('Removing: {}'.format(imgfile))
            os.remove(imgfile)
            if imgname in self.tags:
                del self.tags[imgname]
        else:
            img_tags[tag] = self._bbox
            self.tags[imgname] = img_tags

    def tag_all(self, tag):

        self.load_tags()

        imgs = [os.path.join(self.imgdir, tf)
                for tf in os.listdir(self.imgdir)
                if tf.lower().endswith(('.jpg', '.jpeg'))]
        for imgfile in imgs:
            self.tag_img(imgfile, tag)

        self.save_tags()

# Functions

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('animal')
    parser.add_argument('tag')
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    animal = args.animal.lower().strip()
    tag = args.tag.lower().strip()
    imgdir = os.path.join(THISDIR, 'images', 'val', animal[0], animal)

    tagger = ImageTagger(imgdir)
    tagger.tag_all(tag)


if __name__ == '__main__':
    main()
