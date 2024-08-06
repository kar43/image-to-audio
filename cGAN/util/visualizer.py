import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from torch import is_tensor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa

class Visualizer(): # to visualise results
    def __init__(self, opt):
        self.opt = opt
        self.use_html = opt.isTrain and not opt.no_html
        self.name = opt.name
        self.win_size = opt.display_winsize

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            self.specgram_dir = os.path.join(self.web_dir, opt.spec_type)
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir, self.specgram_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):

        for label, image_tensor in visuals.items():
            if is_tensor(image_tensor):

                if self.opt.npy_spec: # store spectrograms as npy array and images as PIL img
                    if label.endswith('_B'):
                        spec_numpy = tensor_to_numpy(image_tensor)
                        visuals[label] = spec_numpy.copy()
                    else:
                        visuals[label] = util.tensor2im(image_tensor)
                else: # store all as images
                    visuals[label] = util.tensor2im(image_tensor)

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))

                if self.opt.npy_spec: # save spectrograms as npy array and display as pcolourmap
                    if label.endswith('_B'):
                        save_specgram_as_img(image_numpy, img_path) # save pcolourmap
                        npy_path = os.path.join(self.specgram_dir, 'epoch%.3d_%s.npy' % (epoch, label))
                        save_specgram_as_npy(image_numpy, npy_path) # save npy array
                    else:
                        util.save_image(image_numpy, img_path)
                else: # save all as images
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, opt=self.opt, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):

        for label, image_tensor in visuals.items():
            if is_tensor(image_tensor):

                if self.opt.npy_spec: # store spectrograms as npy arrays
                    if label.endswith('_B'):
                        spec_numpy = tensor_to_numpy(image_tensor)
                        visuals[label] = spec_numpy.copy()
                    else:
                        visuals[label] = util.tensor2im(image_tensor)
                else:
                    visuals[label] = util.tensor2im(image_tensor) # store all as images

        image_dir = webpage.get_image_dir()
        spec_dir = webpage.get_spec_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)

            if self.opt.npy_spec: # save spectrograms as npy array and pcolourmap
                if label.endswith('_B'):
                    save_specgram_as_img(image_numpy, save_path) # pcolourmap
                    npy_name = '%s_%s.npy' % (name, label)
                    save_path = os.path.join(spec_dir, npy_name) # npy array
                    save_specgram_as_npy(image_numpy, save_path)
                else:
                    util.save_image(image_numpy, save_path)
            else: # save all as images
                util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

        # self.save_audio(image_dir)

def tensor_to_numpy(tensor):
    image_numpy = tensor[0].data.cpu().float().numpy() # from tensor
    image_numpy = image_numpy.transpose((1,2,0))
    return image_numpy

def save_specgram_as_img(s, path):
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    librosa.display.specshow(s[:,:,0], ax=ax, y_axis='linear', x_axis='time', cmap='plasma')
    fig.savefig(path)

def save_specgram_as_npy(s, path):
    with open(path, 'wb') as f:
        np.save(f, s)

