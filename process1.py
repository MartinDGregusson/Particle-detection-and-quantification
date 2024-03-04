from tkinter.filedialog import askopenfilename
import tkinter as tk
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import rescale_intensity
from matplotlib.widgets import Button
from matplotlib.widgets import LassoSelector
from matplotlib.widgets import Slider
from matplotlib import path
import pickle
from skimage.filters import sobel
from skimage import segmentation
from skimage import measure
from scipy import ndimage as ndi
import pandas as pd

class Index_brightness:
    """
    Class for interactively limiting brightness values of image
    """

    def __init__(self, image, ax, fig, minval, maxval, minslider, maxslider):
        self.image = image
        self.ax = ax
        self.fig = fig
        self.minval = minval
        self.maxval = maxval
        self.minslider = minslider
        self.maxslider = maxslider
    
    def done(self, event):
        plt.close(self.fig)
    
    def show(self):
        self.ax.imshow(self.image["current"], cmap="gray", vmin=self.minval[0], vmax=self.maxval[0],interpolation='nearest')
        self.ax.set_xlim([0, self.image["current"].shape[1]])
        self.ax.set_ylim([0, self.image["current"].shape[0]])
        self.ax.invert_yaxis()
        self.ax.set_title(self.image["name"])
        self.fig.canvas.draw_idle()
    
    def updatemin(self, val):
        if val > self.maxval[0]:
            val = self.maxval[0]
        self.minval[0] = val
        self.show()
    
    def updatemax(self, val):
        if val < self.minval[0]:
            val = self.minval[0]
        self.maxval[0] = val
        self.show()

class Index_blur:
    """
    Class for interavtively making a mask of pixels in an image
    """

    def __init__(self, image, mask, ax, fig, minval=0, maxval=255):
        self.image = image
        self.mask = mask
        self.ax = ax
        self.fig = fig
        self.minval = minval
        self.maxval = maxval

        #These are all the coordinates in the image
        xv, yv = np.meshgrid(np.arange(mask.shape[1]),np.arange(mask.shape[0]))
        self.pix = (np.vstack( (xv.flatten(), yv.flatten()) ).T)

    def done(self, event):
        plt.close(self.fig)

    def onselect(self, verts):
        #This takes the path drawn with the lasso tool and changes the mask accordingly
        p = path.Path(verts)
        ind = np.reshape(p.contains_points(self.pix, radius=1), self.mask.shape)
        self.mask[ind] = True
        self.show()
    
    def reset(self, event):
        self.mask[:][:] = False
        self.show()
    
    def invert(self, event):
        self.mask[:][:] = ~self.mask[:][:]
        self.show()
    
    def show(self):
        self.ax.imshow(self.image["current"]*(1-self.mask), cmap="gray", vmin=self.minval, vmax=self.maxval,interpolation='nearest')
        self.ax.set_xlim([-20, self.image["current"].shape[1]+20])
        self.ax.set_ylim([-20, self.image["current"].shape[0]+20])
        self.ax.invert_yaxis()
        self.ax.set_title(self.image["name"])
        self.fig.canvas.draw_idle()

class SingleImage:
    """
    Class for managing the processing of a single image

    All the important data will be stored in self.image, or a pickle
    file if self.store is used
    """
    
    def __init__(self, image_type = np.uint8):
        """
        Whenever the image is rescaled, the output range will be from the given type
        """
        self.type = image_type
        self.image = {"path":None,"pixelsize":None,"name":None,"curname":None,"data":None,"bar":None,"current":None,"sobel":None,"markers":None,"segmented":None,"particles":None,"amount":None}

    def add_path(self, path = None):
        """
        args:
        path (str)  :   The location of the image you wish to process, if no
                        path is given, the user will be prompted to pick one
        """
        if path is None:
            root = tk.Tk()
            root.attributes("-topmost", True)
            root.iconify()
            path = askopenfilename(title='Select image')
            root.destroy()
        self.image["path"] = path

    def add_info(self, findsize=True):
        """
        Extracts the name and pixelsize of the image
        Name comes from the path given in self.path
        Pixelsize is expected to be found in a .txt file
        with the same name is the image

        args:
        findsize (bool)     :   Whether to attempt to find the pizelsize
        """
        fname = os.path.basename(self.image["path"])
        dirpath = os.path.dirname(self.image["path"])
        name, ext = os.path.splitext(fname)
        self.image["name"] = name
        self.image["curname"] = name
        if findsize is True:
            if ext == ".Tif":
                text_ext = ".Txt"
                with open(os.path.join(dirpath, name + text_ext), 'r') as document:
                    markersize = 1
                    markerwidth = 1
                    for line in document:
                        if line.startswith("$SM_MICRON_MARKER"):
                            text = line.split()[1]
                            value = int(text[:-2])
                            unit = text[-2:]
                            if unit == "um":
                                markerwidth = 1000*value
                            elif unit == "nm":
                                markerwidth = value
                        if line.startswith("$SM_MICRON_BAR"):
                            text = line.split()[1]
                            markersize = float(text)
                self.image["pixelsize"] = markerwidth/markersize  
            else:
                text_ext = ".txt"
                with open(os.path.join(dirpath, name + text_ext), 'r') as document:
                    for line in document:
                        if line.startswith("PixelSize="):
                            self.image["pixelsize"] = float(line[10:-1])
                            break

    def add_data(self, bar_size=None):
        """
        Extracts the data from the image file

        args:
        bar_size (int)  :   The size of the data bar at the bottom of the image

        "data" is the original image and "current" is the one we process
        """
        full_image = plt.imread(self.image["path"])
        if bar_size is not None or bar_size != 0:
            self.image["bar"] = full_image[-bar_size:]
            image = full_image[:-bar_size]
        else:
            image = full_image
        data = rescale_intensity(image, out_range=self.type)
        self.image["data"] = data
        self.image["current"] = data

    def save_current(self, filename):
        """
        Stores self.image in a pickle file

        args:
        filename (str)  :   Name of the pickle file the data should be stored in
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.image, f)
    
    def load_current(self, filename):
        """
        Loads a saved image dict from a pickle file
        
        args:
        filename (str)  :   Name of the pickle file the data should be extracted from
        """
        with open(filename, 'rb') as f:
            self.image = pickle.load(f)
    
    def store(self, folder=""):
        """
        This is used to save some space in the MultipleImages class
        there is no real point in manually using this method instead of
        save_current

        args:
        folder (str)  :   Name of the folder the pickle file should be stored in
        """
        self.location = os.path.join(folder, self.image["curname"]+".pkl")
        self.save_current(self.location)
        self.image.clear()
    
    def get(self):
        """
        Retrieves the data stored using the store method
        The pickle files are removed afterwards
        """
        self.load_current(self.location)
        os.remove(self.location)

    def filter(self, func, *args, **kwargs):
        """
        Filters the image using the given function
        If func needs more than the image as an argument
        other parameters cen be passed

        args:
        func (function)  :   Function for filtering the image
        All named and unnamed arguments other than func will be passed directly to func
        """
        self.image["current"] = func(self.image["current"], *args, **kwargs)
    
    def blur_parts(self, filter, mask = None, background="zero", interact = False, *args, **kwargs):
        """
        Method for removing unwanted areas of the image
        Selected areas will be replaced with a constant value and then blurred in order
        to avoid creating sharp edges
        
        args:
        filter (function)   :   Function for filtering the selected areas
        mask (array)        :   Premade mask of bools, if none is given then one will
                                be created with all indexes being False
        background  (str)   :   Either "zero" or "mean", this value will replace the areas
                                selected using the mask
        interact (bool)     :   Whether to use the interactive lasso tool to remove objects from
                                the image
        any other arguments will be passed directly to the filter function
        """
        if mask is None:
            mask = np.zeros_like(self.image["current"], dtype=np.bool_)
        if interact is True:
            mask = self.select_pixels(mask, title="Blur Image")
        data = self.image["current"]
        if background == "zero":
            replacement = 0
        elif background == "mean":
            replacement = np.mean(data)
        copy_data = np.copy(data)                                               #Creates a copy of the image
        copy_data = np.where(mask, replacement, copy_data)                      #Replaces the unwated objects in the copy
        copy_data = filter(copy_data, *args, **kwargs)                          #Filters the entire copy
        data = np.where(mask, copy_data, data)                                  #Replaces the unwated areas of the original
                                                                                #with the blurred areas in the copy
        self.image["current"] = rescale_intensity(data, out_range=self.type)
    
    def select_pixels(self, mask, title="Select Pixels"):
        """
        Interactively creates a mask of bools with the same shape as the image
        
        args:
        mask (array)        :   boolean mask
        title (str)         :   Name of the pyplot window, useful for distinguishing
                                between blurring and splitting, as they use the same interface
        """
        fig, ax = plt.subplots()
        ax.axis('off')
        img = ax.imshow(self.image["current"], cmap="gray", vmin=0, vmax=255, origin="upper",interpolation='nearest')
        ax.set_xlim([-20, self.image["current"].shape[1]+20])   #The extra border width lets us draw outside the edges
        ax.set_ylim([-20, self.image["current"].shape[0]+20])
        ax.invert_yaxis()
        callback = Index_blur(image=self.image, mask=mask, ax=ax, fig=fig)
        axdone = fig.add_axes([0.81, 0.05, 0.1, 0.075])
        axinv = fig.add_axes([0.7, 0.05, 0.1, 0.075])
        axres = fig.add_axes([0.59, 0.05, 0.1, 0.075])
        bdone = Button(axdone, 'Done')
        bdone.on_clicked(callback.done)
        binv = Button(axinv, 'Invert')
        binv.on_clicked(callback.invert)
        bres = Button(axres, 'Reset')
        bres.on_clicked(callback.reset)
        lasso = LassoSelector(ax, callback.onselect)
        ax.set_title(self.image["name"])
        plt.get_current_fig_manager().set_window_title(str(title))
        plt.show(block = False)                 #We don't want the program the proceed until we have finished our selection,
        while plt.fignum_exists(fig.number):    #but block=True causes jupyter notebook to stall forever. This section
            plt.pause(0.01)                     #can probably be replaced with just plt.show(block = True) when jupyter is fixed.
        return mask
    
    def cap_brightness(self, minval=0, maxval=255, interact = False):
        """
        Narrows the range of values, the rescales
        This is useful for making the particles easier to isolate

        args:
        minval (int)        :   all values below minval will be zero after rescaling
        maxval (int)        :   all values above maxval will be the max(255 for uint8) after scaling
        interact (bool)     :   Whether to use the interactive sliders to adjust minval and maxval
        """
        if interact is True:
            minval = np.array([minval])         #We must stored the values inside arrays so that the interactive tool can
            maxval = np.array([maxval])         #overwrite the values, but not the array
            minval, maxval = self.select_brightness(minval, maxval)
        self.image["current"] = rescale_intensity(self.image["current"], in_range=(minval,maxval), out_range=self.type)
    
    def select_brightness(self, minval, maxval, title="Select Brightness"):
        """
        Interactively creates a mask of bools with the same shape as the image
        
        args:
        minval (array)      :   Array holding the minval
        maxval (array)      :   Array holding the maxval
        title (str)         :   Name of the pyplot window, mainly here to conform with select_pixels
        """
        fig, ax = plt.subplots()
        ax.axis('off')
        img = ax.imshow(self.image["current"], cmap="gray", vmin=minval[0], vmax=maxval[0], origin="upper",interpolation='nearest')
        ax.set_xlim([0, self.image["current"].shape[1]])
        ax.set_ylim([0, self.image["current"].shape[0]])
        ax.invert_yaxis()
        ax_slider_min = plt.axes([0.20, 0.01, 0.65, 0.03])
        slider_min = Slider(ax_slider_min, 'Min brightness', 0, 255, valinit=0, valstep=1)
        ax_slider_max = plt.axes([0.20, -0.01, 0.65, 0.03])
        slider_max = Slider(ax_slider_max, 'Max brightness', 0, 255, valinit=255, valstep=1)
        callback = Index_brightness(image=self.image, ax=ax, fig=fig, minval=minval, maxval=maxval, minslider=slider_min, maxslider=slider_max)
        axdone = fig.add_axes([0.81, 0.05, 0.1, 0.075])
        bdone = Button(axdone, 'Done')
        bdone.on_clicked(callback.done)
        slider_min.on_changed(callback.updatemin)
        slider_max.on_changed(callback.updatemax)
        plt.get_current_fig_manager().set_window_title(str(title))
        plt.show(block = False)                     #We don't want the program the proceed until we have finished our selection,
        while plt.fignum_exists(fig.number):        #but block=True causes jupyter notebook to stall forever. This section
            plt.pause(0.01)                         #can probably be replaced with just plt.show(block = True) when jupyter is fixed.
        return minval[0], maxval[0]
    
    def delete_data(self):
        """
        Removes all the data from the dict
        After this we will have self.image be {}
        """
        self.image.clear()
    
    def sobel(self, overwrite=True):
        """
        Generates and stores the sobel for use in segmentation in self.image["sobel"]

        args:
        overwrite (bool)    :   Whether to overwrite the sobel if one already exists
        """
        if overwrite is True or self.image["sobel"] is None:
            self.image["sobel"] = sobel(self.image["current"])
    
    def markers(self, upper=120, lower=80, overwrite=True):
        """
        Generates and stores the markers for use in segmentation in self.image["markers"]
        Any values between upper and lower will be marked as 0, and the segmenteation will
        determine of they're particles or background

        args:
        upper (int)         :   All values above upper will be marked as particles
        lower (int)         :   All values below lower will be marked as background
        overwrite (bool)    :   Whether to overwrite the markers if they already exist
        """
        if overwrite is True or self.image["markers"] is None:
            markers = np.zeros_like(self.image["current"])
            markers[self.image["current"] < lower] = 1
            markers[self.image["current"] > upper] = 2
            self.image["markers"] = markers
    
    def segmentation(self, erosions=2, dilations=2, overwrite=True):
        """
        Uses the skimage watershed method with the generated markers and sobel to find particles and background
        Then does binary erosions to remove small objects, followed by dilations to increase the size of the
        particles that survived the erosions
        It generally seems best to keep the 2 values equal

        args:
        erosions (int)      :   Amount of binary erosions to be performed, too many will remove actual particles
        dilations (int)     :   Amount of binary dilations to be performed, too many will cause the segmented particles
                                to become diamond shaped
        overwrite (bool)    :   Whether to overwrite the segmentation if one already exists
        """
        if overwrite is True or self.image["segmented"] is None:
            segmented = segmentation.watershed(self.image["sobel"], self.image["markers"])
            segmented_filled = ndi.binary_fill_holes(segmented - 1)     #The generated segmentation is 1 for background and 2 for particle, so we subtract by 1
            if erosions != 0:                                           #and fill the holes in each particle before proceeding
                segmented_filled = ndi.binary_erosion(segmented_filled, iterations = erosions)
            if dilations != 0:
                segmented_filled = ndi.binary_dilation(segmented_filled, iterations = dilations)
            self.image["segmented"] = segmented_filled
    
    def particle_detection(self, overwrite=True):
        """
        User the skimage regionprops method to generate a dataframe of detected particles
        Each row is a particle, and each column is a property
        We store the dataframe in self.images["particles"] and the 
        amount of particles in self.images["amount"]

        args:
        overwrite (bool)    :   Whether to overwrite the dataframe if one already exists
        """
        if overwrite is True or self.image["particles"] is None:
            props_columns = ['pixels', 'equivalent_diameter', 'eccentricity', 'perimeter',
             'ellipse_major_axis', 'ellipse_minor_axis', 'centroid']
            labeled_particles, n = ndi.label(self.image["segmented"])
            properties = measure.regionprops(labeled_particles, intensity_image=self.image["current"])
            particles = pd.DataFrame(columns=props_columns)
            scale = self.image["pixelsize"]
            for region in properties:
                particles.loc[len(particles)] = [region.area,
                                                 region.equivalent_diameter * scale,
                                                 region.eccentricity,
                                                 region.perimeter * scale,
                                                 region.major_axis_length * scale,
                                                 region.minor_axis_length * scale,
                                                 region.centroid]
            self.image["particles"] = particles
            self.image["amount"] = n
    
    def filter_particles(self, size=2, circularity=0.5):
        """
        Filters out particles from self.image["particles"] based on the given criteria
        circularity appears to be slightly biased towards filtering out big particles.
        
        args:
        size (int)          :   amount of pixels in particle, any amount below size will be removed
        circularity (float) :   a product of particle area divided by perimeter, using the fact that 
                                a circle has the lowest ratio of area to perimeter.
                                The higher the value the more are filtered
        """
        pd.options.mode.chained_assignment = None  # default='warn'         #pandas gave some warnings, but the usage appears to be correct
        particles = self.image["particles"]                                 
        particles = particles[particles["pixels"] > size]
        particles['area'] = particles.loc[:,"pixels"].values * self.image["pixelsize"]**2
        particles['circularity'] = 4*np.pi*particles.area.values/particles.perimeter.values**2
        particles = particles[particles['circularity'] > circularity]
        n = len(particles)
        self.image["particles"] = particles
        self.image["amount"] = n
        pd.options.mode.chained_assignment = "warn"
    
    def show_particles(self, im = "data", blocking=False):
        """
        Displays the image with outlines and numberings on the detected particles
        There will also be outlines around the particles removed using filter_particles(),
        but they are not counted

        args:
        im (str)            :   Which image to display, options are
                                "data"        :   The original image
                                "current"     :   The processed image
                                "sobel"       :   The sobel
                                "markers"     :   The markers
                                "segmented"   :   The segmented image
        blocking (bool)     :   Whether the generated image should block further code from executing while displayed
        """
        if self.image["bar"] is not None:
            image = np.concatenate((self.image[im], self.image["bar"]), axis=0)     #Inserts the info bar back into the image to be displayed
        else:
            image = self.image[im]
        plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        plt.contour(self.image["segmented"], linewidths=0.5, colors='r')
        for i, (x, y) in enumerate(self.image["particles"].centroid.values):
            plt.text(y, x, s='%i' % i, fontdict={'size': 8}, color="white")
        plt.title(self.image["name"])
        plt.show(block = False)
        if blocking is True:                   #We don't want the program the proceed until we have finished our selection,
            while plt.get_fignums():           #but block=True causes jupyter notebook to stall forever. This section
                plt.pause(0.01)                #can probably be replaced with just plt.show(block = True) when jupyter is fixed.