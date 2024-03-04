from tkinter.filedialog import askdirectory
import tkinter as tk
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from copy import deepcopy
from process1 import SingleImage

class MultipleImages:
    """
    Class for managing the processing of an entire dataset

    Most of the information will be stored in self.images,
    which is a list of SingleImage objects.
    """
    def __init__(self, image_type = np.uint8):
        """
        Whenever the images are rescaled, the output range will be from the given type
        """
        self.type = image_type
        self.compactness = False
    
    def add_folder(self, path=None):
        """
        args:
        path (str)  :   The folder containing the images you want to process, if no path
                        is given, the user will be prompted to pick one
        """
        if path is None:
            root = tk.Tk()
            root.attributes("-topmost", True)
            root.iconify()
            path = askdirectory(title='Select image')
            root.destroy()
        self.path = path
    
    def get_info(self, threshold=2.5, findsize=True):
        """
        Runs add_info for each image in the saved path
        Excludes images that are not zoomed in enough

        args:
        threshold (float)   :   Any images with pixelsize above threshold will
                                not be included, can be set to None to include all images
        findsize (bool)     :   Whether to attempt to find the pixelsizes
        """
        images = []
        for fname in os.listdir(self.path):
            name, ext = os.path.splitext(fname)
            if ext == ".tif" or ext == ".Tif":
                image = SingleImage(image_type=self.type)
                image.add_path(path=os.path.join(self.path, fname))
                image.add_info(findsize=findsize)
                if findsize is False or threshold is None or image.image["pixelsize"] <= threshold:
                    images.append(image)
        self.images = images
    
    def get_data(self, bar_size=None):
        """
        Runs the add_data method for each image

        args:
        bar_size (int)  :   The size of the data bar at the bottom of the image
        """
        for image in self.images:
            if self.compactness is True:
                image.get()
            image.add_data(bar_size=bar_size)
            if self.compactness is True:
                image.store(folder=self.location_folder)

    def save_current(self, filename):
        """
        Saves the list containing the SingleImage objects
        in the given pickle file
        
        args:
        filename (str)  :   The name of the pickle file
        """
        if self.compactness is True:
            for image in self.images:
                image.get()
        with open(filename, 'wb') as f:
            pickle.dump(self.images, f)
        if self.compactness is True:
            for image in self.images:
                image.store(folder=self.location_folder)
    
    def load_current(self, filename):
        """
        Loads the list containing the SingleImage objects
        from the given pickle file
        
        args:
        filename (str)  :   The name of the pickle file
        """
        with open(filename, 'rb') as f:
            self.images = pickle.load(f)
        if self.compactness is True:
            for image in self.images:
                image.store(folder=self.location_folder)
    
    def filter(self, func, *args, **kwargs):
        """
        Filters the images using the given function
        If func needs more than the image as an argument
        other parameters cen be passed

        args:
        func (function)  :   Function for filtering the image
        All named and unnamed arguments other than func will be passed to func
        """
        for image in self.images:
            if self.compactness is True:
                image.get()
            
            image.filter(func, *args, **kwargs)
            
            if self.compactness is True:
                image.store(folder=self.location_folder)
    
    def make_compact(self, folder=""):
        """
        This will store the contents of each SingleImage object in separate
        pickle files while they are not used, if no folder
        is given the files will be stored in the working directory

        The methods will indovidually unpack images when they are needed
        
        args:
        folder (str)    :   Folder for storing the data, must exist(this code will not
                            create a new folder)"""
        if self.compactness is False:
            for image in self.images:
                image.store(folder=folder)
            self.compactness = True
        self.location_folder = folder

    def unmake_compact(self):
        """
        Unpacks all the image dictionaries and removes the pickle files
        """
        if self.compactness is True:
            for image in self.images:
                image.get()
            self.compactness = False
    
    def blur_parts(self, filter, masks = None, background="zero", interact=False, *args, **kwargs):
        """
        Method for removing unwanted areas of the images
        Selected areas will be replaced with a constant value and then blurred in order
        to avoid creating sharp edges
        
        args:
        filter (function)   :   Function for filtering the selected areas
        masks (list)        :   List of masks of bools, if none is given then one will
                                be created with all indexes being False
        background  (str)   :   Either "zero" or "mean", this value will replace the areas
                                selected using the mask
        interact (bool)     :   Whether to use the interactive lasso tool to remove objects from
                                the image
        Any other arguments will be passed directly to the filter function
        """
        for image, i in zip(self.images, range(len(self.images))):
            if self.compactness is True:
                image.get()
            
            if masks is None:
                mask = None
            else:
                mask = masks[i]
            image.blur_parts(filter=filter, mask=mask, background=background, interact=interact, *args, **kwargs)

            if self.compactness is True:
                image.store(folder=self.location_folder)
        
    def split_images(self, filter, store_new = False, indexes = "all", masks = None, background="zero", interact=False, *args, **kwargs):
        """
        Splits one image into two images that can be processed separately
        Useful if one part of an image has much brighter background and particles than another
        The new image will be placed directly after the original in the list
        
        args:
        filter (function)   :   The sections contained in one image will be removed from the other
        store_new (bool)    :   Whether the new image should be stored in a pickle file
                                This is mainly used not to clash with the process_singles method and
                                can be left at False
        indexes (list)      :   List indexes for the images you want to split, can also be an int for a single
                                image, or "all" to use all the images
                                The indexes will not be updated until after all the selected images has been processed
        masks (list)        :   masks (list)        :   List of masks of bools, if none is given then one will
                                be created with all indexes being False
        background  (str)   :   Either "zero" or "mean", this value will replace the areas
                                selected using the mask in the new image, and in the opposite areas
                                in the working image
        interact (bool)     :   Whether to use the interactive lasso tool to make the masks
        Any other arguments will be passed directly to the filter function
        """
        new_images = []
        if indexes == "all":
            indexes = range(len(self.images))
        elif isinstance(indexes, int):
            indexes = [indexes]
        
        #This part makes the masks and stores them along with the correct indexes
        for i in indexes:
            image = self.images[i]
            if self.compactness is True:
                image.get()
            if masks is None:
                mask = np.zeros_like(image.image["current"], dtype=np.bool_)
            else:
                mask = masks[i]
            if interact is True:
                mask = image.select_pixels(mask, title="Split Image")
            if np.any(mask):
                new_images.append((i,mask))

            if self.compactness is True:
                image.store(folder=self.location_folder)
        new_images.reverse()

        #This part creates the new images and changes the old ones
        for i, mask in new_images:
            image = self.images[i]
            if self.compactness is True:
                image.get()
            
            newimage = deepcopy(image)
            image.image["curname"] = image.image["curname"] + "1"           #Curname is used when storing the objects
            newimage.image["curname"] = newimage.image["curname"] + "2"     #If all image names are unique and the same length, this should not cause problems
            newimage.blur_parts(filter=filter, mask=mask, background=background, interact=False, *args, **kwargs)
            image.blur_parts(filter=filter, mask=~mask, background=background, interact=False, *args, **kwargs)
            self.images.insert(i+1, newimage)

            if self.compactness is True:
                image.store(folder=self.location_folder)
                newimage.store(folder=self.location_folder)
            elif store_new is True:                                         #process_singles overwrites self.compactness, so this is needed
                newimage.store(folder=self.location_folder)                 #to avoid problems
    
    def cap_brightness(self, minvals=0, maxvals=255, interact=False):
        """
        Narrows the range of values, the rescales
        This is useful for making the particles easier to isolate
        If minval or maxval is an int, it will be used for all images,
        If they are list they will be applied to the corresponding image by index

        args:
        minval (int/list)   :   all values below minval will be zero after rescaling
        maxval (int/list)   :   all values above maxval will be the max(255 for uint8) after scaling
        interact (bool)     :   Whether to use the interactive sliders to adjust minval and maxval
        """
        if isinstance(minvals, int):
            minvals = np.tile([minvals], len(self.images))
        if isinstance(maxvals, int):
            maxvals = np.tile([maxvals], len(self.images))
        for image, i in zip(self.images, range(len(self.images))):
            if self.compactness is True:
                image.get()
            
            image.cap_brightness(minval=minvals[i], maxval=maxvals[i], interact=interact)

            if self.compactness is True:
                image.store(folder=self.location_folder)
    
    def sobel(self, overwrite=True):
        """
        Generates and stores the sobel for each image

        args:
        overwrite (bool)    :   Whether to overwrite the sobel if one already exists
        """
        for image in self.images:
            if self.compactness is True:
                image.get()
            
            image.sobel(overwrite=overwrite)

            if self.compactness is True:
                image.store(folder=self.location_folder)
    
    def markers(self, upper=120, lower=80, overwrite=True):
        """
        Generates and stores the markers for use in segmentation in self.image["markers"]
        Any values between upper and lower will be marked as 0, and the segmenteation will
        determine of they're particles or background
        If upper or lower is an int, it will be used for all images,
        If they are lists they will be applied to the corresponding image by index

        If you already have the correct marker values, using cap_brightness may make
        your results worse

        args:
        upper (int/list)    :   All values above upper will be marked as particles
        lower (int/list)    :   All values below lower will be marked as background
        overwrite (bool)    :   Whether to overwrite the markers if they already exist
        """
        if isinstance(upper, int):
            upper = np.tile([upper], len(self.images))
        if isinstance(lower, int):
            lower = np.tile([lower], len(self.images))
        for image, i in zip(self.images, range(len(self.images))):
            if self.compactness is True:
                image.get()

            image.markers(upper=upper[i], lower=lower[i], overwrite=overwrite)
            
            if self.compactness is True:
                image.store(folder=self.location_folder)
    
    def segmentation(self, erosions=2, dilations=2, overwrite=True):
        """
        Uses the skimage watershed method with the generated markers and sobel to find particles and background
        Then does binary erosions to remove small objects, followed by dilations to increase the size of the
        particles that survived the erosions
        If erosions or dilations is an int, it will be used for all images,
        If they are lists they will be applied to the corresponding image by index

        args:
        erosions (int/list)     :   Amount of binary erosions to be performed, too many will remove actual particles
        dilations (int/list)    :   Amount of binary dilations to be performed, too many will cause the segmented particles
                                    to become diamond shaped
        overwrite (bool)        :   Whether to overwrite the segmentation if one already exists
        """
        if isinstance(erosions, int):
            erosions = np.tile([erosions], len(self.images))
        if isinstance(dilations, int):
            dilations = np.tile([dilations], len(self.images))
        for image, i in zip(self.images, range(len(self.images))):
            if self.compactness is True:
                image.get()

            image.segmentation(erosions=erosions[i], dilations=dilations[i], overwrite=overwrite)
            
            if self.compactness is True:
                image.store(folder=self.location_folder)

    def particle_detection(self, overwrite=True):
        """
        User the skimage regionprops method to generate a dataframe of detected particles
        Each row is a particle, and each column is a property
        To access the particles in the dataframe, you can access self.images[i].image["particles"]

        args:
        overwrite (bool)    :   Whether to overwrite the dataframes if they already exist
        """
        for image, i in zip(self.images, range(len(self.images))):
            if self.compactness is True:
                image.get()

            image.particle_detection(overwrite=overwrite)
            
            if self.compactness is True:
                image.store(folder=self.location_folder)
    
    def filter_particles(self, size=2, circularity=0.5):
        """
        Filters out particles from each image based on the given criteria
        circularity appears to be slightly biased towards filtering out big particles.
        
        args:
        size (int/list)             :   amount of pixels in particle, any amount below size will be removed
        circularity (float/list)    :   a product of particle area divided by perimeter, using the fact that 
                                        a circle has the lowest ratio of area to perimeter.
                                        The higher the value the more are filtered
        """
        if isinstance(size, int):
            size = np.tile([size], len(self.images))
        if isinstance(circularity, float):
            circularity = np.tile([circularity], len(self.images))
        for image, i in zip(self.images, range(len(self.images))):
            if self.compactness is True:
                image.get()

            image.filter_particles(size=size[i], circularity=circularity[i])
            
            if self.compactness is True:
                image.store(folder=self.location_folder)
    
    def show_particles(self, im="data"):
        """
        Displays the images with outlines and numberings on the detected particles
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
        for image in self.images:
            if self.compactness is True:
                image.get()

            image.show_particles(im=im)
            
            if self.compactness is True:
                image.store(folder=self.location_folder)

    def return_data(self, im="data"):
        """
        Used extracts a single type of data from each image
        The output is a list

        args:
        im (str)    :   Which image to display, options are:
                            "path"          :   The loacation of the image
                            "pixelsize"     :   The sidelength of a pixel, assumed to be square
                            "name"          :   The name of the image file
                            "curname"       :   The name of the image used for storing,
                                                will be equal to image if image hasn't been split
                            "data"          :   The original image
                            "bar            :   The info bar from the bottom of the image
                            "current"       :   The processed image
                            "sobel"         :   The sobel
                            "markers"       :   The markers
                            "segmented"     :   The segmented image
                            "particles"     :   The dataframe of detected particles
                            "amount"        :   The amount of detected particles
        """
        images = []
        for image in self.images:
            if self.compactness is True:
                image.get()
            
            images.append(image.image[im])

            if self.compactness is True:
                image.store(folder=self.location_folder)
        return images
    
    def process_singles(self, filter=None, split_args=None, blur_args=None, cap_args=None, markers_args={}, segment_args={}, filter_args={}, show_args={}):
        """
        Processes each image completely before moving to the next one
        
        args:
        filter  (dict)      :   Arguments for filtering images, will be passed on to the filtering function
        split_args (dict)   :   Arguments for splitting images, will be passed on to the splitting method
        blur_args (dict)    :   Arguments for blurring images, will be passed on to the blurring method
        cap_args (dict)     :   Arguments for limiting brightness values, will be passed on to the capping method
        markers_args (dict) :   Arguments for finding the markers, will be passed on to the markers method
        segment_args (dict) :   Arguments for segmenting the image, passed to the segmentation method
        filter_args (dict)  :   Arguments for filtering the particles, will be passed to the filter_particles method
        show_args (dict)    :   Arguments for showing the particles, will be passed to the show_particles method

        filter, split_args, blur_args, cap_args and show_args will skip their step when set to None
        """
        i = 0
        recompact_after = False
        while i < len(self.images):

            image = self.images[i]

            if self.compactness is True:
                self.compactness = False
                recompact_after = True
                image.get()

            if filter is not None:
                image.filter(**filter)
            if split_args is not None:
                self.split_images(indexes=i, store_new=recompact_after, **split_args)
            if blur_args is not None:
                image.blur_parts(**blur_args)
            if cap_args is not None:
                image.cap_brightness(**cap_args)
            image.sobel()
            image.markers(**markers_args)
            image.segmentation(**segment_args)
            image.particle_detection()
            image.filter_particles(**filter_args)
            if show_args is not None:
                image.show_particles(**show_args)

            if recompact_after is True:
                self.compactness=True
                recompact_after=False
                image.store(folder=self.location_folder)
            i += 1

    def combine_particles(self, combine=True):
        """
        Extracts the detected particles from each image object
        If combine is True, this will return a single concatenated dataframe
        of all the particles along with the total area spanned by the images
        If combine is false it will return a list of dataframes, one for each image,
        and a list of areas, one for each image.

        If an image has been split, only one area will be counted, and other images split
        from one image will count as 0 on the list
        
        args:
        combine (bool)      :   Whether to combine the data"""
        all_particles = []
        seen_images = []
        all_areas = []
        for image in self.images:
            if self.compactness is True:
                image.get()
            
            current = image.image["particles"].loc[:,("equivalent_diameter", "area")]

            #This part ensures split images don't count for double the area
            if image.image["path"] not in seen_images:
                seen_images.append(image.image["path"])
                all_areas.append(image.image["current"].size*(image.image["pixelsize"]**2))
                all_particles.append(current)
            else:
                all_particles[-1] = pd.concat([all_particles[-1],current])
                all_areas.append(0)

            if self.compactness is True:
                image.store(folder=self.location_folder)

        if combine is True:
            all_particles = pd.concat(all_particles, ignore_index=True)
            all_areas = np.sum(all_areas)
        
        return all_particles, all_areas
    
    def overall_stats(self, bin_width=5, show=True, return_values=False, bins_range=None, num_bins=None):
        """
        Computes the histograms of the detected particles, can show the graphs and return the statistics

        args:
        bin_width (int)         :   How wide the bins should be, will be ignored if num_bins is not None
        show (bool)             :   Whether to show the bar graphs
        return_values (bool)    :   Whether to return a dictionary containing the statistics
        bins_range (tuple)      :   Tuple containing the min and max vals of the bins, any values outside the range will be ignored
                                    If None, the min and max particle sizes will be used
        num_bins (int)          :   How many bins to use
        """
        particles, area = self.combine_particles()
        x_max = np.max(particles["equivalent_diameter"])
        x_min = np.min(particles["equivalent_diameter"])
        if num_bins is None:
            num_bins = max(int((x_max-x_min)/bin_width),1)
        if bins_range is None:
            bins_range=(x_min, x_max)
        bin_width = (bins_range[1]-bins_range[0])/num_bins
        hist, bins = np.histogram(particles["equivalent_diameter"], bins=num_bins, range=bins_range)
        centre = (bins[:-1] + bins[1:])/2
        hist_area, bins_area = np.histogram(particles["equivalent_diameter"],
                                    bins=num_bins, weights=particles["area"],
                                    density=True, range=bins_range)
        hist_norm_area = hist_area/np.sum(hist_area)
        centre_area = (bins_area[:-1] + bins_area[1:])/2
        ecd_mean = particles["equivalent_diameter"].mean()
        ecd_std = particles["equivalent_diameter"].std()
        ecd_mean_area = particles["area"].mean()
        ecd_std_area = particles["area"].std()
        total_particle_area = particles["area"].sum()
        particle_amount = len(particles.index)
        density_frequency = 1000000*particle_amount/area        #Converts from square nanometers to micrometers
        density_area = 100*total_particle_area/area

        #Plots the histograms
        if show is True:
            bar_width = 0.8*bin_width
            bar_align = 'center'
            fig, ax = plt.subplots(figsize=(12, 5))
            label_mean = r'$\bar{\delta}_{\mathrm{ECD}}$ = %i nm' % ecd_mean
            ax.bar(centre, hist, width=bar_width, align=bar_align)
            ax.set_xlabel(r'Particle size $\delta_{\mathrm{ECD}}$ [nm]')
            ax.set_ylabel('Frequency weighted')
            ax.plot([ecd_mean, ecd_mean], [0, hist.max()], 'r', label=label_mean)
            ax.legend()
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(centre, hist_norm_area, width=bar_width, align=bar_align)
            ax.plot([ecd_mean, ecd_mean], [0, hist_norm_area.max()], 'r', label=label_mean)
            ax.set_xlabel(r'Particle size $\delta_{\mathrm{ECD}}$ [nm]')
            ax.set_ylabel('Area weighted volume fraction')
            ax.legend()
            print("Particle density = {:.3E} #/µm^2\nParticle area = {:.4f}%".format(density_frequency, density_area))
        
        #Returns a dictionary containing the statistics
        if return_values is True:
            return {"hist":hist,"bins":bins,"centre":centre,"hist_area":hist_area,"hist_norm_area":hist_norm_area,
                    "ecd_mean":ecd_mean,"ecd_std":ecd_std,"ecd_mean_area":ecd_mean_area,"ecd_std_area":ecd_std_area,"total_particle_area":total_particle_area,"particle_amount":particle_amount,
                    "density_frequency":density_frequency,"density_area":density_area}

    def save_stats_dict(self, filename, bin_width=5, bins_range=None, num_bins=None):
        """
        Computes the statistics using the overall_stats method, then saves the generated dictionary
        to the given location

        args:
        filename (str)      :   Name of the pickle file
        bin_width (int)         :   How wide the bins should be, will be ignored if num_bins is not None
        bins_range (tuple)      :   Tuple containing the min and max vals of the bins, any values outside the range will be ignored
                                    If None, the min and max particle sizes will be used
        num_bins (int)          :   How many bins to use 
        """
        stats_dict = self.overall_stats(bin_width=bin_width, show=False, return_values=True, bins_range=bins_range, num_bins=num_bins)
        with open(filename, 'wb') as f:
            pickle.dump(stats_dict, f)