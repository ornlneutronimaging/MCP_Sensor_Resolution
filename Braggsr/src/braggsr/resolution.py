'''def calc_resolution(pix_pos: np.ndarray, intensity: np.ndarray) -> float: 
    Calculating the resolution of a TIF neutron radiograph by pulling in the edge space function from the integrated .TIFF gray scale value

    Parameter
    ---------
    pix_pos: pixel location in .TIFF as a 1D NumPy array
    intensity: gray scale value at the associated pixel location 

    Returns
    -------
    Resolution in units of milimeters
    
    print("To be implemented")
    return 0.0'''
import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sc
import pandas as pd
import tifffile 
from tifffile.tifffile import imread
import imageio
import cv2 as cv
#import lmfit.models
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel
def NR_Resolution():

#establish prefix for files
    def data (data_path):
        """Ingests a file, checks the path, runs through tifffile package to convert to numpy array, and then checks the data type to ensure it's compatible with the image processing regime
        """
        #data_path = '/SNS/VENUS/IPTS-35945/shared/images_normalized/Gd Mask Normalization' 
        assert os.path.exists(data_path)
    #display file dimension and data type (determine whether file is Float 32 and needs to be converted to uint8 [single channel unsigned])
        img = imread(data_path+'/normalized_sample_7998_obs_8015/integrated.tif')
        if img.dtype == "float32":
            print ("Image needs processing")
            return(img)
        elif img.dtype == "uint8":
            print ("Image does not need further conversion")
        else:
            print ("Image needs",img.dtype,"conversion method")
        #img.shape
        #fig,ax = plt.subplots(1, 1, figsize=(15,15))
        #ax.imshow(img, cmap="gray")
        #print (img.shape)
        #print (img.dtype)
        
    #files from TimePix1 will be Float32, and will need to undergo conversion to uint8
    def data_conversion (img, alpha, beta) #alpha is 0, beta is 255 for RGB to grayscale conversion
        gray = cv.bitwise_not(img)
        cv.normalize(gray, gray, alpha, beta, norm_type=cv.NORM_MINMAX)
        gray_8bit =cv.convertScaleAbs(gray)
        #Replots in true B/W -- notice introduction of artifacts
        bw = cv.adaptiveThreshold(gray_8bit, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
        return (bw)
    #creates two separate images at highest contrast to separate horizontal and vertical features
    def horizontal_image_processing(bw):
        horizontal = np.copy(bw)
        #horizontal processing -- separates and saves horizontal features
        cols = horizontal.shape[1]
        horizontal_size = cols // 30
        horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size,1))
        horizontal = cv.erode(horizontal, horizontalStructure)
        horizontal = cv.dilate(horizontal, horizontalStructure)
        horizontal = cv.bitwise_not(horizontal)
        H_edges = cv.adaptiveThreshold(horizontal, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, -2)
        kernel = np.ones((2,2),np.uint8)
        H_edges = cv.dilate(H_edges, kernel)
        smooth = np.copy(horizontal)
        smooth = cv.blur(smooth, (2,2))
        (rows,cols) = np.where(H_edges != 0)
        horizontal [rows,cols] = smooth[rows, cols]
        return (horizontal)
        #plt.imshow(horizontal)
    def vert_image_processing(bw):
        #vertical processing -- separates and saves vertical features
        vertical = np.copy(bw)
        rows = vertical.shape[0]
        verticalsize = rows // 30
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1,verticalsize))
        #vertical processing -- *enhance!*
        vertical = cv.erode(vertical, verticalStructure)
        vertical = cv.dilate(vertical, verticalStructure)
        vertical = cv.bitwise_not(vertical)
        V_edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, -2)
        kernel = np.ones((2,2),np.uint8)
        V_edges = cv.dilate(V_edges, kernel)
        smooth = np.copy(vertical)
        smooth = cv.blur(smooth, (2,2))
        (rows,cols) = np.where(V_edges != 0)
        vertical [rows,cols] = smooth[rows, cols]
        return(vertical)
        #Recombines the horizontal and vertical images into a single processed and edge-optimized image using 
        #OpenCV "addWeighted" function
        #print (horizontal.shape)
        #print (vertical.shape)
    def imagine_recombine(horizontal, alpha, vertical, beta)    
        """ Recombines the horizontal and vertical processed images into a single image using OpenCV weighted image stacking method (addWeighted)

        Parameters
        ----------
        horizontal: np.ndarray (2D)
            The horizontal portion of the parent image post-processing
        vertical: np.ndarray (2d)
            The vertical portion of the parent image post-processing
        alpha: float
            Value between 0 and 1; the weighting of the horizontal portion of the image. Defaults to .5.
        beta: float
            Value between 0 and 1-alpha; the weighting of the vertical portion of the image. Defaults to 1-alpha (.5).
        
        Returns
        -------
        recombined_image: np.ndarray (2D)
            Agglomerated 2D array representing the weighted values of the horizontal image with the vertical image weight superimposed. 
        """
        #alpha = .5
        #beta = 1-alpha
        recombined_image = cv.addWeighted(horizontal, alpha, vertical, beta, 0.0)
        #plt.imshow (combined_image)
        #plt.colorbar ()
        return (recombined_image)
    #Defines the ROI and processes the image to detect and identify the edges
    def ROI (recombined_image, x1,y1,width1, height1)
        #x1,y1,width1,height1 =  110,100,130,215
        roi_1=recombined_image[y1:y1+height1,x1:x1+width1]
        edges=cv.Canny(roi_1,0,150)
        return (edges)

    #export the edge locations
    def exp_edge_loc (edges, x1, y1,,file_name_edges) #save_loc
        edge_locations = np.column_stack(np.where(edges != 0))
        edge_locations[:,0] += y1
        edge_locations[:,1] += x1
        np.savetxt(file_name_edges, edge_locations, fmt='%d', header = "Row, Column")
        return(edge_locations)
    #establish individual 9x1 boxes along egdes to run through script
    def ROI_zones (edge_locations, x1, y1, left_range, right_range, box_edge_locations_fileName)
        #box_width = 9
        #left_range = 5
        #right_range = 3
        boxes=[]
        for (x1,y1) in edge_locations:
            left_edge=x1-left_range
            right_edge=x1+right_range
            box=(left_edge,y1, right_edge, y1)
            boxes.append(box)
        boxes = np.array(boxes)
        #print(boxes)
        np.savetxt(box_edge_locations_fileName, boxes, fmt='%d', header = "Row, Column")
        return(boxes)
    
    #establish the class for the model comparison (picking fit model with the highest R^2 value)
    def run_fit (boxes, y1, left_edge, right_edge, integrated_image_file):
        #Saves and iterates all saved ROIs through the resolution script
        image=imageio.imread(integrated_image_file)
        class Model:
            def __init__ (self, name ,r_squared, fwhm):
                self.name = name 
                self.r_squared = r_squared 
                self.fwhm = fwhm 
            def get_r_squared(self):
                    return self.r_squared
            def get_fwhm(self):
                return self.fwhm
        # prepare results container
        results_dict = {}  
        for box in boxes:
            #establishes the box
            left_edge,y1,right_edge,y1=box
            roi = image[y1,left_edge:right_edge]

            #converts the ROI to dataframe to follow rest of program
            df_roi = pd.DataFrame(roi, columns=['Intensity'])
            df1 = df_roi["Intensity"]

            #finds derivative of edge spread function using NumPy gradient (small dataset)
            dydxdf = np.gradient(df1, 1)

            #Gaussian fit model
            y = dydxdf
            x = np.arange(len(y))
            mod = GaussianModel()
            pars = mod.guess(y, x=x)
            out = mod.fit(y,pars, x=x)
            r_squared_G=out.rsquared
            FWHM_G=out.params['fwhm'].value

            #Lorentzian fit model
            y = dydxdf
            x = np.arange(len(y))
            mod = LorentzianModel()
            pars = mod.guess(y, x=x)
            outL = mod.fit(y,pars, x=x)
            r_squared_L = outL.rsquared
            FWHM_L=outL.params['fwhm'].value

            #Voigt fit model
            y = dydxdf
            x = np.arange(len(y))
            mod = VoigtModel()
            pars = mod.guess(y, x=x)
            outV = mod.fit(y,pars, x=x)
            r_squared_V=outV.rsquared
            FWHM_V=outV.params['fwhm'].value

            #Determines best model fit by comparing R-squared values, selects best to determine FWHM value from
            ModelG = Model("Gaussian",r_squared_G,FWHM_G)
            ModelL = Model("Lorentzian", r_squared_L, FWHM_L)
            ModelV = Model("Voigt", r_squared_V, FWHM_V)
            models = [ModelG, ModelL, ModelV]
            best_model = max (models, key=lambda model: model.get_r_squared())
            pixel_density = .055 *best_model.get_fwhm() #mm
        
            # record results
            if best_model.r_squared > 0.9:
                results_dict[(left_edge, y1, right_edge)] = best_model.get_fwhm()
            #print(results_dict)
            fwhm_values = list(results_dict.values())
            std_dev_FWHM = np.std(fwhm_values)
            if fwhm_values:
                average_fwhm = sum(fwhm_values)/len (fwhm_values)
            spatial_resolution = .055 *average_fwhm #mm
            print ("The sample size of the calculation was", (len(results_dict)/4),",using data with R^2 value in excess of .9.")
            print (f"The average resolution across the ROI is {average_fwhm:.8f} pixels, or {spatial_resolution:.8f} mm. Data-set standard deviation was {std_dev_FWHM:.8f}.")




