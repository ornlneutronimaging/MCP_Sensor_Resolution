"""Resolution is a module that determines neutron radiography spatial resolution through automated edge identification and
establishment of the line spread function full width, half maximum value"""

import os 
import numpy as np
import pandas as pd
from tifffile.tifffile import imread
import imageio
import cv2 as cv
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel
def nr_normalized_data (data_path:str)->str:
    """
    Checks file path, establishes callable object for normalized data. 

    Ingests a file, checks the path, runs through tifffile package to convert to numpy array, and then checks the data type to ensure it's compatible with the image processing regime (built for the normalized TimePix 1 file output--32 point floating bit TIF file). If the 
    image is either already a unsigned single channel 8-bit image (uint8) or a completely separate data type, the program will warn of this and then recommend the next step. 

    Parameters
    ----------
    data_path: str
        The file path for the normalized image-file should be 32-bit floating point TIF file

    Returns
    -------
    img: np.ndarray (2D)
        The image, as a callable numpy N-dimensional array (2D), identified with a print statement affirming it is either 1) in need of conversion from Float32 to uint8, 2) it is already formatted as a uint8 data type and so doesn't need conversion, or
        3) is another non-compatible type that will need further conversion. Note: ALL TimePix-1 normalized images *should* be natively Float32 TIFFs.
    """ 
    assert os.path.exists(data_path)
    #display file dimension and data type (determine whether file is Float 32 and needs to be converted to uint8 [single channel unsigned])
    img = imread(data_path)
    if img.dtype == "float32":
        print ("Image needs processing")
        return(img)
    elif img.dtype == "uint8":
        print ("Image does not need further conversion--skip to horizontal_image_processing function")
    else:
        print ("Image needs",img.dtype,"conversion method")
    return (img)

def data_conversion (img:np.ndarray, alpha:int, beta:int)->np.ndarray:
    """Takes the image(img) identified in the nr_normalized_data function and converts it from native Float32 to uint8 for compatibility with OpenCV image processing processes. Finally, converts to a true black/white image for highest contrast prior to edge identification. For
    TPX-1 normalized images, alpha is 0, beta is 255 for conversion. 

    Parameters
    ----------
    img: np.ndarray (2D)
        The image, as a callable numpy N-dimensional array (2D), formatted in Float32 (native data type for normalized TimePix-1 neutron radiographs)
    alpha: int
        A value, between 0 and 255, that sets the floor for color-channel values (grayscale will be converted to RGB valuation with much higher granularity than 16-step grayscale)
    beta: int
        A value, between 0 and 255, that sets the ceiling for color-channel values (grayscale will be converted to RGB valuation with much higher granularity than 16-step grayscale)

    Returns
    -------
    bw: np.ndarray (2D)
        A uint8-type ndarray compatible with OpenCV image processing techniques used to increase contrast and resolution 
    """
    gray = cv.bitwise_not(img)
    cv.normalize(gray, gray, alpha, beta, norm_type=cv.NORM_MINMAX)
    gray_8bit =cv.convertScaleAbs(gray)
    #Replots in true B/W -- notice introduction of artifacts
    bw = cv.adaptiveThreshold(gray_8bit, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    return (bw)
    
def horizontal_image_processing(bw:np.ndarray, h_scale_factor: int, blocksize: int, c_factor: int)->np.ndarray:
    """Takes the black and white, uint8 type image (bw) and processes its horizontal features to increase contrast using OpenCV's erosion and dilation techniques. Introduces some smoothing (cv.blur) at the end in order to finalize contrast and reduce noise at edges prior to Canny Edge
    detection technique being introduced to processed image.  

    Parameters
    ----------
    bw: np.ndarray (2D)
        A uint8-type ndarray compatible with OpenCV image processing techniques used to increase contrast and resolution 
    h_scale_factor: int
        An integer for which to scale (or bin) the processing of the horizontal data. 30 works well for images for the MCP sensor. 
    blocksize: int
        An integer that specifies the dimensions (square) of the neighborhood for calculating the adaptive threshold. The larger the block, the lower the local resolution,
        but can better handle image-wide variations in illumination. 
    c_factor: 
    Returns
    -------
    horizontal: np.ndarray(2D)
        The horizontal portions of the image, eroded, dilated, and then smoothed in order to increase contrast and resolution of the edges for follow-on Canny edge detection 
    """
    horizontal = np.copy(bw)
    #horizontal processing -- separates and saves horizontal features
    cols = horizontal.shape[1]
    horizontal_size = cols // h_scale_factor
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size,1))
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    horizontal = cv.bitwise_not(horizontal)
    H_edges = cv.adaptiveThreshold(horizontal, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blocksize, c_factor)
    kernel = np.ones((2,2),np.uint8)
    H_edges = cv.dilate(H_edges, kernel)
    smooth = np.copy(horizontal)
    smooth = cv.blur(smooth, (2,2))
    (rows,cols) = np.where(H_edges != 0)
    horizontal [rows,cols] = smooth[rows, cols]
    return (horizontal, h_scale_factor)

def vert_image_processing(bw:np.ndarray,h_scale_factor:int )->np.ndarray:
    """Takes the black and white, uint8 type image (bw) and processes its vertical features to increase contrast using OpenCV's erosion and dilation techniques. Introduces some smoothing (cv.blur) at the end in order to finalize contrast and reduce noise at edges prior to Canny Edge
    detection technique being introduced to processed image.

    Parameters
    ----------
    bw: np.ndarray (2D)
        A uint8-type ndarray compatible with OpenCV image processing techniques used to increase contrast and resolution 
    h_scale_factor: int
        An integer for which to scale (or bin) the processing of the horizontal data. 30 works well for images for the MCP sensor. Please
        note that this may need to be changed if the image is not square (i.e. m x n pixels where m =/= n). 
    Returns
    -------
    vertical: np.ndarray(2D)
        The vertical portions of the image, eroded, dilated, and then smoothed in order to increase contrast and resolution of the edges for follow-on Canny edge detection 
    """
    #vertical processing -- separates and saves vertical features
    vertical = np.copy(bw)
    rows = vertical.shape[0]
    verticalsize = rows // h_scale_factor
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1,verticalsize))
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

def imagine_recombine(horizontal:np.ndarray, alpha:float, vertical:np.ndarray, beta:float)->np.ndarray:    
    """Recombines the horizontal and vertical processed images into a single image using OpenCV weighted image stacking method (addWeighted)

    Parameters
    ----------
    horizontal: np.ndarray (2D)
        The horizontal portion of the parent image post-processing
    vertical: np.ndarray (2D)
        The vertical portion of the parent image post-processing
    alpha: float
        Value between 0 and 1; the weighting of the horizontal portion of the image (defaults to .5)
    beta: float
        Value between 0 and 1-alpha; the weighting of the vertical portion of the image (defaults to 1-alpha)
    
    Returns
    -------
    recombined_image: np.ndarray (2D)
        Agglomerated 2D array representing the weighted values of the horizontal image with the vertical image weight superimposed 
    """
    #alpha = .5
    #beta = 1-alpha
    recombined_image = cv.addWeighted(horizontal, alpha, vertical, beta, 0.0)
    #plt.imshow (combined_image)
    #plt.colorbar ()
    return (recombined_image)

#Defines the ROI and processes the image to detect and identify the edges
def Canny_edges (recombined_image:np.ndarray, x1:int,y1:int,width1:int, height1:int)->np.ndarray:
    """Takes in the 2D array recombined_image and conducts the Canny edge detection across the established region of interest (ROI)

    Parameters
    ----------
    recombined_image: np.ndarray (2D)
        Agglomerated 2D array representing the weighted values of the horizontal image with the vertical image weight superimposed 
    x1: int
        The left-most horizontal position for the desired region of interest
    y1: int
        The uppermost (closest to the origin--image is plotted in 3rd quadrant as abs (x,y)) vertical position for the desired region of interest 
    width1: int
        The width of the desired region of interest 
    height1: int
        The height for the desired region of interest 
    
    Returns
    -------
    edges: np.ndarray (2D)
        All identified edges in the desired region of interest highlighted and stores in 2D numpy array with spans for regions of interest
    """
    #x1,y1,width1,height1 =  110,100,130,215
    roi_1=recombined_image[y1:y1+height1,x1:x1+width1]
    edges=cv.Canny(roi_1,0,150)
    return (edges, x1, y1)

#export the edge locations
def exp_edge_loc (edges:np.ndarray, x1:int, y1:int,file_name_edges:str) -> np.ndarray:
    """Takes in the 2D edges numpy array for the desired region of interest, identifies the indices of the non-zero row and column elements, combines those two lists
    (rows, columns) into a single 2D array corresponding to the coordinates of an edge pixel, converts those coordinates back to 
    parent image coordinates, and stores coordinates for plotting ROIs across parent image. 

    Parameters
    ----------
    edges: np.ndarray (2D)
        All identified edges with regions of interest highlighted and stores in 2D numpy array with spans for regions of interest
    x1: int
        The left-most horizontal position for the desired region of interest
    y1: int
        The uppermost (closest to the origin--image is plotted in 3rd quadrant as abs (x,y)) vertical position for the desired region of interest 
    file_name_edges: str
        Desired output file name 

    Return
    ------
    edge_locations: np.ndarray (2D)
        A 2D array containing the locations of the edges in the parent image for processing of the resolution function on the parent 
        image 
    """ 
    edge_locations = np.column_stack(np.where(edges != 0))
    edge_locations[:,0] += y1
    edge_locations[:,1] += x1
    np.savetxt(file_name_edges, edge_locations, fmt='%d', header = "Row, Column")
    return(edge_locations)

def ROI_zones (edge_locations:np.ndarray, x1:int, y1:int, left_range:int, right_range:int, zone_locations_fileName:str)->np.ndarray:
    """Takes in the edge locations, establishes zones of a desired height and width (default is 9 pixels wide, 1 pixel tall), and saves
    boxes as 2D array in which the y-value is steady across the box; output data represents left edge, y1 --> right edge, y1.

    Parameters
    ----------
    edge_locations: np.ndarray (2D)
        A 2D array containing the locations of the edges in the parent image for processing of the resolution function on the parent 
        image
    x1: int
        The left-most horizontal position for the desired region of interest
    y1: int
        The uppermost (closest to the origin--image is plotted in 3rd quadrant as abs (x,y)) vertical position for the desired region of interest 
    left_range: int
        The distance from the left of x1 to the left edge of the desired zone for analysis 
    right_range: int
        The distance from the right of x1 to the right edge of the zone
    
    Returns
    -------
    zones: np.ndarray (2D)
        Saves a 2D array with the zones as a series of boxes that have a single y value and x values that span from left to right lateral limits 

    """
    #box_width = 9
    #left_range = 5
    #right_range = 3
    zones=[]
    for (x1,y1) in edge_locations:
        left_lat_lim=x1-left_range
        right_lat_lim=x1+right_range
        zone=(left_lat_lim,y1, right_lat_lim, y1)
        zones.append(zone)
    zones = np.array(zones)
    #print(boxes)
    np.savetxt(zone_locations_fileName, zones, fmt='%d', header = "Row, Column")
    return(zones, y1, left_lat_lim, right_lat_lim)

def run_fit (zones:np.ndarray, y1:int, left_lat_lim:int, right_lat_lim:int, img:np.ndarray)->str:
    """Takes in the 2D array zones, the left and right lateral limits for the zone, and a file name. Redefines the image for analysis as the normalized, stacked image without the region of interest. Establishes
    the "Model" class and the data of interest (r-squared value of fit, full-width/half-maximum values of fit) from the LMFit Gaussian, Lorentzian, and Voigt fit models' internal parameter reports. Finally, analyzes
    the fit parameters, determines the strongest fit of the three based on the r-sqaured values, calls the FWHM value for that model (self-reported in the LMFit model report) and uses that FWHM value to determine sensor 
    spatial resolution. 

    Parameters
    ----------
    zones: np.ndarray (2D)
        A 2D array with the zones as a series of boxes that have a single y value and x values that span from left to right lateral limits 
    y1: int
        The uppermost (closest to the origin--image is plotted in 3rd quadrant as abs (x,y)) vertical position for the desired region of interest 
    left_lat_limit: int
        The leftmost extent of the zone
    right_lat_limit: int
        The rightmost extent of the zone
    img: np.ndarray (2D)
        The image, as a callable numpy N-dimensional array (2D), identified with a print statement affirming it is either 1) in need of conversion from Float32 to uint8, 2) it is already formatted as a uint8 data type and so doesn't need conversion, or
        3) is another non-compatible type that will need further conversion. Note: ALL TimePix-1 normalized images *should* be natively Float32 TIFFs.

    Returns
    -------
    str
        Print statements detailing the data sample size, average resolution and spatial resolution of the sensor, and the standard deviation of the data set as a measure of the strength of the data correlation 
    """
    #Saves and iterates all saved ROIs through the resolution script
    image=imageio.imread(img)
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
    for zone in zones:
        #establishes the zone
        left_lat_lim,y1,right_lat_lim,y1=zone
        roi = image[y1,left_lat_lim:right_lat_lim]

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
            results_dict[(left_lat_lim, y1, right_lat_lim)] = best_model.get_fwhm()
        #print(results_dict)
        fwhm_values = list(results_dict.values())
        std_dev_FWHM = np.std(fwhm_values)
        if fwhm_values:
            average_fwhm = sum(fwhm_values)/len (fwhm_values)
        spatial_resolution = .055 *average_fwhm #mm
        print ("The sample size of the calculation was", (len(results_dict)),",using data with R^2 value in excess of .9.")
        print (f"The average resolution across the ROI is {average_fwhm:.8f} pixels, or {spatial_resolution:.8f} mm. Data-set standard deviation was {std_dev_FWHM:.8f}.")









