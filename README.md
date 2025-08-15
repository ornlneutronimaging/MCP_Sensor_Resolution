# MCP_Sensor_Resolution

> As of Aug 15, 2025, the development of the project is shifted to https://code.ornl.gov/ornlneutronimaging/BraggSR.

This repository allows the determination of the resolution of radiographs produced by the MCP sensor for VENUS (BL-10). 
It uses radiographs normalized using the Normalization_TOF program. 
The program "Resolution" functions by using image processing to increase contrast and employs Canny edge detection to find and catalog all edges in the horizontal and vertical planes. 
The identification of these edges forms the basis for regions of interest across which the edge spread function can be defined; this in turn marks the end of the image processing and the beginning of the mathematical manipulation. 
The edge spread function shows the intensity of the illumination across the region of interest; ideally, the region of interest spans dark to light, and has its minimum and maximum values at the left and right sides of the function, respectively. 
The derivative of this function is the line spread function, and shows the rate of change of the intensities as a function of the change in the pixels; as the pixel number increases, ideally the change in intensity at first steeply increases, then levels off, and then once again steeply decreases. 
This bell-curve shape enables curve fitting using a a standardized series of curve-fitting approximations (Gaussian, Laurentzian, and Voigt) from LMFit; the program makes use of all three and compares the strength of the three fits using their R^2 value as the selection criteria for which approximation to use before continuing on. 
The fit report for the curve produces a full-width, half-maximum (FWHM) value which is interpreted as the pixel resolution for the image; knowing that the MCP sensor has a maximum theoretical spatial resolution of .055mm/pixel, we multiply the two values together (determined FWHM * max theoretical spatial resolution) to obtain the empirical spatial resolution. 

This program draws on several other programs, to include making heavy use of image processing modules from OpenCV and curve fitting approximations from LMFit. 


