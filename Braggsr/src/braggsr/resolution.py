import numpy as np

def calc_resolution(pix_pos: np.ndarray, intensity: np.ndarray) -> float: 
    '''Calculating the resolution of a TIFF radiograph by pulling in the edge space function from the integrated .TIFF gray scale value

    Parameter
    ---------
    pix_pos: pixel location in .TIFF as a 1D NumPy array
    intensity: gray scale value at the associated pixel location 

    Returns
    -------
    Resolution in units of milimeters
    '''
    print("To be implemented")
    return 0.0
