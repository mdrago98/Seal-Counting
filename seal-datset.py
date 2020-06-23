from math import sqrt
from pandas import read_excel

SEAL_WIDTH = 80
BOUNDING_BOX_DIAGONAL_TRANSLATION = sqrt(2*SEAL_WIDTH)/2

def get_excel_file(location, name='PixelCoordinates'):
    locations = read_excel(location, sheet_name=name)

def main(image_loc, excel_image_defs):
    pass