import os
from multiprocessing import Pool
from PIL import Image
from glob import glob
import tqdm

Image.MAX_IMAGE_PIXELS = 99999999999999999999
current_path = '/data2/seals/TIFFs'


def convert_file(name):
    outputfile = os.path.splitext(os.path.join(current_path, 'png', os.path.basename(name)))[0] + ".png"
    try:
        im = Image.open(os.path.join(name))
        im.thumbnail(im.size)
        im.save(outputfile, "PNG", quality=100)
    except Exception as e:
        print(e)


with Pool(6) as p:
    for _ in tqdm.tqdm(p.imap_unordered(convert_file, glob(f'{current_path}/*.tif')), total=100):
        pass
