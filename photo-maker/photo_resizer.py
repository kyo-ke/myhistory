import numpy as np
from PIL import Image
import argparse


parser = argparse.ArgumentParser(
    description='Resize Size Of Photo')
parser.add_argument('--left', '-l', default='0',
                    help='beginning of left side 0 to 100 l < r')
parser.add_argument('--right', '-r', default='100',
                    help='beginning of right side 0 to 100 l < r')
parser.add_argument('--top', '-t', default='0',
                    help='top of the photo 0 to 100 t < b')
parser.add_argument('--bottom', '-b', default='100',
                    help='bottom of the photo 0 to 100 t < b')
parser.add_argument('--original_image', '-oi', default='image.png',
                    help='original image')
args = parser.parse_args()

img_name = args.original_image
img = Image.open(img_name)
#row : 横　col : 縦
row,col = img.size
r = np.float32(args.right)
l = np.float32(args.left)
t = np.float32(args.top)
b = np.float32(args.bottom)
img_arr = np.asarray(img)
# Image　では　タテ，横、チャネルの順番
l_p = (np.float32(row)/100.0)*l
r_p = (np.float32(row)/100.0)*r
t_p = (np.float32(col)/100.0)*t
b_p = (np.float32(col)/100.0)*b

#キリトリ
n_img_arr = img_arr[int(l_p):int(r_p),int(t_p):int(b_p),:]
n_img = Image.fromarray(n_img_arr)

#セーブ
n_img.save('r_' + args.original_image,quality = 95)
