from PIL import Image
import os
import os.path

rootdir = '/home/dong/python-project/Dehaze/DATA/Test/val/30_rotate270.png'

im = Image.open(rootdir)
out_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
newname = '/home/dong/python-project/Dehaze/DATA/Test/val/30_rotate270_flip.png'

# out_rotate = im.transpose(Image.ROTATE_270)
out_flip.save(newname)

# out_flip = out_rotate.transpose(Image.FLIP_LEFT_RIGHT)
# out_flip_name = 'home/dong/python-project/Dehaze/DATA/Test/val/30_rotate270_flip.png'
# out_flip.save(out_flip_name)