from PIL import Image, ImageDraw
import math
S=256
img=Image.new("RGBA",(S,S),(0,0,0,0))
d=ImageDraw.Draw(img)
# ears - floppy brown, behind head
d.ellipse([18,55,95,210], fill=(139,69,19,255))
d.ellipse([161,55,238,210], fill=(139,69,19,255))
d.ellipse([30,70,82,190], fill=(101,50,12,255))
d.ellipse([174,70,226,190], fill=(101,50,12,255))
# head - warm tan
d.ellipse([48,28,208,225], fill=(255,205,140,255), outline=(180,120,50,255), width=4)
# white patches around eyes / muzzle area
d.ellipse([68,95,125,150], fill=(255,255,255,255))
d.ellipse([131,95,188,150], fill=(255,255,255,255))
# muzzle
d.ellipse([78,145,178,205], fill=(255,250,240,255), outline=(220,170,110,255), width=3)
# eyes
for cx in (95,161):
    d.ellipse([cx-14,105,cx+14,133], fill=(20,20,20,255))
    d.ellipse([cx-3,110,cx+7,120], fill=(255,255,255,255))
# nose
d.ellipse([118,158,138,178], fill=(20,20,20,255))
d.ellipse([123,161,131,168], fill=(80,80,80,255))
# mouth
d.arc([108,175,128,195], start=0, end=180, fill=(120,60,20,255), width=3)
d.arc([128,175,148,195], start=0, end=180, fill=(120,60,20,255), width=3)
# tongue
d.rounded_rectangle([118,190,138,214], radius=9, fill=(255,110,140,255))
# blush
d.ellipse([62,150,88,168], fill=(255,150,150,180))
d.ellipse([168,150,194,168], fill=(255,150,150,180))
img.save("/home/hatch/workspace/dog-tray/dog.ico", sizes=[(256,256),(48,48),(32,32),(16,16)])
print("saved")
