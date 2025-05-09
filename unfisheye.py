import cv2
import numpy as np
from numpy import arange, sqrt, arctan, sin, tan, meshgrid, pi, pad, radians
from numpy import ndarray, hypot

# Callback function for trackbars (does nothing, just a placeholder)
def nothing(x):
    pass

# Create a named window
cv2.namedWindow('Slider Window')

# Create trackbars for FOV and PFOV with max value 180
cv2.createTrackbar('FOV', 'Slider Window', 142, 180, nothing)
cv2.createTrackbar('PFOV', 'Slider Window', 129, 180, nothing)
cv2.createTrackbar('PAD', 'Slider Window', 0, 100, nothing)
cv2.createTrackbar('Distortion', 'Slider Window', 0, 3, nothing)

def open_window(image, window_name, width, height):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)

image = cv2.imread('cctv.png')
format ="fullframe"

#INIT
width = image.shape[1]
height = image.shape[0]
xcenter = width // 2
ycenter = height // 2

dim = min(width, height)
x0 = xcenter - dim // 2
x1 = xcenter + dim // 2
y0 = ycenter - dim // 2
y1 = ycenter + dim // 2

#image = image[y0:y1, x0:x1]
width = image.shape[1]      #update cropped res
height = image.shape[0]

xcenter = (width-1) // 2      #update xcenter
ycenter = (height-1) // 2    #update ycenter

while True:
    
    output_image = image.copy()
    
    fov = cv2.getTrackbarPos('FOV', 'Slider Window')
    pfov = cv2.getTrackbarPos('PFOV', 'Slider Window')
    pad = cv2.getTrackbarPos('PAD', 'Slider Window')
    distort = cv2.getTrackbarPos('Distortion', 'Slider Window')

    # print(f"FOV: {fov}, PFOV: {pfov} PAD: {pad}, Distortion: {distort}")

#>>>>>>>>>>>>>>CONVERT<<<<<<<<<<<<<<<<<<<<<
    #FULLFRAME MODE
    diagonal = sqrt(width**2 + height**2)
    focal = diagonal / (2 * tan(pfov * pi / 360))
    inv_focal = 1.0 / focal

    x_arrange = arange(width)               #[1,2,3...width]
    y_arrange = arange(height)              #[1,2,3...height] 
    x, y = meshgrid(x_arrange, y_arrange)   #X=[1,2,3...width],[1,2,3...width] Y=[1,1,1],[2,2,2],...,[height,height,height]
    
    #map(x,y,inv_focal, diagonal)

#>>>>>>>>>>>>MAPPING<<<<<<<<<<<<<<<<<

    xd = x - xcenter    #jarak setiap horizontal dari centerimg
    yd = y - ycenter    #jarak setiap vertical dari centerimg

    rd = hypot(xd, yd)  #jarak dari centerimg ke setiap pixel
    phiang = arctan(inv_focal * rd)
    
    if distort == 0:
        ifoc = diagonal * 180 / (fov * pi)
        rr = ifoc * phiang
    
    if distort == 1:
        ifoc = diagonal / (2.0 * sin(fov * pi / 720))
        rr = ifoc * sin(phiang / 2)
    
    if distort == 2:
        ifoc = diagonal / (2.0 * sin(fov * pi / 360))
        rr = ifoc * sin(phiang)
    
    if distort == 3:
        ifoc = diagonal / (2.0 * tan(fov * pi / 720))
        rr = ifoc * tan(phiang / 2)
    
    
    rdmask = rd != 0                    #take all non zero from rd
    xs = xd.astype(np.float32).copy()   #xd = x - xcenter
    ys = yd.astype(np.float32).copy()   #yd = y - ycenter

    xs[rdmask] = (rr[rdmask] / rd[rdmask]) * xd[rdmask] + xcenter #new x pos = distort / diagonal from center * x from x center
    ys[rdmask] = (rr[rdmask] / rd[rdmask]) * yd[rdmask] + ycenter #new y pos = distort / diagonal from center * y from y center

    xs[~rdmask] = 0 #center img = 0
    ys[~rdmask] = 0

    output_image = cv2.remap(output_image, xs, ys, cv2.INTER_LINEAR) #remap the image to new x and y pos
    open_window(output_image, 'OUTPUT', 1366, 768) #open the window with the new image
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('output.png', output_image)
        break




# Destroy all windows
cv2.destroyAllWindows()