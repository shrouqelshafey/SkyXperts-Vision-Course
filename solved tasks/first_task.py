import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#Task 1
path = 'C:\\Users\\Prof. Ahmed ElShafee\\Documents\\Python Projects\\parrot.jpg'
read_image=cv.imread(path)
img = cv.imread("parrot.jpg")
#BGR has 3 channels -> Blue, Green, Red
#Grayscale images have 1 channel only -> this channel represents the brightness of the pixel starting from 0 (black) to 255 (white)

img_bgr=cv.imread("parrot.jpg", cv.IMREAD_COLOR) #cv2.imread() func reads the image in BGR order 
img_gray=cv.imread("parrot.jpg", cv.IMREAD_GRAYSCALE) #cv2.imread() func reads the image in grayscale order 
#Displaying both
cv.imshow("BGR Image", img_bgr) #used plt to display image 
cv.imshow("Gray Image", img_gray)
 

#Task 2
#DOWNSCALING
scale= 60
#height=scale*height/100 width=scale*width/100
h,w , _= img.shape
h= int(scale*h / 100)
w= int(scale*w / 100 )
dim= (h,w)

downscaled_img= cv.resize(img, dim , interpolation = cv.INTER_AREA) #interpolation determines how opencv will be calculating the pixels in the new image
#cv.INTER_AREA is usually used when downscalling an image
#cv.INTER_CUBIC is higher quality but slow and it's commonly used when upscaling an image
#cv.INTER_NEAREST is fastest but low quality and makes the image look pixelated 

#UPSCALING
scale= 200
h,w , _= img.shape
h= int(scale *h / 100)
w= int(scale * w / 100 )
dim= (h,w)
upscaled_img= cv.resize(img, dim , interpolation = cv.INTER_CUBIC)  #i used INTER_CUBIC because it is better for upscaling (good quality images)

cv.imshow("Orignal", img_bgr)
cv.imshow("UP Scaled image ", upscaled_img)
cv.imshow("Down Scaled image ", downscaled_img)

#Task 3
print("Original image shape: ", img.shape)

h, w, _= img.shape
#Resize width to 100 pixels
dim= (100, h)
width_img= cv.resize(img, dim, interpolation=cv.INTER_AREA)

#Resize only height to 200 pixels
dim= (w, 200)
height_img= cv.resize(img, dim, interpolation=cv.INTER_AREA)

#Resize both width and height to 200 pixels
h= 200
w= 200
dim= (w, h)
HW_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

cv.imshow("Width = 100", width_img) #when the width changed and the height stayed the same, the image looked narrower and compressed
cv.imshow("Height = 200", height_img) #when the height changed and the width stayed the same, the image looked taller and stretched vertically
cv.imshow("200 x 200", HW_img) #both dimensions got compressed and the image tends to look like a square

#Task 4
#Scale up by 1.2 using INTER_LINEAR
up_linear= cv.resize(
    img, None, fx=1.2, fy=1.2, interpolation=cv.INTER_LINEAR
)

#Scale up by 1.2 using INTER_NEAREST
up_nearest= cv.resize(
    img, None, fx=1.2, fy=1.2, interpolation=cv.INTER_NEAREST
)

#Scale down by 0.6 using INTER_LINEAR
down_linear= cv.resize(
    img, None, fx=0.6, fy=0.6, interpolation=cv.INTER_LINEAR
)

#Scale down by 0.6 using INTER_NEAREST
down_nearest= cv.resize(
    img, None, fx=0.6, fy=0.6, interpolation=cv.INTER_NEAREST
)

#Display
cv.imshow("Original", img)
cv.imshow("Up 1.2 / Linear", up_linear)
cv.imshow("Up 1.2 / Nearest", up_nearest)
cv.imshow("Down 0.6 / Linear", down_linear)
cv.imshow("Down 0.6 / Nearest", down_nearest)
#When scaling the image up by 1.2, INTER_LINEAR produces a smoother image compared with INTER_NEAREST, which appears more pixelated.
#When scaling down by 0.6, both methods reduce the image size, but INTER_LINEAR generally gives smoother results,
#while INTER_NEAREST can lose more visual detail.

#Task 5

cropped_img=img[20:200, 50:200]

cv.imshow("Original Image", img)
cv.imshow("Cropped Image", cropped_img)

#Task 6
#Calculate midpoints for height and width to divide image into 4 blocks and know the end points for cropping
h, w, _= img.shape
mid_h=h // 2
mid_w=w // 2

#Cropping top-left, top-right, bottom-left, bottom-right
top_left= img[0:mid_h, 0:mid_w]
top_right= img[0:mid_h, mid_w:w]
bottom_left= img[mid_h:h, 0:mid_w]
bottom_right= img[mid_h:h, mid_w:w]

#Display each block
cv.imshow("Top Left", top_left)
cv.imshow("Top Right", top_right)
cv.imshow("Bottom Left", bottom_left)
cv.imshow("Bottom Right", bottom_right)

#Stitch back using NumPy
top= np.hstack((top_left, top_right)) #stitching the top left and right blocks horizontally
bottom= np.hstack((bottom_left, bottom_right)) #stitching the bottom left and right blocks horizontally
stitched_img= np.vstack((top, bottom)) #the final stitched image is now like the original image 

#Display stitched image
cv.imshow("Stitched Image", stitched_img) 

cv.waitKey(0)
cv.destroyAllWindows()

#Task 7
height, width, _= img.shape
#Calculate the center of the image
center=(width/2,height/2)

#rotation matrix for 45 degrees
rotate_matrix = cv.getRotationMatrix2D(center=center,angle=45,scale=1) #creates the matrix needed to rotate the image around its center
#The 45 rotation tilts the image diagonally. Since the output size remains the same,the corners is cut off

#Rotate the image using warpAffine
rotated_image = cv.warpAffine(src=img,M=rotate_matrix,dsize=(width, height)) #applies the rotation matrix to the image



#rotation matrix for 90 degrees
rotate_matrix_90= cv.getRotationMatrix2D(center=center,angle=90,scale=1) #The 90 rotation turns the image sideways

#rotation matrix for 180 degrees
rotate_matrix_180 = cv.getRotationMatrix2D(center=center,angle=180,scale=1) #The 180 rotation turns the image upside down.

rotated_image_90 = cv.warpAffine(src=img,M=rotate_matrix_90,dsize=(width, height))
rotated_image_180 = cv.warpAffine(src=img,M=rotate_matrix_180,dsize=(width, height))

cv.imshow("Rotated image 45", rotated_image)
cv.imshow("Rotated image 90", rotated_image_90)
cv.imshow("Rotated image 180", rotated_image_180)
#cv.waitKey(0)
#cv.destroyAllWindows()

#Task 8

height, width, _=img.shape
center=(width / 2, height / 2)

rotate_matrix = cv.getRotationMatrix2D(center=center,angle=45,scale=0.5) #when scale is 1, there is no resizing
rotated_scaled = cv.warpAffine(src=img,M=rotate_matrix,dsize=(width, height))
cv.imshow("Rotate 45 + Scale 0.5", rotated_scaled)

resized = cv.resize(img,None,fx=0.5,fy=0.5,interpolation=cv.INTER_LINEAR)
#Get the new height and width beacuse the image has been resized
new_height, new_width, _ = resized.shape
new_center = (new_width / 2, new_height / 2)
rotate_matrix_separate = cv.getRotationMatrix2D(center=new_center,angle=45,scale=1)
rotated_separate = cv.warpAffine(src=resized,M=rotate_matrix_separate,dsize=(new_width, new_height))

cv.imshow("Resize 0.5 then Rotate 45", rotated_separate)

#Task 9

#Convert BGR to RGB
rgb_img= cv.cvtColor(img, cv.COLOR_BGR2RGB)

#Convert BGR to HSV
hsv_img= cv.cvtColor(img, cv.COLOR_BGR2HSV)

#Convert BGR to LAB
lab_img= cv.cvtColor(img, cv.COLOR_BGR2LAB)

#Convert BGR to Grayscale
gray_img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow("BGR Image", img)
cv.imshow("RGB Image", rgb_img)
cv.imshow("HSV Image", hsv_img)
cv.imshow("LAB Image", lab_img)
cv.imshow("Grayscale Image", gray_img)

cv.waitKey(0)
cv.destroyAllWindows()

#Task 10
blurred = cv.blur(rgb_img, (5, 5))

#normal sharpening
kernel1= np.array([ [0, -1, 0],[-1, 5, -1],[0, -1, 0]])

#stronger sharpening
kernel2= np.array([[0, -2, 0],[-2, 9, -2],[0, -2, 0]])

#apply sharpening to BLURRED images

sharpened1= cv.filter2D(blurred, -1, kernel1)
sharpened2= cv.filter2D(blurred, -1, kernel2)

cv.imshow("Original", img)
cv.imshow("Blurred 5x5", blurred)
cv.imshow("Sharpened - Strength 1", sharpened1)
cv.imshow("Sharpened - Strength 2", sharpened2)


#The 5×5 blur makes the image smoother and less detailed. 
#Sharpening restores the edges, while stronger sharpening makes them more noticeable.

def add_salt_pepper_noise(image, density=0.05): #density controls how much noise i add, 0.05 means that 5% of pixals will become noise
    noisy = image.copy() #to modify the copy and dont change the original image

    total_pixels = image.shape[0] #height * image.shape[1] #width

    num_noise = int(total_pixels * density)

    rows = np.random.randint(0, image.shape[0], num_noise) 
    cols = np.random.randint(0, image.shape[1], num_noise)
    #these random rows and columns are going to get modified 

    half = num_noise // 2
    #half->salt, and other half->black

    noisy[rows[:half], cols[:half]] = 255 #add salt (changed pixels to white)
    noisy[rows[half:], cols[half:]] = 0 #add pepper (changed pixels to black)

    return noisy

noisy_img = add_salt_pepper_noise(img, 0.05)

cv.imshow("Original", img)
cv.imshow("salt and pepper noise", noisy_img)

#Task 12
median3= cv.medianBlur(noisy_img, 3)
median5= cv.medianBlur(noisy_img, 5)
median7= cv.medianBlur(noisy_img, 7)

cv.imshow("Noisy Image", noisy_img)
cv.imshow("Median Filter 3x3", median3) #removes some noises and preserve details of the image
cv.imshow("Median Filter 5x5", median5) #removes more noise but causes more smoothing
cv.imshow("Median Filter 7x7", median7) #remove image details and causes very strong and noticable smoothing

cv.waitKey(0)
cv.destroyAllWindows()











