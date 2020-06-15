import cv2
IMG_SIZE = (220,66)

def image_preprocess(image: str):
    """
        Resive image, and crop top black area from the image
    """
    if type(image) == str:
        image = cv2.imread(image)
        
    image_cropped = image[25:375, :]
    
    image = cv2.resize(image_cropped, IMG_SIZE, interpolation = cv2.INTER_AREA)
    
    return image

def frame_capture(path, dest):
    """
        Extract frames from video
        
        path - path to video files
        dest - destination path for extracted images
    """
      
    # Path to video file 
    cap = cv2.VideoCapture(path)
    
    # Total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # cap object calls read 
        # function extract frames 
        success, image = cap.read()
        
        if image is not None:
            image = image_preprocess(image)
            # Saves the frames with frame-count 
            cv2.imwrite(f"{dest}/frame{count}.jpg", image)
        else:
            break
  
        count += 1
    percentage = int(count/frame_count) * 100
    return f'{count}/{frame_count} ({percentage})% of frames extracted'

frame_capture('data/train/train.mp4', 'data/train')
frame_capture('../data/test/test.mp4', '../data/test')