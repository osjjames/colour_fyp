import cv2

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.') # Find OpenCV version

def get_fps(video):
  if int(major_ver)  < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
  else :
    fps = video.get(cv2.CAP_PROP_FPS)

  return fps

def get_dimensions(video):
  if int(major_ver)  < 3 :
    width  = video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
  else :
    width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

  return (int(width), int(height))

def setup_writer(video_path, save_path):
  video = cv2.VideoCapture(video_path)

  dimensions = get_dimensions(video)
  fps = get_fps(video)

  video.release()

  writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('D','I','V','X'), fps, dimensions)
  return writer