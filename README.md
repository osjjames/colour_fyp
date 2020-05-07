# Automatic Video Colourisation Through Spatial Colour Adjustment

## Information

This is an automatic video colourisation application created by Oliver James (oj247@bath.ac.uk) for the University of Bath Department of Computer Science final year project.

A ZIP file containing some examples of output videos from this application can be downloaded at https://drive.google.com/file/d/1ZcnuGgaDEmt1wGRL0OOzSOzfFHchgjA4/view?usp=sharing .

## Get Started

1. Open a bash terminal and clone this Github repository to your local machine.

`git clone https://github.com/osjjames/colour_fyp.git`

2. Navigate to the repository.

`cd colour_fyp`

3. Build and run the Docker container (this may take a few minutes). 

`docker-compose build && docker-compose up`

4. Open a web browser and navigate to `localhost:8888`. You may change the port mapping in the `.env` file if you wish.

5. Enter the password `test`. Click on the file `getStarted.ipynb`. You are now in a Jupyter notebook running the application.

6. To test the application, place a video file in the `colour_fyp/videos` directory. Set the `video_name` variable in the notebook to the name of the video file (e.g. "test_video.mp4"). Finally, click "Run" at the top of the page. When completed, the output video will appear in the `colour_fyp/videos` directory. 
