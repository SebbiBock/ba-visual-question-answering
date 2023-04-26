import os


### FILES AND FOLDERS ###

# the DIR constant contains the full path to the current directory, which will
# be used to determine where to store and retrieve data files
DIR = os.path.dirname(__file__)

# the DATADIR is the path to the directory where data files will be stored
DATADIR = os.path.join(DIR, 'exp_output')

# the IMGDIR is the path to the directory that contains the image files
IMGDIR = os.path.join(DIR, 'images')

# For all necessary directories, create them if they do not yet exist
for d in [DATADIR, IMGDIR]:
    if not os.path.isdir(d):
        os.mkdir(d)



### DISPLAY ###

# for the DISPTYPE, you can choose between 'pygame' and 'psychopy'; go for
# 'psychopy' if you need millisecond accurate display refresh timing, and go
# for 'pygame' if you experience trouble using PsychoPy
DISPTYPE = 'psychopy'

# the DISPSIZE is the monitor resolution, e.g. (1024,768)
DISPSIZE = (2560, 1440)

# the SCREENSIZE is the physical screen size in centimeters. LAB: (59.5, 33.5).
SCREENSIZE = (59.5, 33.5)

# the SCREENDIST is the distance in centimeters between the participant and the display. LAB: 100.
SCREENDIST = 100.0

# Whether to use a Fullscreen or not
FULLSCREEN = True

# If the mouse should be visible
MOUSEVISIBLE = False

# BGC is for BackGroundColour, FGC for ForeGroundColour; both are RGB guns,
# which contain three values between 0 and 255, representing the intensity of
# Red, Green, and Blue respectively, e.g. (0,0,0) for black, (255,255,255) for
# white, or (255,0,0) for the brightest red
BGC = (0, 0, 0)
FGC = (255, 255, 255)

# the TEXTSIZE determines the size of the text in the experiment
TEXTSIZE = 20

# TIMING
# the intertrial interval (ITI) is the minimal amount of time between the
# presentation of two consecutive images
ITI = 1500  # ms



### EYE TRACKING ###

# the TRACKERTYPE indicates the brand of eye tracker, and should be one of the
# following: 'eyelink', 'smi', 'tobii' 'dumbdummy', 'dummy'
TRACKERTYPE = 'eyelink'

# the EYELINKCALBEEP constant determines whether a beep should be sounded on
# the appearance of every calibration target (EyeLink only)
EYELINKCALBEEP = False

# set DUMMYMODE to True if no tracker is attached
DUMMYMODE = False



### INSTRUCTIONS ###

INSTRUCTIONS = """
Your task is that of Visual Question Answering (VQA):
\n
You will be presented a series of X Image-Question pairs, and you have to answer the question based on the given image.\n
First, you'll be shown the question that you can read as long as you want.\n
With the press of any key, the question will disappear, and the corresponding image to the question will appear.\n
You can view the image for as long as you want.\n
Once you want to answer the given question, simply press another key which will lead you to an input field for your answer.\n
Once you entered your answer, this repeats until every Image-Question pair has been completed.\n
\n
Ready? Press any key to start the experiment...
"""

CALIBRATION_INSTRUCTIONS = """
Now, we will calibrate you to the EyeTracker.\n
For this, we will xyz.\n
\n
Press any key to start the calibration...
"""

VP_INSTRUCTIONS = """
First, please generate your VP-Code. Follow these three steps and concatenate the results to create your VP-Code:\n
1) Take the first two letters of your mother's first name. Example: Eva -> EV\n
2) Take the last two digits of the year you were born. Example: 1998 -> 98\n
3) Take the last two letters of the city you were born in. Example: Frankfurt -> RT\n
Then, simply concatenate the result and enter it below. Full Example: EV98RT\n
"""

EXIT_TEXT = """
This is the end of the experiment. Thank you for participating!\n
Press any key to stop the experiment.\n
"""
