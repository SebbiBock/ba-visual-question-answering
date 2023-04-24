import sys
import os
import random

# Hacky solution for parent importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import loader

import pygaze.libtime as timer

from constants import *
from pygaze.libscreen import Display, Screen
from pygaze.libinput import Keyboard
from pygaze.eyetracker import EyeTracker
from pygaze.liblog import Logfile



### SETUP ###

# visuals
disp = Display()
scr = Screen()

# Tuple for displaying
center_screen_x = int(scr.dispsize[0] / 2)
center_screen_y = int(scr.dispsize[1] / 2)

# input
kb = Keyboard()
tracker = EyeTracker(disp)

# output
log = Logfile()
log.write(["trialnr","image","imgtime"])



### LOAD DATA ###

# read all image names
images = os.listdir(IMGDIR)

# Get question ids and questions to images
question_ids = [str(x).split(".")[0] for x in images]
questions = loader.load_questions(question_ids)



### CALIBRATION ###

scr.draw_text(text=CALIBRATION_INSTRUCTIONS, fontsize=TEXTSIZE)
disp.fill(scr)
disp.show()

# wait for a keypress
kb.get_key(keylist=None, timeout=None, flush=True)

# calibrate the eye tracker
tracker.calibrate()



### INSTRUCTIONS ###

scr.clear()
scr.draw_text(text=INSTRUCTIONS, fontsize=TEXTSIZE)
disp.fill(scr)
disp.show()

# wait for a keypress
kb.get_key(keylist=None, timeout=None, flush=True)

for trial_nr, (image, question, question_id) in enumerate(zip(images, questions, question_ids)):

	# Clear the screen
	scr.clear()

	# Present question and let the participant view it as long as they want
	scr.draw_text(text=f"Question: {question}", fontsize=TEXTSIZE)
	scr.draw_text(text="Press any key to view the image...", fontsize=TEXTSIZE - 1, pos=(center_screen_x, center_screen_y + 250), centre=True)
	disp.fill(scr)
	disp.show()

	# Continue with any button press and clear the screen afterwards
	kb.get_key(keylist=None, timeout=None, flush=True)
	scr.clear()

	# perform a drift check
	tracker.drift_correction()

	# Start tracking and log some stuff
	tracker.start_recording()
	tracker.log("TRIALSTART %d" % trial_nr)
	tracker.log("IMAGENAME %s" % image)
	tracker.status_msg(f"Starting trial {trial_nr + 1} / {len(images)}")

	# present image
	scr.draw_image(os.path.join(IMGDIR, image))
	disp.fill(scr)
	t0 = disp.show()
	tracker.log("image online at %d" % t0)
	
	# Wait for the participant to continue
	kb.get_key(keylist=None, timeout=None, flush=True)

	# stop recording
	tracker.log("TRIALEND %d" % trial_nr)
	tracker.stop_recording()

	# reset screen
	disp.fill()
	t1 = disp.show()
	tracker.log("image offline at %d" % t1)
	
	# TRIAL AFTERMATH
	# bookkeeping
	log.write([trial_nr, image, t1-t0])
	
	# inter trial interval
	timer.pause(ITI)


# # # # #
# CLOSE

# loading message
scr.clear()
scr.draw_text(text="Transferring the data file, please wait...", fontsize=TEXTSIZE)
disp.fill(scr)
disp.show()

# neatly close connection to the tracker
# (this will close the data file, and copy it to the stimulus PC)
tracker.close()

# close the logfile
log.close()

# exit message
scr.clear()
scr.draw_text(text="This is the end of this experiment. Thank you for participating!\n\n(press any key to exit)", fontsize=TEXTSIZE)
disp.fill(scr)
disp.show()

# wait for a keypress
kb.get_key(keylist=None, timeout=None, flush=True)

# close the Display
disp.close()
