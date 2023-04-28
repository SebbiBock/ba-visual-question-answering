import random
import sys
import os

import pygaze.libtime as timer

from constants import *
from pygaze.libscreen import Display, Screen
from pygaze.libinput import Keyboard
from pygaze.eyetracker import EyeTracker
from pygaze.liblog import Logfile

# Hacky solution for parent importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import loader
from textbox import TextInputBox
from postprocessing import convert_edf_to_dataframe
from experiment_util import *


### SETUP ###

# Create Screen and Display
disp = Display()
scr = Screen()

# Tuple for displaying
center_screen_x = int(scr.dispsize[0] / 2)
center_screen_y = int(scr.dispsize[1] / 2)

# Create the keyboard for inputs
kb = Keyboard()

# Get the group for the given participant
participant_group = get_participant_group()

# Get the directory to the images for the given group and test images
image_dir = get_group_image_directory(participant_group)
test_dir = get_test_image_directory()


### VP-CODE + LOGGING ###

vp_code = TextInputBox(
	scr,
	disp,
	FGC,
	instruction="Enter your VP-Code",
	supertext=VP_INSTRUCTIONS,
	caps=True
).main_loop()

# Create and get output directory path
participant_string = create_participant_string(vp_code)
output_dir = create_and_return_output_directory(participant_string)

# Initialize the EyeTracker-Class and set its output file to the participant folder
tracker = EyeTracker(disp, data_file=f"exp_output/{participant_string}/tracker.edf")

# Create the logging file with the VP Code and the timestamp string
log = Logfile(filename=os.path.join(output_dir, "logger"))
log.write([
"trial", "question_id", "time_on_image", "answer", "bb_image_x_min", "bb_image_x_max", "bb_image_y_min, bb_image_y_max"
])



### LOAD DATA ###

# Read all image names and shuffle their order to avoid sequencing effects
images = os.listdir(image_dir)
random.shuffle(images)

# Get question ids and questions to images
question_ids = [str(x).split(".")[0] for x in images]
questions = loader.load_questions(question_ids)

# Load test data
test_images = os.listdir(test_dir)
test_questions = loader.load_questions([str(y).split(".")[0] for y in test_images])



### DEMOGRAPHIC DATA ###

# Create input box for the gender where only three letters are valid and get its input
gender = TextInputBox(
	scr,
	disp,
	FGC,
	instruction="Please enter your gender (m/w/d)",
	key_list=["m", "w", "d"]
).main_loop()

# Create input box for the age where only numbers are valid and get its input
age = TextInputBox(
	scr,
	disp,
	FGC,
	instruction="Please enter your age",
	key_list=[str(x) for x in range(0, 10)]
).main_loop()

# Save demographic data
save_demographic_data(output_dir, age=age, gender=gender, group=participant_group)



### CALIBRATION ###

scr.clear()
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



### TEST RUNS ###

# Clear the screen
for test_question, test_image in zip(test_questions, test_images):

	# Clear the screen
	scr.clear()

	# Present question and let the participant view it as long as they want
	scr.draw_text(text=f"Question: {test_question}", fontsize=TEXTSIZE)
	scr.draw_text(text="Press any key to view the image...", fontsize=TEXTSIZE - 1, pos=(center_screen_x, center_screen_y + 250), centre=True)
	disp.fill(scr)
	disp.show()

	# Continue with any button press and clear the screen afterwards
	kb.get_key(keylist=None, timeout=None, flush=True)
	scr.clear()

	# Get image scale
	test_scale = get_image_scale(os.path.join(test_dir, test_image), scr.dispsize)

	# Present the image
	scr.draw_image(os.path.join(test_dir, test_image), scale=test_scale)
	disp.fill(scr)
	disp.show()

	# Wait for the participant to continue
	kb.get_key(keylist=None, timeout=None, flush=True)

	# Create input box for the answer
	test_answer = TextInputBox(
		scr,
		disp,
		FGC,
		instruction="Your answer",
	).main_loop()

# Show message indicating that real experiment will now start
scr.clear()
scr.draw_text(text=EXPERIMENT_START_TEXT, fontsize=TEXTSIZE)
disp.fill(scr)
disp.show()

# wait for a keypress
kb.get_key(keylist=None, timeout=None, flush=True)



### MAIN EXPERIMENT ###

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

	# Get image scale
	scale = get_image_scale(os.path.join(image_dir, image), scr.dispsize)

	# perform a drift check
	tracker.drift_correction()

	# Start tracking and log some stuff
	tracker.start_recording()
	tracker.log("TRIALSTART %d" % trial_nr)
	tracker.log("IMAGENAME %s" % image)
	tracker.status_msg(f"Starting trial {trial_nr + 1} / {len(images)}")

	# Present the image
	scr.draw_image(os.path.join(image_dir, image), scale=scale)

	# Get image bboxes: For the maximum, add 0.5 so it rounds up in doubt, for the minimum int automatically rounds down
	x_min = center_screen_x - int(scr.screen[-1].size[0] / 2)
	x_max = center_screen_x + int(scr.screen[-1].size[0] / 2 + 0.5)
	y_min = center_screen_y - int(scr.screen[-1].size[1] / 2)
	y_max = center_screen_y + int(scr.screen[-1].size[1] / 2 + 0.5)

	# Draw fixations for debug to assure proper bounding box calculation. TODO: Remove once assured!
	scr.draw_fixation(pos=(x_min, y_min), color="red")
	scr.draw_fixation(pos=(x_max, y_max), color="blue")

	# Actually show the image and log
	disp.fill(scr)
	t0 = disp.show()
	tracker.log("IMGON at %d" % t0)


	# Wait for the participant to continue
	kb.get_key(keylist=None, timeout=None, flush=True)

	# stop recording
	tracker.log("TRIALEND %d" % trial_nr)
	tracker.stop_recording()

	# Reset screen to blank
	disp.fill()
	t1 = disp.show()
	tracker.log("IMGOFF %d" % t1)

	# Create input box for the answer
	answer = TextInputBox(
		scr,
		disp,
		FGC,
		instruction="Your answer",
	).main_loop()

	# Write to log
	log.write([trial_nr, question_id, t1-t0, answer, x_min, x_max, y_min, y_max])

	# inter trial interval
	timer.pause(ITI)



### CLEAN-UP ###

# Loading message
scr.clear()
scr.draw_text(text="Transferring the data file, please wait...", fontsize=TEXTSIZE)
disp.fill(scr)
disp.show()

# neatly close connection to the tracker
# (this will close the data file, and copy it to the stimulus PC)
tracker.close()

# Close the logfile
log.close()

# Show final message
scr.clear()
scr.draw_text(text=EXIT_TEXT, fontsize=TEXTSIZE)
disp.fill(scr)
disp.show()

# wait for a keypress
kb.get_key(keylist=None, timeout=None, flush=True)

# close the Display
disp.close()

# If not in dummy mode: run the postprocessing script to convert the files
if not DUMMYMODE:
	convert_edf_to_dataframe(participant_string)
