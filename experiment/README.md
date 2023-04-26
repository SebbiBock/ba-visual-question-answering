### Run the experiment:

In order to run the experiment, you need to create an environment and install the requirements provided in the requirements.txt
in this directory. Furthermore, in data.loader, you need to adapt the paths
to point to the proper directions of where the data is stored.

### Configure Lab Setup:

#### Start-Up:
1) Plug Power Cable into EyeTracker
2) EyeTracker-PC: Boot-Up
3) EyeTracker-PC: Select User "eyetracker"
4) EyeTracker-PC: Go to File-Manager -> Configuration -> Screen settings
5) EyeTracker-PC: Enter Screen Dimensions and Resolution, if they have changed. Default resolution: 2560, 1440; Default screen dimensions: 59.5, 33.5.
6) Display-PC: Connect it to a power source, the monitor and the ethernet (LAN cable)
7) Display-PC: Boot-Up

#### Participant:

First, make sure that the participant is not wearing any glasses!

1) Determine the dominant eye of the participant: Place them at the window and make them look toward a specific point
on the shelf. Then, they should form a triangle with their hands around the point and move it closer towards them. The eye it lands on is their dominant eye.
If it stays in the middle, just pick the right eye.
2) Let the participant sit in the chair and adjust the chair as well as the headrest to their preference. However, make sure that their eye leyel is at the top 25% of the monitor.
3) Make sure that the EyeTracker, Headrest and Monitor are placed central (so in a line) from the viewpoint of the participant.
Also, make sure that the EyeTracker is not in their line of sight to the monitor, but as close to the monitor as possible.
4) Measure the necessary distances and enter them into the settings in the EyeTracker-PC.
5) Carefully remove the cap from the EyeTracker.
6) Adjust the pupil size so that the pupil is filled as much as possible without any other dark blue in the eyeball region
7) Adjust the corneal reflection (CR) so that the corneal reflection is close to the pupil with only one cyan area in the eyeball area
8) Start the experiment.

#### Calibration:
1) The calibration view will open itself once the experiment runs.
2) Tell the participant what the calibration will look like: First, look at the center dot, once you're ready, press space. Points will appear
in the corners of the screen, your task is to fixate on these points to the best of your ability.
3) Validation: Same thing.
4) If the results are not good, simply re-do the calibration.
5) Noise correction: Simply focus on the stimulus.
6) Done. Now, the experiment can start.

### Add experiment data

To add own experiment data you need to create a directory called "images" in the same parent directory as the experiment
file is in. Here, insert the images you want to display, and name them according to the corresponding question ID (VQAv2),
for example: `109945002.jpg`. The script will then automatically load in the rest.