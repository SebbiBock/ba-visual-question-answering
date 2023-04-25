### Run the experiment:

In order to run the experiment, you need to create an environment and install the requirements provided in the requirements.txt
in this directory. Furthermore, in data.loader, you need to adapt the paths
to point to the proper directions of where the data is stored. In

Kontaktlinsen okay, Brille raus: Falls Kontaktlinsen, Dioptren eintragen
-> Sonst noch demographische Daten (Alter, Geschlecht, Feedback + Anmerkung)

### Configure Lab Setup:

#### Start-Up:
1) Plug Power Cable into EyeTracker
2) EyeTracker-PC: Boot-Up
3) EyeTracker-PC: Select User "eyetracker"
4) EyeTracker-PC: Go to File-Manager -> Configuration -> Screen settings
5) EyeTracker-PC: Enter Screen Dimensions and Resolution, if they have changed. Default: xyz
6) Display-PC: Connect it to a power source, the monitor and the ethernet (LAN cable)
7) Display-PC: Boot-Up
8) Start Experiment
9) Carefully remove the Cap from the EyeTracker

#### Participant:
1) Determine the dominant eye of the participant: Place them at the window and make them look toward a specific point
on the shelf. Then, they should form a triangle and move it closer towards them. The eye it lands on is their dominant eye.
If it stays in the middle, just pick the right eye.
2) Let the participant sit in the chair and adjust the chair as well as the headrest to their preference
3) Make sure that the EyeTracker, EyeRest and Monitor are placed central (so in a line) from the viewpoint of the participant.
Also, make sure that the EyeTracker is not in their line of sight.
4) Measure the distance between the lens of the .. and the ... and enter it into the settings in the EyeTracker-PC
5) Enter the settings menu and Adjust the pupil and corneal so that 


### Add experiment data

To add own experiment data you need to create a directory called "images" in the same parent directory as the experiment
file is in. Here, insert the images you want to display, and name them according to the corresponding question ID (VQAv2),
for example: `109945002.jpg`. The script will then automatically load in the rest.