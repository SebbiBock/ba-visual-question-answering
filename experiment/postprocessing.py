"""
    File containing the necessary postprocessing steps to convert the .edf file into usable format
"""

import copy
import numpy
import pandas as pd
import pickle
import subprocess
import os
import os.path

from typing import Any, Dict, List, Tuple, Union


def replace_missing(value, missing=0.0):
    """

    TAKEN FROM THE OFFICIAL PYGAZEANALYZER

    Returns missing code if passed value is missing, or the passed value
    if it is not missing; a missing value in the EDF contains only a
    period, no numbers; NOTE: this function is for gaze position values
    only, NOT for pupil size, as missing pupil size data is coded '0.0'

    arguments
    value		-	either an X or a Y gaze position value (NOT pupil
                    size! This is coded '0.0')

    keyword arguments
    missing		-	the missing code to replace missing data with
                    (default = 0.0)

    returns
    value		-	either a missing code, or a float value of the
                    gaze position
    """

    if value.replace(' ', '') == '.':
        return missing
    else:
        return float(value)


def read_edf(filename, start, stop=None, missing=0.0, debug=False):
    """Returns a list with dicts for every trial. A trial dict contains the
    following keys:
        x		-	numpy array of x positions
        y		-	numpy array of y positions
        size		-	numpy array of pupil size
        time		-	numpy array of timestamps, t=0 at trialstart
        trackertime	-	numpy array of timestamps, according to EDF
        events	-	dict with the following keys:
                        Sfix	-	list of lists, each containing [starttime]
                        Ssac	-	list of lists, each containing [starttime]
                        Sblk	-	list of lists, each containing [starttime]
                        Efix	-	list of lists, each containing [starttime, endtime, duration, endx, endy]
                        Esac	-	list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
                        Eblk	-	list of lists, each containing [starttime, endtime, duration]
                        msg	-	list of lists, each containing [time, message]
                        NOTE: timing is in EDF time!

    arguments
    filename		-	path to the file that has to be read
    start		-	trial start string

    keyword arguments
    stop			-	trial ending string (default = None)
    missing		-	value to be used for missing data (default = 0.0)
    debug		-	Boolean indicating if DEBUG mode should be on or off;
                if DEBUG mode is on, information on what the script
                currently is doing will be printed to the console
                (default = False)

    returns
    data			-	a list with a dict for every trial (see above)
    """

    # # # # #
    # debug mode

    if debug:
        def message(msg):
            print(msg)
    else:
        def message(msg):
            pass

    # # # # #
    # file handling

    # check if the file exists
    if os.path.isfile(filename):
        # open file
        message("opening file '%s'" % filename)
        f = open(filename, 'r')
    # raise exception if the file does not exist
    else:
        raise Exception("Error in read_edf: file '%s' does not exist" % filename)

    # read file contents
    message("reading file '%s'" % filename)
    raw = f.readlines()

    # close file
    message("closing file '%s'" % filename)
    f.close()

    # # # # #
    # parse lines

    # variables
    data = []
    x = []
    y = []
    size = []
    time = []
    trackertime = []
    events = {'Sfix': [], 'Ssac': [], 'Sblk': [], 'Efix': [], 'Esac': [], 'Eblk': [], 'msg': []}
    starttime = 0
    started = False
    trialend = False
    finalline = raw[-1]

    # loop through all lines
    for line in raw:

        # check if trial has already started
        if started:
            # only check for stop if there is one
            if stop != None:
                if stop in line:
                    started = False
                    trialend = True
            # check for new start otherwise
            else:
                if (start in line) or (line == finalline):
                    started = True
                    trialend = True

            # # # # #
            # trial ending

            if trialend:
                message("trialend %d; %d samples found" % (len(data), len(x)))
                # trial dict
                trial = {}
                trial['x'] = numpy.array(x)
                trial['y'] = numpy.array(y)
                trial['size'] = numpy.array(size)
                trial['time'] = numpy.array(time)
                trial['trackertime'] = numpy.array(trackertime)
                trial['events'] = copy.deepcopy(events)
                # add trial to data
                data.append(trial)
                # reset stuff
                x = []
                y = []
                size = []
                time = []
                trackertime = []
                events = {'Sfix': [], 'Ssac': [], 'Sblk': [], 'Efix': [], 'Esac': [], 'Eblk': [], 'msg': []}
                trialend = False

        # check if the current line contains start message
        else:
            if start in line:
                message("trialstart %d" % len(data))
                # set started to True
                started = True
                # find starting time
                starttime = int(line[line.find('\t') + 1:line.find(' ')])

        # # # # #
        # parse line

        if started:
            # message lines will start with MSG, followed by a tab, then a
            # timestamp, a space, and finally the message, e.g.:
            #	"MSG\t12345 something of importance here"
            if line[0:3] == "MSG":
                ms = line.find(" ")  # message start
                t = int(line[4:ms])  # time
                m = line[ms + 1:]  # message
                events['msg'].append([t, m])

            # EDF event lines are constructed of 9 characters, followed by
            # tab separated values; these values MAY CONTAIN SPACES, but
            # these spaces are ignored by float() (thank you Python!)

            # fixation start
            elif line[0:4] == "SFIX":
                message("fixation start")
                l = line[9:]
                events['Sfix'].append(int(l))
            # fixation end
            elif line[0:4] == "EFIX":
                message("fixation end")
                l = line[9:]
                l = l.split('\t')
                st = int(l[0])  # starting time
                et = int(l[1])  # ending time
                dur = int(l[2])  # duration
                sx = replace_missing(l[3], missing=missing)  # x position
                sy = replace_missing(l[4], missing=missing)  # y position
                events['Efix'].append([st, et, dur, sx, sy])
            # saccade start
            elif line[0:5] == 'SSACC':
                message("saccade start")
                l = line[9:]
                events['Ssac'].append(int(l))
            # saccade end
            elif line[0:5] == "ESACC":
                message("saccade end")
                l = line[9:]
                l = l.split('\t')
                st = int(l[0])  # starting time
                et = int(l[1])  # endint time
                dur = int(l[2])  # duration
                sx = replace_missing(l[3], missing=missing)  # start x position
                sy = replace_missing(l[4], missing=missing)  # start y position
                ex = replace_missing(l[5], missing=missing)  # end x position
                ey = replace_missing(l[6], missing=missing)  # end y position
                events['Esac'].append([st, et, dur, sx, sy, ex, ey])
            # blink start
            elif line[0:6] == "SBLINK":
                message("blink start")
                l = line[9:]
                events['Sblk'].append(int(l))
            # blink end
            elif line[0:6] == "EBLINK":
                message("blink end")
                l = line[9:]
                l = l.split('\t')
                st = int(l[0])
                et = int(l[1])
                dur = int(l[2])
                events['Eblk'].append([st, et, dur])

            # regular lines will contain tab separated values, beginning with
            # a timestamp, follwed by the values that were asked to be stored
            # in the EDF and a mysterious '...'. Usually, this comes down to
            # timestamp, x, y, pupilsize, ...
            # e.g.: "985288\t  504.6\t  368.2\t 4933.0\t..."
            # NOTE: these values MAY CONTAIN SPACES, but these spaces are
            # ignored by float() (thank you Python!)
            else:
                # see if current line contains relevant data
                try:
                    # split by tab
                    l = line.split('\t')
                    # if first entry is a timestamp, this should work
                    int(l[0])
                except:
                    message("line '%s' could not be parsed" % line)
                    continue  # skip this line

                # check missing
                if float(l[3]) == 0.0:
                    l[1] = 0.0
                    l[2] = 0.0

                # extract data
                x.append(float(l[1]))
                y.append(float(l[2]))
                size.append(float(l[3]))
                time.append(int(l[0]) - starttime)
                trackertime.append(int(l[0]))

    # # # # #
    # return

    return data


def convert_data_to_dataframe(participant_string: str) -> None:
    """
        Converts all data for the given participant into pd.DataFrames and saves them to the participant's output
        directory. Specifically, this method does the following postprocessing steps:

        (1) Converts the edf file created for the given participant string and converts it to asc (ASCII) using the
            edf2asc command from the EyeLink Developer Kit (which therefore needs to be installed).
        (2) The .asc file is converted into a dictionary containing all gaze data and events.
        (3) The dictionary is converted into a pd.DataFrame containing all gaze data for every trial.
        (4) Image coordinates are added and gaze data not concentrated on the image are removed.
        (5) An event dataframe is created.
        (6) The logger data is transformed into a usable pd.DataFrame.

        Finally, the produced outputs are saved into a /postprocessed/ dictionary in the given participant output
        folder using pickle.

        :param participant_string: Name of the output folder for a participant.
        :return: Nothing
    """

    # Assemble path
    path_to_edf_file = f"exp_output/{participant_string}/tracker.edf"
    path_to_asc_file = f"exp_output/{participant_string}/tracker.asc"
    path_to_postproc_dir = f"exp_output/{participant_string}/postprocessed/"

    # Check whether the edf file even exists
    if not os.path.isfile(path_to_edf_file):
        raise Exception("No edf file could be found under the specified path.")

    # Run the edf2asc command for the given participant output folder if it does not yet exist
    if not os.path.isfile(path_to_asc_file):
        subprocess.run(["edf2asc", path_to_edf_file], shell=True)

    # Convert the asc file to a dictionary
    data = read_edf(path_to_asc_file, start="IMGON", stop="TRIALEND")

    # Read in logger data
    logger_df = read_logger_output(participant_string)

    # Fill list of dataframes to concatenate and list of event dictionaries
    trial_df_list = []
    event_df_list = []

    # Loop through every trial and compute the pd.DataFrame for it
    for trial, trial_dict in enumerate(data):

        # Get respective logger slice with bounding boxes and question id
        x_min, x_max, y_min, y_max, q_id = logger_df[logger_df["trial"] == trial][
            ["bb_image_x_min", "bb_image_x_max", "bb_image_y_min", "bb_image_y_max", "question_id"]
        ].values[0]

        # Construct the dataframe for this trial
        trial_df, event_df = create_trial_and_event_dict_from_dict(trial_dict, trial=trial, question_id=q_id)

        # Filer out gaze data not on the image
        trial_df = filter_for_gaze_on_image(trial_df, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        # Append the image coordinate df
        trial_df_list.append(compute_image_coordinates_in_df(trial_df, x_min, y_min))

        # Append the event df while also adding image coordinates
        event_df_list.append(compute_image_coordinates_in_df(
            event_df,
            x_min,
            y_min,
            existing_cols=["x_start", "y_start", "x_end", "y_end"],
            generated_cols=["image_x_start", "image_y_start", "image_x_end", "image_y_end"])
        )

    # Concatenate all dataframes to one
    final_gaze_df = pd.concat(trial_df_list, ignore_index=True)
    final_event_df = pd.concat(event_df_list, ignore_index=True)

    # Create output dir if it does not exist
    if not os.path.isdir(path_to_postproc_dir):
        os.mkdir(path_to_postproc_dir)

    # Save all output files
    for obj, f_name in zip([final_gaze_df, final_event_df, logger_df], ["gaze_df", "event_df", "logger_df"]):
        with open(os.path.join(path_to_postproc_dir, f"{f_name}.pkl"), "wb") as f:
            pickle.dump(obj, f)


def read_logger_output(participant_string: str) -> pd.DataFrame:
    """
        Reads and converts the logger file containing the participant answers and bounding boxes into
        a pd.DataFrame.

        :param participant_string: The participant to load the data from
        :return: pd.DataFrame containing the logger data
    """

    # Assemble path and load in the data using pandas and return
    return pd.read_csv(f"exp_output/{participant_string}/logger.txt", sep="\t", index_col=False)


def create_trial_and_event_dict_from_dict(
        trial_dict: Dict[str, Union[Dict[str, Any], List[Any], numpy.ndarray]],
        trial: int,
        question_id: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        For the given trial, the gaze data is extracted and put into a pd.DataFrame along with identifiers for the
        question. Furthermore, the event dictionary is extracted and identifier are inserted as well. Finally,
        both are returned as a tuple.

        :param trial_dict: The dictionary containing the trial data.
        :param trial: Int identifier for the trial number
        :param question_id: Int question identifiers
        :return: Tuple of (trial_df, event_df)
    """

    # Construct the pd.DataFrame
    df = pd.DataFrame([trial_dict["x"], trial_dict["y"], trial_dict["size"], trial_dict["trackertime"]]).transpose()
    df.columns = ["screen_x", "screen_y", "pupil_size", "tracker_time"]

    # Add trial and question id
    df.insert(loc=0, column='trial_id', value=trial)
    df.insert(loc=0, column="question_id", value=question_id)

    # Get event dicts per event and convert them into a pd.DataFrame and store the kind of event
    # Note: Three kinds of events are tracked: Fixations, Saccades and Blinks. They are either stored with S[event] e.g.
    # Sfix or with E[event], e.g. Efix. The E-Storage has more information, namely event start and end, duration, x and
    # y. The S-Storage seems to only have the fixation start as information.
    event_dfs = []

    for event_name, event in zip(["fixation", "saccade", "blink"], ["Efix", "Esac", "Eblk"]):

        # Create dataframe with proper columns
        col_names = ["start", "end", "duration", "x_start", "y_start"]
        if event_name == "saccade":
            col_names.extend(["x_end", "y_end"])

        ev_df = pd.DataFrame(trial_dict["events"][event], columns=col_names)

        # Insert end coordinates and set them to start coordinates if the current event is not a saccade, since it
        # has this data on its own.
        if event_name != "saccade":
            ev_df["x_end"] = ev_df["x_start"]
            ev_df["y_end"] = ev_df["y_start"]

        # Add event identifier and add to list of events
        ev_df["event"] = event_name
        event_dfs.append(ev_df)

    # Concatenate them and add identifiers
    event_df = pd.concat(event_dfs, ignore_index=True)
    event_df.insert(loc=0, column='trial_id', value=trial)
    event_df.insert(loc=0, column="question_id", value=question_id)

    return df, event_df


def compute_image_coordinates_in_df(
        df: pd.DataFrame,
        x_min: int,
        y_min: int,
        existing_cols: List[str] = ["screen_x", "screen_y"],
        generated_cols: List[str] = ["image_x", "image_y"],
) -> pd.DataFrame:
    """
        Computes the gaze coordinates on the image from the screen coordinates and the bounding boxes of the
        image. The columns on which this action is to be performed can be defined individually.

        :param df: The pd.DataFrame containing the data for the given trial
        :param x_min: The minimum x coordinate for the bounding box of the given image
        :param y_min: The minimum y coordinate for the bonding box of the given image
        :param existing_cols: The columns that should be transformed.
        :param generated_cols: The corresponding new column names.
        :return: The modified pd.DataFrame
    """

    # For every given column, extract the proper value
    for ex_col, gen_col in zip(existing_cols, generated_cols):

        # Sanity check: Only either one x or one y must be in any of the existing columns
        assert (ex_col.count("x") == 1) ^ (ex_col.count("y") == 1), "Incorrect column names! Exactly one x or y!"

        df[gen_col] = df[ex_col] - (x_min if "x" in ex_col else y_min)

    return df


def filter_for_gaze_on_image(
        df: pd.DataFrame,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        x_col: str = "screen_x",
        y_col: str = "screen_y"
) -> pd.DataFrame:
    """
        Filters the given pd.DataFrame so that only gaze data on the image itself remains.

        :param df: The pd.DataFrame to consider
        :param x_min: The minimum x coordinate for the bounding box of the given image
        :param x_max: The maximum x coordinate for the bounding box of the given image
        :param y_min: The minimum y coordinate for the bonding box of the given image
        :param y_max: The maxmimum y coordinate for the bonding box of the given image
        :param x_col: The column name of the gaze x data
        :param y_col: The column name of the gaze y data
        :return: The modified pd.DataFrame with gaze data not on the image removed
    """

    # Check that screen x-y pairs are inside the bounding boxes and return the slice
    return df[
        (df[x_col] >= x_min) &
        (df[x_col] <= x_max) &
        (df[y_col] >= y_min) &
        (df[y_col] <= y_max)
    ]

