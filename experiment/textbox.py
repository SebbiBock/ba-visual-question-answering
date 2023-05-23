import pygaze
import psychopy as pspy

from typing import List
from constants import TEXTSIZE


class TextInputBox(object):
    """
        Class to draw a text input box.
    """

    def __init__(self,
                 scr,
                 disp,
                 text_color,
                 instruction,
                 subtext: str = "Press Enter to continue...",
                 supertext: str = None,
                 key_list: List[str] = None,
                 caps: bool = False
                 ) -> None:
        """
            Constructor for the text box.

            :param scr: The screen to draw upon
            :param disp: The display to render to
            :param text_color: The color of the text
            :param instruction: The instructions to display before the text input
            :param subtext: Optional subtext to display under the textbox
            :param supertext: Optional supertext to display over the textbox
            :param key_list: List of string characters that are allowed for use
            :param caps: Whether all letters should be capitalized
        """

        # Initialize attributes
        self.scr = scr
        self.disp = disp
        self.textbox = pspy.visual.TextStim(pygaze.expdisplay, color=text_color, height=TEXTSIZE)
        self.text = ''
        self.instruction = instruction
        self.caps = caps

        # Create sub and super text boxes
        self.subtext_box = pspy.visual.TextStim(pygaze.expdisplay, subtext, pos=(0, -220), color=text_color, height=TEXTSIZE - 1) if subtext is not None else None
        self.supertext_box = pspy.visual.TextStim(pygaze.expdisplay, supertext, pos=(0, 220), color=text_color, height=TEXTSIZE - 1) if supertext is not None else None

        # Make font size of the subtext smaller
        if self.subtext_box is not None:
            self.subtext_box.size -= (1, 1)

        # Default key List: low-case, upper-case and numbers
        if key_list is None:
            self.key_list = [chr(x) for x in range(65, 90 + 1)]
            self.key_list.extend([chr(y) for y in range(97, 122 + 1)])
            self.key_list.extend([chr(z) for z in range(48, 57 + 1)])
        else:
            self.key_list = key_list

        # Flush keyboard
        pspy.event.clearEvents(eventType="keyboard")

    def main_loop(self):
        """
            Runs the main loop for the Textbox. This terminates once enter is pressed.
        """

        while True:

            # Clear the screen
            self.scr.clear()

            # Set textbox text to currently entered text
            self.textbox.setText(f'{self.instruction}: {self.text}')

            # Append all textboxes to the screen
            self.scr.screen.append(self.textbox)

            if self.subtext_box is not None:
                self.scr.screen.append(self.subtext_box)
            if self.supertext_box is not None:
                self.scr.screen.append(self.supertext_box)

            # Fill the display with the screen and show
            self.disp.fill(self.scr)
            self.disp.show()

            # myWindow.flip()

            # If any of the keys is pressed, append it to the answer
            for letter in self.key_list:
                if pspy.event.getKeys([letter]):

                    # Upper-case if any shift is pressed
                    if pspy.event.getKeys(["lshift", "rshift", "capslock"]) or self.caps:
                        letter = letter.upper()

                    # Append to answer
                    self.text += letter

            # Add whitespace if space is pressed
            if pspy.event.getKeys(['space']):
                self.text += " "

            # Delete the last entry of the answer if the backspace is pressed
            if pspy.event.getKeys(['backspace']):
                self.text = self.text[:-1]

            # Submit the answer if the enter key is pressed
            if pspy.event.getKeys(['return']):
                return self.text
