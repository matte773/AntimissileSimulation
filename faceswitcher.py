#!/usr/bin/env python3

from std_srvs.srv import Empty
from PIL import Image, ImageTk
import tkinter as tk
import os
from itertools import count

# ROS imports removed
# import rospy
# import roslib.packages

# Code adapted from https://stackoverflow.com/questions/40619731/display-animated-gif-in-tkinter-python-3-5
class GifLabel(tk.Label):
    """a label that displays images, and plays them if they are gifs"""

    def load(self, im, global_delay = None):
        """Loads new GIF and then removes the past frame if any"""
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()
        
        if isinstance(im, str):
            im = Image.open(im)
        next_frames = []
        new_delays = []
        # try to do this before unloading current image
        try:
            for i in count(1):
                # resizes each frame, but adds delay between signal and switch
                next_frames.append(
                    ImageTk.PhotoImage(im.resize((width,height)))
                    )
                try:
                    new_delays.append(im.info['duration'] if global_delay is None else global_delay)
                except:
                    new_delays.append(1)
                im.seek(i)
        except EOFError:
            pass
        outframes = next_frames.copy()


        self.frames = outframes
        self.loc = 0
        self.delays = new_delays

        #stop updating previous image
        if hasattr(self, 'after_call'):
            self.after_cancel(self.after_call)


        if len(self.frames) == 1:
            self.config(image=self.frames[0])
        else:
            self.next_frame()


    def next_frame(self):
        if self.frames:
            self.loc %= len(self.frames)
            self.config(image=self.frames[self.loc])
            delay = self.delays[self.loc]
            self.loc += 1
            self.after_call = self.after(delay, self.next_frame)

class FaceSwitcher:
    def __init__(self):
        # Load the GIFs
        # home_dir = os.path.expanduser("~")
        package_base_path = roslib.packages.get_pkg_dir('quori_osu')
        face_dir = os.path.join('quori_osu_supplemental', 'faces')
        self.default_face_path = os.path.join(package_base_path, face_dir, 'default_face.gif')
        self.thinking_face_path = os.path.join(package_base_path, face_dir, 'thinking_face.gif')
        self.talking_face_path = os.path.join(package_base_path, face_dir, 'talking_face.gif')
        self.current_image_path = self.default_face_path

        # Set up Tkinter
        self.root = tk.Tk()
        self.root.configure(bg="black")
        self.root.attributes('-fullscreen', True)  # Fullscreen mode
        # self.root.bind("f", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.toggle_fullscreen)
        # self.root.bind("<Escape>", lambda e: self.root.quit())  # Exit on Escape key
        self.root.bind("<Control-c>", lambda e: self.root.quit())
        self.root.attributes('-topmost', True) # keeps it on top permanently



        # Create a label to display the images
        self.label = GifLabel(self.root)
        self.label.configure(bg="black")
        self.label.pack(expand=True)
        self.label.load(self.current_image_path)

        # Set up ROS node
        rospy.init_node('face_switcher_node')

        # Create service servers
        rospy.Service('/default_face', Empty, self.show_default_face)
        rospy.Service('/thinking_face', Empty, self.show_thinking_face)
        rospy.Service('/talking_face', Empty, self.show_talking_face)

    def toggle_fullscreen(self, event=None):
        """Exit fullscreen mode."""
        self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen'))

    def show_default_face(self, req):
        if self.current_image_path == self.default_face_path:
            return []
        rospy.loginfo("Switching to default face")
        self.current_image_path = self.default_face_path
        self.update_display()
        return []

    def show_thinking_face(self, req):
        if self.current_image_path == self.default_face_path:
            rospy.loginfo("Switching to thinking face")
            self.current_image_path = self.thinking_face_path
            self.update_display()
        return []
    
    def show_talking_face(self, req):
        if self.current_image_path in [self.default_face_path, self.thinking_face_path]:
            rospy.loginfo("Switching to talking face")
            self.current_image_path = self.talking_face_path
            self.update_display()
        return []

    def update_display(self, global_delay=None):
        self.label.load(self.current_image_path, global_delay)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    face_switcher = FaceSwitcher()
    face_switcher.run()
