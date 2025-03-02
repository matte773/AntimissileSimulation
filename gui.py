#!/usr/bin/env python3

from std_srvs.srv import Empty, EmptyResponse
import sys
from threading import Thread
import tkinter as tk
import subprocess
import random
import time
import threading
import socket

# ROS imports removed
# import rospy
# from quori_osu.srv import GetQuestion, KeyID, KeyIDRequest


# Global variables
lastest_question = "Waiting for message..." # Default message
scale_type = "Likert" # Default scale type

# Global Flags 
use_UUID = True
demo_question = True
gen_question_key = True

# Color constants
LIGHT_BLUE = "#d4e1ff"
GREEN = "#8CD47E"
LIGHT_GREEN = "#C4E6C1"

class GuiApp:
    """Main GUI application class."""

    def __init__(self, root, question_service, key_id_service, question_label):
        """Initialize the GUI application."""
        self.root = root
        self.question_label = question_label

        self.root.title("ROS Noetic GUI")
        self.root.geometry("1280x720")

        self.root.configure(bg=LIGHT_BLUE)

        # Set the window to fullscreen
        self.root.attributes('-fullscreen', True)

        # Bind the ESC key to exit fullscreen
        # self.root.bind("<Escape>", self.exit_fullscreen)
        self.root.bind("<Escape>", self.toggle_fullscreen)
        self.root.bind("<Control-c>", lambda e: self.root.quit())

        # Initialize services
        self.question_service = question_service
        self.key_id_service = key_id_service

        # Create the ID entry screen
        self.create_id_screen()

    def handle_client(self, client_socket):
        """Handles incoming messages from the main node."""
        while True:
            try:
                message = client_socket.recv(1024).decode()
                if not message:
                    break  # Connection closed

                print(f"Received command: {message}")
                
                # Handle GUI-related commands
                if message == "start_gui":
                    print("Starting GUI...")
                elif message == "stop_gui":
                    print("Stopping GUI...")
                    break  # Stop listening and close server
                elif message.startswith("change_face:"):
                    face = message.split(":")[1]
                    print(f"Switching to face: {face}")

            except Exception as e:
                print(f"Error: {e}")
                break

        client_socket.close()

    def start_gui_server(self, host="127.0.0.1", port=5000):
        """Starts the GUI server to listen for messages."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((host, port))
        server.listen(5)  # Allow up to 5 connections
        print(f"GUI Server listening on {host}:{port}")

        while True:
            client_socket, addr = server.accept()
            print(f"Connection established with {addr}")
            client_handler = threading.Thread(target=self.handle_client, args=(client_socket,))
            client_handler.start()


    def create_id_screen(self):
        """Set up the screen to enter the ID and integer value."""
        self.id_frame = tk.Frame(self.root, bg=LIGHT_BLUE)
        self.id_frame.pack(expand=True)

        # Name Entry
        self.id_label = tk.Label(self.id_frame, text="Enter Name:", font=("Arial", 24), bg=LIGHT_BLUE)
        self.id_label.pack(pady=10)
        self.id_entry = tk.Entry(self.id_frame, width=20, font=("Arial", 24))
        self.id_entry.pack(pady=10)

        if gen_question_key == False:
            if self.question_label is None:
                # Integer Entry
                self.int_label = tk.Label(self.id_frame, text="Enter Key Integer Value:", font=("Arial", 24), bg=LIGHT_BLUE)
                self.int_label.pack(pady=10)
                self.int_entry = tk.Entry(self.id_frame, width=20, font=("Arial", 24))
                self.int_entry.pack(pady=10)
        else:
            # Generate a random key integer between 0 and 99
            self.random_key_value = random.randint(0, 99)  # Random int between 0 and 99

        

        # Toggle Button for Scale Type
        self.scale_toggle_button = tk.Button(
            self.id_frame,
            text=f"Scale: {scale_type}",
            font=("Arial", 24),
            command=self.toggle_scale,
            bg = LIGHT_BLUE,
            activebackground = LIGHT_BLUE
        )
        self.scale_toggle_button.pack(pady=10)

        # Submit Button
        self.id_button = tk.Button(
            self.id_frame,
            text="Submit",
            font=("Arial", 24), 
            command=self.send_key_id,
            bg=GREEN,
            activebackground = LIGHT_GREEN,
            )
        self.id_button.pack(pady=10)


    def toggle_scale(self):
        """Toggle between 1-3 "Triad" and 1-5 "Likert" scale types."""
        global scale_type

        if scale_type == "Triad":
            scale_type = "Likert"
            self.scale_toggle_button.config(text="Scale: Likert")
        else:
            scale_type = "Triad"
            self.scale_toggle_button.config(text="Scale: Triad")

        self.scale_toggle_button.pack(pady=10)


    def create_main_gui(self):
        """Set up the main GUI layout after the ID and integer are entered."""

        # Clear the ID entry frame
        self.id_frame.destroy()

        # Request the first question
        response = self.question_service(-1)
        lastest_question = response.question
        # rospy.loginfo(f"First question received from service: {lastest_question}")
        print(f"First question received from service: {lastest_question}")

        # Container frame for the question label and the question itself
        self.question_frame = tk.Frame(self.root, bg=LIGHT_BLUE)
        self.question_frame.pack(pady=20)

        # Add a label that says "Question:" above the actual question
        self.question_label = tk.Label(self.question_frame, text="Question:", font=("Arial", 24), fg="black", bg=LIGHT_BLUE)
        self.question_label.pack()

        # Display the actual question
        self.label = tk.Message(self.question_frame, text=lastest_question, font=("Arial", 24), width=600, bg=LIGHT_BLUE)
        self.label.pack(pady=10)

        # Container frame for buttons and title label
        self.container_frame = tk.Frame(self.root, bg=LIGHT_BLUE)
        self.container_frame.pack(expand=True)  # Center the frame vertically
        
        if scale_type == "Triad":
            # Title label for interaction question
            self.title_label = tk.Label(
                self.container_frame,
                text="How would you rate the response time of Quori?",  # This label is now above the buttons
                font=("Arial", 18),
                fg="black",
                pady=10,
                bg=LIGHT_BLUE,
            )
        else:
            # Title label for interaction question
            self.title_label = tk.Label(
                self.container_frame,
                text="The response time of Quori was satisfactory",  # This label is now above the buttons
                font=("Arial", 18),
                fg="black",
                pady=10,
                bg=LIGHT_BLUE,
            )
        self.title_label.pack(pady=(10, 5))  # Add some padding for spacing

        # Buttons frame
        self.frame = tk.Frame(self.container_frame, bg=LIGHT_BLUE)
        self.frame.pack(side=tk.TOP, pady=20)

        width = self.frame.winfo_width()
        height = self.frame.winfo_height()
        
        self.buttons = []
        self.selected_button = None  # Track the selected button

        if scale_type == "Triad":
            button_config = [
                ("   Too Slow  ", "#FF9999", "#FFCCCC"),        # Soft red and lighter soft red
                ("Somewhat Slow", "#FFD1A6", "#FFE5CC"),   # Soft amber and lighter soft amber
                ("   Not Slow  ", "#99FF99", "#CCFFCC"),        # Soft green and lighter soft green
            ]

            # Use the same font and size as the submit button
            button_font = ("Arial", 20)
            button_width = width // 4
            button_height = height // 6
        else:
            button_config = [
                ("Strongly Disagree", "#FF8981", "#FFCCC7"),  # Softer red and lighter soft red
                ("     Disagree    ", "#FFB54C", "#FFD6A1"),           # Same orange and lighter orange
                ("      Neutral    ", "#F8D66D", "#FAE6A8"),            # Same yellow and lighter yellow
                ("       Agree     ", GREEN, LIGHT_GREEN),              # Same green and lighter green
                ("  Strongly Agree ", "#6FAF72", "#AFCFB2"),     # Darker green and lighter green
            ]

            # Use the same font and size as the submit button
            button_font = ("Arial", 20)
            button_width = width // 6
            button_height = height // 6

        for i, (label, color, selected_color) in enumerate(button_config):
            btn = tk.Button(
                self.frame,
                text=label,
                width=button_width,
                height=button_height,
                bg=color,
                activebackground=selected_color,  # Use the selected_color for active background
                font=button_font,  # Same font as the submit button
                command=lambda b=i: self.select_button(b)
            )
            btn.pack(side=tk.LEFT, anchor=tk.CENTER, padx=5)
            self.buttons.append(btn)


        # Adjust row height and column width for responsiveness
        for col in range(5):
            self.frame.grid_columnconfigure(col, weight=1)

        self.frame.update_idletasks()


    def calculate_dynamic_font_size(self):
        """Calculate a dynamic font size based on available space."""
        width = self.frame.winfo_width()  # Get the width of the button container frame
        available_width = width // 5  # Divide by 5 because we have 5 buttons
        font_size = available_width // 10  # Adjust this ratio to get the ideal font size
        
        # Ensure a minimum font size to prevent being too small
        return max(font_size, 14)  # Ensure the font size is at least 14


    def bring_to_front(self):
        """Bring the window to the front and ensure it stays on top."""
        self.root.focus_force()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)  # Reset -topmost attribute

        try:
            # Get the current window ID using xprop
            window_id = subprocess.check_output(
                ["xprop", "-root", "_NET_ACTIVE_WINDOW"],
                text=True
            ).strip().split()[-1]

            # Use wmctrl to raise the window to the front
            subprocess.run(["wmctrl", "-i", "-a", window_id])
        except Exception as e:
            # rospy.logwarn(f"Failed to bring window to front: {e}")
            print(f"Failed to bring window to front: {e}")


    def exit_fullscreen(self, event=None):
        """Exit fullscreen mode."""
        self.root.attributes('-fullscreen', False)


    def toggle_fullscreen(self, event=None):
        """Exit fullscreen mode."""
        self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen'))


    def send_key_id(self):
        """Send the user ID, key ID, and scale type to the /key_id service."""

        # Remove any previous error messages
        for widget in self.id_frame.winfo_children():
            if isinstance(widget, tk.Label) and widget.cget("fg") == "red":
                widget.destroy()  # Remove previous error messages
        
        id_string = self.id_entry.get()

        if gen_question_key == False:
            int_value = self.int_entry.get()
        else:
            int_value = self.random_key_value

        try:
            int_value = int(int_value)  # Attempt to convert to integer
            # rospy.loginfo(f"Attempting to send key ID: {int_value}")
            print(f"Attempting to send key ID: {int_value}")

            # Check if the integer is within a valid range if needed
            if int_value < 0:  
                raise ValueError("Integer value must be non-negative.")

            if id_string:
                # Create the request for the KeyID service
                key_id_srv = KeyIDRequest()  # Use KeyIDRequest instead of KeyID
                key_id_srv.user_id = id_string
                key_id_srv.key_id = int_value
                key_id_srv.scale_type = scale_type

                # Call the key_id_service with the request and capture the response
                response = self.key_id_service(key_id_srv)

                # Check the response
                if response.success:
                    # rospy.loginfo(f"Service call successful: User ID: {id_string}, Key ID: {int_value}, Scale Type: {scale_type}")
                    print(f"Service call successful: User ID: {id_string}, Key ID: {int_value}, Scale Type: {scale_type}")
                    # Proceed to the main GUI if successful
                    self.create_main_gui()
                else:
                    # rospy.logerr("Service call failed. Success flag is false.")
                    print("Service call failed. Success flag is false.")
                    error_label = tk.Label(self.id_frame, text="Service call failed. Please try again.", fg="red", font=("Arial", 16))
                    error_label.pack(pady=10)

        except ValueError as e:
            # rospy.logerr(f"Invalid integer value entered: {e}. No value sent.")
            print(f"Invalid integer value entered: {e}. No value sent.")

            # Display an error message to the user
            error_label = tk.Label(self.id_frame, text="Please enter a valid integer.", fg="red", font=("Arial", 16))
            error_label.pack(pady=10)
            return

        except Exception as e:
            # rospy.logerr(f"Service call to /key_id failed: {e}")
            print(f"Service call to /key_id failed: {e}")

            # Display an error message to the user
            error_label = tk.Label(self.id_frame, text="Service call failed. Please try again.", fg="red", font=("Arial", 16))
            error_label.pack(pady=10)
            return


    def select_button(self, button_index):
        """Handles button selection and question service call."""
        global lastest_question

        # Reset the previously selected button
        if self.selected_button is not None:
            self.buttons[self.selected_button].config(relief=tk.RAISED)

        # Highlight the selected button
        self.selected_button = button_index
        self.buttons[button_index].config(relief=tk.SUNKEN)

        try:
            # rospy.loginfo(f"Calling question service with selected button index: {self.selected_button}")
            print(f"Calling question service with selected button index: {self.selected_button}")
            # Call the service with the selected button index
            response = self.question_service(self.selected_button)  # Capture the response
            lastest_question = response.question  # Update the latest question
            self.update_label(lastest_question)  # Update the GUI with the new question
        except Exception as e:
            # rospy.logerr(f"Failed to call question service: {e}")
            print(f"Failed to call question service: {e}")

        # Reset the selected button appearance
        self.buttons[self.selected_button].config(relief=tk.RAISED)
        self.selected_button = None


    def update_label(self, text):
        """Update the label with new text."""
        if hasattr(self, 'label'):
            self.label.config(text=text)


    def update_label_with_latest_question(self):
        global lastest_question
        """Update the label with the latest question."""
        self.update_label(lastest_question)  # Use the global variable
        self.root.after(1000, self.update_label_with_latest_question)  # Continue updating


    def run(self):
        """Run the Tkinter main loop."""
        self.root.mainloop()


    def close(self):
        """Close the application window safely."""
        rospy.signal_shutdown("Application window closed.")
        self.root.quit()  # Stop the main loop if it's running
        self.root.destroy()


class GuiNode:
    """ROS Node for the GUI application."""

    def __init__(self, question_label = None):
        """Initialize the GUI Node."""
        # rospy.init_node('gui_node')
        print("Initializing GUI Node")
        self.gui_app = None
        self.gui_thread = None
        self.question_label = question_label # from commandline, which set of questions to run
        self.lastest_question = "Waiting for message..."

        # Services to start and stop GUI
        self.start_service = rospy.Service('start_gui', Empty, self.start_gui)
        self.stop_service = rospy.Service('stop_gui', Empty, self.stop_gui)

        # Initialize the service client for /get_question
        # rospy.wait_for_service('/get_question')
        self.get_question_service = rospy.ServiceProxy('/get_question', GetQuestion, self.request_question)

        self.remote_update = rospy.Service('/remote_update', GetQuestion, self.request_question)

        # Initialize the service client for /key_id
        # rospy.wait_for_service('/key_id')
        self.key_id_service = rospy.ServiceProxy('/key_id', KeyID)


    def request_question(self, req):
        """Request a question from the service."""
        try:
            response = self.get_question_service(req)
            self.lastest_question = response.question
            print(f"New Question: {self.lastest_question}")
            # Update your GUI with the received question
            self.gui_app.update_label(self.lastest_question)
            return response
        except Exception as e:
            # rospy.logerr(f"Service call failed: {e}")
            print(f"Service call failed: {e}")


    def start_gui(self, req):
        """Start the GUI application."""
        if self.gui_app is None:
            # rospy.loginfo("Starting GUI")
            print("Starting GUI")
            self.gui_thread = Thread(target=self.launch_gui)
            self.gui_thread.start()
        else:
            # rospy.logwarn("GUI is already running")
            print("GUI is already running")
        return EmptyResponse()


    def stop_gui(self, req):
        """Stop the GUI application."""
        if self.gui_app is not None:
            # rospy.loginfo("Stopping GUI")
            print("Stopping GUI")
            self.gui_app.close()
            # rospy.loginfo("GUI closed")
            print("GUI closed")
            self.gui_thread.join()  # Wait for the GUI thread to finish
            # rospy.loginfo("GUI thread finished")
            print("GUI thread finished")
            self.gui_app = None
        else:
            # rospy.logwarn("GUI is not running")
            print("GUI is not running")
        return EmptyResponse()


    def launch_gui(self):
        """Launch the Tkinter GUI application."""
        root = tk.Tk()
        self.gui_app = GuiApp(root, self.get_question_service, self.key_id_service, self.question_label)
        # Ensure the latest question is shown on screen immediately
        self.gui_app.update_label(self.lastest_question)
        self.gui_app.run()

    def run(self):
        """Run the ROS node."""
        # rospy.loginfo("GUI Node is running")
        print("GUI Node is running")
        self.launch_gui()
        # rospy.spin()


if __name__ == '__main__':
    """Main entry point for the GUI Node."""

    try:
        question_label = int(sys.argv[1])
    except (IndexError, TypeError):
        # rospy.loginfo("No Question label found, prompting during startup")
        print("No Question label found, prompting during startup")
        question_label = None
    node = GuiNode(question_label)
    node.run()
