import itertools
import pyautogui
import time

# Define the base code with three underscores
base_code = "09C0A-8___8-P66YB"

# Characters to replace underscores (0-9 and A-Z)
characters = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Generate all possible combinations for the three underscores
combinations = itertools.product(characters, repeat=3)

# Wait 5 seconds before starting to allow user to focus on the input field
time.sleep(5)

# Simulate pasting each combination
for combo in combinations:
    # Replace the underscores with the current combination
    current_code = base_code.replace("___", "{}{}{}".format(*combo))
    
    # Highlight all text in the text box (Ctrl+A) and delete it (Backspace)
    pyautogui.hotkey("ctrl", "a")  # Select all text
    pyautogui.press("backspace")   # Delete selected text

    # Simulate typing the code (this will paste into the currently active field)
    pyautogui.write(current_code)
    pyautogui.press("enter")
    
    # Simulate clicking "Continue" button (adjust coordinates as needed)
    pyautogui.click(x=1350, y=645)  # Replace x and y with actual button coordinates
    pyautogui.click(x=1000, y=645)
    
    # Add a small delay to avoid overwhelming the system
    time.sleep(0.001)
