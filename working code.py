import cv2
import numpy as np
import os
import math

# --- Global Variables ---
# We use a list to store the (x, y) coordinates of the clicks
clicked_points = []
# Make a copy of the original image to draw on
display_image = None

def mouse_callback(event, x, y, flags, params):
    """
    Handles mouse click events. Appends the click coordinates to a list
    and draws a circle on the image for visual feedback.
    """
    global display_image, clicked_points

    # Check if the left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            # Store the coordinates
            clicked_points.append((x, y))
            
            # Define colors for the points for clarity
            # Point 1&2 (Catheter): Blue
            # Point 3&4 (Electrodes): Red
            color = (255, 0, 0) if len(clicked_points) <= 2 else (0, 0, 255)
            
            # Draw a circle at the clicked point
            cv2.circle(display_image, (x, y), 10, color, -1)
            
            # Refresh the image window
            cv2.imshow("Manual Catheter Analysis", display_image)
            print(f"Point {len(clicked_points)} recorded at: ({x}, {y})")

def manual_analysis(image_path, output_dir="manual_analysis_output"):
    """
    Main function to load an image, collect four user clicks, calculate
    centerlines, and save the resulting annotated image.
    """
    global display_image, clicked_points

    # --- 1. Load Image and Setup ---
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    display_image = original_image.copy()
    
    # Create a resizable window and set the mouse callback function
    cv2.namedWindow("Manual Catheter Analysis", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Manual Catheter Analysis", mouse_callback)

    print("Please click 4 points in this order:")
    print("1. Top edge of the catheter")
    print("2. Bottom edge of the catheter")
    print("3. Top edge of the electrode")
    print("4. Bottom edge of the electrode")
    print("\nPress 'q' to quit or 'r' to reset points.")

    # --- 2. Interaction Loop ---
    while True:
        cv2.imshow("Manual Catheter Analysis", display_image)
        key = cv2.waitKey(1) & 0xFF

        # Quit if 'q' is pressed
        if key == ord('q'):
            break
        
        # Reset if 'r' is pressed
        if key == ord('r'):
            print("Resetting points.")
            clicked_points = []
            display_image = original_image.copy() # Restore original image

        # Break the loop once 4 points are collected
        if len(clicked_points) == 4:
            print("\n4 points collected. Calculating results...")
            
            # --- 3. Calculation ---
            # Catheter points (highest and lowest y-values)
            catheter_y1 = clicked_points[0][1]
            catheter_y2 = clicked_points[1][1]
            catheter_midpoint_y = (catheter_y1 + catheter_y2) // 2
            
            # Calculate catheter radius in pixels
            catheter_diameter = abs(catheter_y1 - catheter_y2)
            catheter_radius = catheter_diameter / 2.0

            # Electrode points
            electrode_y1 = clicked_points[2][1]
            electrode_y2 = clicked_points[3][1]
            electrode_midpoint_y = (electrode_y1 + electrode_y2) // 2
            
            # Calculate the linear offset in pixels from the catheter's center
            offset = abs(catheter_midpoint_y - electrode_midpoint_y)
            
            # --- 4. Convert Offset to Degrees ---
            rotation_angle = 0.0
            # Ensure radius is not zero and offset is not larger than the radius
            if catheter_radius > 0 and offset <= catheter_radius:
                # Use the arcsin formula: angle = asin(opposite / hypotenuse)
                # Here, 'opposite' is the offset and 'hypotenuse' is the radius
                ratio = offset / catheter_radius
                rotation_angle = math.degrees(math.asin(ratio))
            else:
                print("Error: Invalid points. Electrode midpoint is outside the catheter radius.")

            # --- 5. Visualization ---
            img_width = display_image.shape[1]
            # Draw catheter centerline (Cyan)
            cv2.line(display_image, (0, catheter_midpoint_y), (img_width, catheter_midpoint_y), (255, 255, 0), 2)
            # Draw electrode centerline (Yellow)
            cv2.line(display_image, (0, electrode_midpoint_y), (img_width, electrode_midpoint_y), (0, 255, 255), 2)

            # Display the final angle value
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Rotation Angle: {rotation_angle:.2f} degrees"
            cv2.putText(display_image, text, (50, 50), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            
            print(f"Catheter Radius: {catheter_radius:.2f} pixels")
            print(f"Pixel Offset: {offset} pixels")
            print(f"Calculated Angle: {rotation_angle:.2f} degrees")

            # --- 6. Save and Exit ---
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.basename(image_path)
            filename_no_ext = os.path.splitext(base_filename)[0]
            output_path = os.path.join(output_dir, f"{filename_no_ext}_manual_analysis.png")
            cv2.imwrite(output_path, display_image)
            print(f"\nAnalysis complete. Saved result to: {output_path}")
            
            # Display final image until user quits
            cv2.imshow("Manual Catheter Analysis", display_image)
            cv2.waitKey(0)
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Use the original, clean image for this manual process
    input_image_path = r"C:\Users\Clemens Eder\Documents\Alignment-assetment-\pictures\pic5.jpg"
    manual_analysis(input_image_path)
