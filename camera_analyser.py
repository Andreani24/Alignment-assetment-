import cv2
import numpy as np
import os
import math
import time


class CatheterAnalyser:
    """
    A class to encapsulate the logic for manually analyzing catheter images.

    This class first guides the user to align the image horizontally, then
    captures four points to calculate the rotational offset with zoom/pan.
    """

    def __init__(self, image, filename_prefix="capture"):
        """
        Initializes the analyzer with an image array.
        """
        self.output_dir = "manual_analysis_output"
        self.window_name = "Manual Catheter Analysis"
        self.filename_prefix = filename_prefix

        # State variables
        self.original_image = image
        if self.original_image is None:
            raise ValueError("Input image cannot be None.")

        self.processed_image = self.original_image.copy()
        self.display_image = self.original_image.copy()
        self.clicked_points = []

        # State machine for the two-phase process
        self.state = 'ALIGNMENT'  # Can be 'ALIGNMENT' or 'MEASUREMENT'

        # Zoom and Pan variables
        self.zoom_level = 1.0
        self.pan_start_x, self.pan_start_y = 0, 0
        self.panning = False
        self.view_offset_x, self.view_offset_y = 0, 0

        # Info panel for text display
        self.info_panel_height = 150

    def _mouse_callback(self, event, x, y, flags, params):
        """
        Handles mouse events for alignment, point placement, zoom, and pan.
        """
        # Ignore clicks on the info panel
        if y > self.display_image.shape[0]:
            return

        # --- Point Placement (State-dependent) ---
        if event == cv2.EVENT_LBUTTONDOWN:
            max_points = 2 if self.state == 'ALIGNMENT' else 4
            is_in_confirmation = self.state == 'ALIGNMENT' and len(self.clicked_points) == 2
            if len(self.clicked_points) < max_points and not is_in_confirmation:
                original_x = int(x / self.zoom_level + self.view_offset_x)
                original_y = int(y / self.zoom_level + self.view_offset_y)

                self.clicked_points.append((original_x, original_y))
                print(
                    f"Point {len(self.clicked_points)} recorded at: ({original_x}, {original_y}) for {self.state} phase.")

        # --- Zooming (centered on mouse) ---
        elif event == cv2.EVENT_MOUSEWHEEL:
            img_x = x / self.zoom_level + self.view_offset_x
            img_y = y / self.zoom_level + self.view_offset_y

            if flags > 0:
                self.zoom_level *= 1.1
            else:
                self.zoom_level /= 1.1
            self.zoom_level = max(1.0, self.zoom_level)

            self.view_offset_x = img_x - x / self.zoom_level
            self.view_offset_y = img_y - y / self.zoom_level

        # --- Panning ---
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.panning = True
            self.pan_start_x, self.pan_start_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE and self.panning:
            dx = (x - self.pan_start_x) / self.zoom_level
            dy = (y - self.pan_start_y) / self.zoom_level
            self.view_offset_x -= dx
            self.view_offset_y -= dy
            self.pan_start_x, self.pan_start_y = x, y

        elif event == cv2.EVENT_RBUTTONUP:
            self.panning = False

    def _align_image(self):
        """Rotates the image based on the two alignment points."""
        p1, p2 = self.clicked_points
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

        h, w = self.original_image.shape[:2]
        center = (w // 2, h // 2)

        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.processed_image = cv2.warpAffine(self.original_image, rot_mat, (w, h))

        print(f"\nImage aligned by {angle:.2f} degrees.")
        self.clicked_points = []
        self.state = 'MEASUREMENT'
        self.zoom_level = 1.0
        self.view_offset_x, self.view_offset_y = 0, 0

    def _update_display(self):
        """Creates the transformed view and combines it with an info panel."""
        current_image = self.processed_image if self.state == 'MEASUREMENT' else self.original_image
        h, w = current_image.shape[:2]

        self.view_offset_x = np.clip(self.view_offset_x, 0, w - w / self.zoom_level)
        self.view_offset_y = np.clip(self.view_offset_y, 0, h - h / self.zoom_level)

        x1, y1 = int(self.view_offset_x), int(self.view_offset_y)
        x2, y2 = int(x1 + w / self.zoom_level), int(y1 + h / self.zoom_level)

        zoomed_view = current_image[y1:y2, x1:x2]
        self.display_image = cv2.resize(zoomed_view, (w, h))

        for i, (px, py) in enumerate(self.clicked_points):
            view_x = int((px - self.view_offset_x) * self.zoom_level)
            view_y = int((py - self.view_offset_y) * self.zoom_level)
            color = (0, 255, 0) if self.state == 'ALIGNMENT' else ((255, 0, 0) if i < 2 else (0, 0, 255))
            cv2.circle(self.display_image, (view_x, view_y), 3, color, -1)

        # Create the info panel
        canvas = np.zeros((h + self.info_panel_height, w, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = self.display_image

        # Add text to the info panel
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # Adjusted font scale
        font_color = (255, 255, 255)
        thickness = 1  # Adjusted thickness

        phase_text = f"Phase: {self.state}"
        cv2.putText(canvas, phase_text, (20, h + 30), font, font_scale, font_color, thickness)

        if self.state == 'ALIGNMENT':
            for i, (px, py) in enumerate(self.clicked_points):
                coord_text = f"P{i + 1}: ({px}, {py})"
                cv2.putText(canvas, coord_text, (20, h + 60 + i * 25), font, font_scale, font_color, thickness)

            if len(self.clicked_points) == 2:
                confirm_text = "Align using these points? Press 'y' (yes) or 'n' (no)."
                cv2.putText(canvas, confirm_text, (20, h + 120), font, font_scale, (0, 255, 255), thickness)

        return canvas

    def run(self):
        """Main method to run the entire analysis process."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("--- PHASE 1: ALIGNMENT ---")
        print("Please click 2 points along the catheter to set it horizontally.")
        print("\nControls:")
        print("- Scroll Wheel: Zoom | Right-click+drag: Pan | 'r': Reset | 'q': Quit")

        while True:
            canvas = self._update_display()
            cv2.imshow(self.window_name, canvas)

            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'): break
            if key == ord('r'):
                print("Resetting points for current phase.")
                self.clicked_points = []

            if self.state == 'ALIGNMENT' and len(self.clicked_points) == 2:
                if key == ord('y'):
                    self._align_image()
                    print("\n--- PHASE 2: MEASUREMENT ---")
                    print("Please click 4 points:")
                    print("1. Top edge of catheter | 2. Bottom edge of catheter")
                    print("3. Top edge of electrode | 4. Bottom edge of electrode")
                elif key == ord('n'):
                    print("Alignment cancelled. Please re-select 2 points.")
                    self.clicked_points = []

            if self.state == 'MEASUREMENT' and len(self.clicked_points) == 4:
                self._calculate_and_save_results()
                cv2.waitKey(0)
                break

        cv2.destroyAllWindows()

    def _calculate_and_save_results(self):
        """
        Performs final calculations using individual point correction and saves the annotated image.
        """
        print("\n4 points collected. Calculating results with geometric correction...")

        # 1. Establish Catheter Reference Frame
        catheter_y1, catheter_y2 = self.clicked_points[0][1], self.clicked_points[1][1]
        catheter_midpoint_y = (catheter_y1 + catheter_y2) / 2.0
        catheter_radius = abs(catheter_y1 - catheter_y2) / 2.0

        if catheter_radius == 0:
            print("Error: Catheter radius is zero. Cannot calculate angle.")
            return

        # 2. Calculate Individual Offsets for Each Electrode Edge
        electrode_y_top = self.clicked_points[2][1]
        electrode_y_bottom = self.clicked_points[3][1]

        offset_top = electrode_y_top - catheter_midpoint_y
        offset_bottom = electrode_y_bottom - catheter_midpoint_y

        # 3. Convert Each Offset to a True Angle (in radians)
        final_rotation_angle_deg = 0.0
        try:
            # Ensure the ratio is within the valid domain for arcsin [-1, 1]
            ratio_top = np.clip(offset_top / catheter_radius, -1.0, 1.0)
            ratio_bottom = np.clip(offset_bottom / catheter_radius, -1.0, 1.0)

            angle_top_rad = math.asin(ratio_top)
            angle_bottom_rad = math.asin(ratio_bottom)

            # 4. Find the True Centerline Angle by Averaging
            final_rotation_angle_rad = (angle_top_rad + angle_bottom_rad) / 2.0
            final_rotation_angle_deg = math.degrees(final_rotation_angle_rad)

        except ValueError as e:
            print(f"Error: Math domain error during calculation. This likely means one or more "
                  f"electrode points were clicked outside the catheter bounds. Details: {e}")
            return

        # 5. Visualize the *True* Electrode Centerline
        # Convert the final angle back to a pixel offset for drawing
        true_offset_pixels = catheter_radius * math.sin(final_rotation_angle_rad)
        true_electrode_midpoint_y = int(catheter_midpoint_y + true_offset_pixels)

        # --- Create Final Annotated Image ---
        final_image = self.processed_image.copy()
        img_width = final_image.shape[1]

        # Draw the 4 clicked points
        for i, (px, py) in enumerate(self.clicked_points):
            color = (255, 0, 0) if i < 2 else (0, 0, 255)
            cv2.circle(final_image, (px, py), 5, color, -1)

        # Draw catheter centerline (Cyan)
        cv2.line(final_image, (0, int(catheter_midpoint_y)), (img_width, int(catheter_midpoint_y)), (255, 255, 0), 2)
        # Draw the TRUE electrode centerline (Yellow)
        cv2.line(final_image, (0, true_electrode_midpoint_y), (img_width, true_electrode_midpoint_y), (0, 255, 255), 2)

        # Display the final calculated angle on the image
        text = f"Rotation Angle: {final_rotation_angle_deg:.2f} degrees"
        cv2.putText(final_image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        print(f"Catheter Radius: {catheter_radius:.2f} pixels")
        print(f"Calculated True Angle: {final_rotation_angle_deg:.2f} degrees")

        # Save the result
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = int(time.time())
        output_path = os.path.join(self.output_dir, f"{self.filename_prefix}_{timestamp}_analysis.png")
        cv2.imwrite(output_path, final_image)
        print(f"\nAnalysis complete. Saved result to: {output_path}")

        # Display final result with its own info panel
        h, w = final_image.shape[:2]
        final_canvas = np.zeros((h + self.info_panel_height, w, 3), dtype=np.uint8)
        final_canvas[0:h, 0:w] = final_image
        cv2.putText(final_canvas, f"Final Angle: {final_rotation_angle_deg:.2f} deg", (20, h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(final_canvas, "Press any key to exit.", (20, h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1)
        cv2.imshow(self.window_name, final_canvas)

def capture_from_camera():
    """
    Opens a camera feed, allows the user to capture a frame, and returns it.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return None

    window_name = "Camera Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        # Draw text on a copy, so the original frame remains clean
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Adjusted scale

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            captured_frame = frame  # Return the clean frame
            print("Frame captured!")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_frame


if __name__ == '__main__':
    captured_image = capture_from_camera()
    if captured_image is not None:
        try:
            analyzer = CatheterAnalyser(captured_image)
            analyzer.run()
        except (ValueError, FileNotFoundError) as e:
            print(e)
