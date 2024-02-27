import numpy as np
from PIL import Image, ImageDraw
import cv2

class CrossingLine:
    """
    A class for drawing a line on an image.

    Attributes:
        color (tuple): The color of the line in RGB format, e.g., (255, 0, 0) for red.
        thickness (int): The thickness of the line.
    """

    def __init__(self, color='green', thickness=2):
        self.color = color
        self.thickness = thickness

    def annotate(self, scene: np.ndarray, crossing_line: np.ndarray) -> np.ndarray:
        """
        Draws a line on the given frame using the provided crossing line coordinates.

        Args:
            scene (np.ndarray): The image on which the line will be drawn.
            crossing_line (np.ndarray): A list of crossing line coordinates as a NumPy array.

        Returns:
            np.ndarray: The image with the line drawn on it.
        """
        image = Image.fromarray(scene)
        draw = ImageDraw.Draw(image)

        for line_coords in crossing_line:
            x1, y1 = line_coords[0]
            x2, y2 = line_coords[1]
            draw.line([(x1, y1), (x2, y2)], fill=self.color, width=self.thickness)

        return np.array(image)
