from typing import List, Optional, Union

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import supervision
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette

class PolygonZoneAnnotator:
    """
    A class for drawing zones on an image.

    Attributes:
        color (Union[Color, ColorPalette]): The color to draw the zones,
            can be a single color or a color palette
        thickness (int): The thickness of the zone lines, default is 2
        text_color (Color): The color of the text on the zone, default is white
        text_scale (float): The scale of the text on the zone, default is 0.5
        text_thickness (int): The thickness of the text on the zone,
            default is 1
        text_padding (int): The padding around the text on the zone,
            default is 5
        font_path (str): The path to the font file, e.g., 'font/CascadiaMono-Bold.otf'
        font_size (int): The font size, default is 20
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        text_color: Color = Color.white(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 5,
        font_path='font/CascadiaMono-Bold.otf',
        font_size=20
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.font_path = font_path
        self.font_size = font_size

    def annotate(
        self,
        scene: np.ndarray,
        polygon_zones: List[np.ndarray],
        labels: Optional[List[str]] = None,
        color = Color.black().as_rgb()
    ) -> np.ndarray:
        """
        Draws zones on the image using the provided polygon zones.

        Args:
            scene (np.ndarray): The image on which the zones will be drawn
            polygon_zones (List[np.ndarray]): A list of polygon zones, where each zone
                is defined as an array of vertices (x, y).
            zone_labels (Optional[List[str]]): An optional list of labels corresponding
                to each zone. If `zone_labels` are not provided, labels won't be drawn.
        Returns:
            np.ndarray: The image with the zones drawn on it
        """
        image = Image.fromarray(scene)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(self.font_path, self.font_size)

        for i, zone in enumerate(polygon_zones):
            # Draw the zone using the provided vertices
            draw.polygon(zone.flatten().tolist(), outline=color, width=self.thickness)

            # if zone_labels and i < len(zone_labels):
            #     text = zone_labels[i]
            #     if text:
            #         # Calculate the center of the zone to place the label
            #         center_x = int(np.mean(zone[:, 0]))
            #         center_y = int(np.mean(zone[:, 1]))

            #         text_width, text_height = draw.textsize(text, font)

            #         text_x = center_x - text_width // 2
            #         text_y = center_y - text_height // 2

            #         text_background_x1 = text_x
            #         text_background_y1 = text_y
            #         text_background_x2 = text_x + text_width
            #         text_background_y2 = text_y + text_height

            #         # Draw a background rectangle for the label
            #         draw.rectangle([text_background_x1, text_background_y1, text_background_x2, text_background_y2],
            #                        fill=self.color.as_rgb())

            #         # Draw the label text
            #         draw.text((text_x, text_y), text, fill=self.text_color.as_rgb(), font=font)

        return np.array(image)
    
    
    # def trigger(self, detections: Detections):
        
        
        
    #     return detections
    
    
    def trigger(self, detections, polygon_zones):
        """
        Create a mask to filter detections that are inside any of the provided polygon zones.

        Args:
            detections: The detections in the format you provided.
            polygon_zones: A list of polygon zones, where each zone is defined as an array of vertices (x, y).

        Returns:
            A boolean mask that is True for detections inside a zone, and False for detections outside.
        """
        mask = np.zeros(len(detections), dtype=bool)

        for zone in polygon_zones:
            for j, detection in enumerate(detections.xyxy):
                x1, y1, x2, y2 = detection.astype(float)  # Convert to float
                # Check if the center of the detection is inside the zone
                center_x = (x1 + x2) / 2  # Use float division
                center_y = (y1 + y2) / 2  # Use float division
                if cv2.pointPolygonTest(zone, (center_x, center_y), False) >= 0:
                    mask[j] = True  # Detection is inside the zone

        return mask