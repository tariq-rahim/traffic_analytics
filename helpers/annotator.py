from typing import List, Optional, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import supervision
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette

colors_dict = {
    0: (102, 60, 165),     # ALLIGATOR (H) - #a53c66
    1: (227, 57, 18),      # BLOCK - intense blue
    2: (253, 69, 56),      # LONGITUDIONAL - intense blue
    3: (255, 187, 1),      # TRANSVERSE - intense blue
    4: (255, 155, 12),     # RUTTING - intense blue
    5: (253, 69, 56),      # RAVELING (H) - #3845fd
    6: (130, 159, 0),      # CORRUGATION - intense green
    7: (226, 66, 233),     # POTHOLE - #e942e2
    8: (80, 127, 255),     # DEPRESSION - coral
    9: (0, 215, 255),      # EDGE - gold
    10: (0, 128, 128),     # RAIL ROAD CROSSING - olive
    11: (128, 0, 0),       # BLEEDING - navy
    12: (203, 192, 255),   # JOINT REFLECTION - pink
    13: (255, 155, 12),    # PATCHING - #0c9bff
    14: (130, 159, 0),     # POLISHED AGGREGATE - intense green
    15: (18, 57, 227),     # SHOVING - #e33912
    16: (250, 230, 230),   # SLIPPAGE - lavender
    17: (208, 224, 64),    # BUMPS & SAGS - turquoise
    18: (130, 0, 75),      # SWELL - indigo
    19: (255, 187, 1),     # WEATHERING - #01bbff
    20: (18, 57, 227),     # CARRIAGEWAY - #e33912
    21: (0, 0, 255),       # ALLIGATOR (M) - green
    22: (102, 60, 165),    # ALLIGATOR (L) - #a53c66 (Same as ID 0)
    23: (227, 57, 18)      # RAVELING (M) - intense blue (Same as ID 1)
}

class BoxAnnotator:
    """
    A class for drawing bounding boxes on an image using detections provided.

    Attributes:
        color (Union[Color, ColorPalette]): The color to draw the bounding box,
            can be a single color or a color palette
        thickness (int): The thickness of the bounding box lines, default is 2
        text_color (Color): The color of the text on the bounding box, default is white
        text_scale (float): The scale of the text on the bounding box, default is 0.5
        text_thickness (int): The thickness of the text on the bounding box,
            default is 1
        text_padding (int): The padding around the text on the bounding box,
            default is 5
    """

    def __init__(
        self,
        # color: Union[Color, ColorPalette] = ColorPalette.default(),
        colors_list: Union[Color, ColorPalette] = [Color(r=c[0], g=c[1], b=c[2]) for c in list(colors_dict.values())],
        thickness: int = 2,
        text_color: Color = Color.white(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 5,
        font_path='font/CascadiaMono-Bold.otf',
        font_size=20
    ):
        self.colors_list: Union[Color, ColorPalette] = colors_list
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.font_path = font_path
        self.font_size = font_size




    def drawing_center_points(self, draw_obj, box, box_color='black'):
        fill_color = box_color
        draw = draw_obj

        # Calculate center points and plot them
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        circle_radius = 3  # Adjust the radius of the circle if needed
        circle_box = [
            (center_x - circle_radius, center_y - circle_radius),
            (center_x + circle_radius, center_y + circle_radius)
        ]
        draw.ellipse(circle_box, fill=fill_color)
        
        # Draw lines from center to corners
        draw.line([(x1, y1), (center_x, center_y)], fill=fill_color, width=1)
        draw.line([(x2, y1), (center_x, center_y)], fill=fill_color, width=1)
        draw.line([(x1, y2), (center_x, center_y)], fill=fill_color, width=1)
        draw.line([(x2, y2), (center_x, center_y)], fill=fill_color, width=1)
        
        return draw






    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
        skip_label: bool = False,
    ) -> np.ndarray:
        """
        Draws bounding boxes on the frame using the detections provided.

        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which the
                bounding boxes will be drawn
            labels (Optional[List[str]): An optional list of labels
                corresponding to each detection. If `labels` are not provided,
                corresponding `class_id` will be used as label.
            skip_label (bool): Is set to `True`, skips bounding box label annotation.
        Returns:
            np.ndarray: The image with the bounding boxes drawn on it
        """
        image = Image.fromarray(scene)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(self.font_path, self.font_size)


        


        
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            class_id = (
                detections.class_id[i] if detections.class_id is not None else None
            )
            # idx = class_id if class_id is not None else i
            idx = class_id

            color = self.colors_list[class_id]
            
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=color.as_rgb(),
                width=self.thickness
            )
            
            if not skip_label:
                text = (
                    f"{class_id}"
                    if (labels is None or len(detections) != len(labels))
                    else labels[i]
                )
                
                # Create a temporary image to get the size of the text
                temp_image = Image.new('RGB', (1, 1))  # Create a 1x1 pixel image
                drawx = ImageDraw.Draw(temp_image)

                # Get the bounding box of the text
                bbox = drawx.textbbox((0, 0), text, font)

                # Calculate the width and height of the text
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                
                text_x = x1
                text_y = y1  # Adjusted to be inside the bounding box

                text_background_x1 = x1
                text_background_y1 = y1

                text_background_x2 = x1 + text_width
                text_background_y2 = y1 + text_height + self.text_padding

                draw.rectangle(
                    [text_background_x1, text_background_y1, text_background_x2, text_background_y2],
                    fill=color.as_rgb()
                )
                draw.text(
                    (text_x, text_y),
                    text,
                    fill=self.text_color.as_rgb(),
                    font=font
                )
            draw = self.drawing_center_points(draw, detections.xyxy[i], box_color=color.as_rgb())
                
        return np.array(image)
    
    
    

