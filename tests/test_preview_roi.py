import unittest

import numpy as np

from material_classifier.preview import center_crop


class RegionOfInterestTests(unittest.TestCase):
    def test_off_center_crop_targets_belt_without_foreground(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        crop, box = center_crop(frame, 0.4, center_x=0.25, center_y=0.2)
        self.assertEqual(crop.shape, (432, 432, 3))
        self.assertEqual(box, (264, 0, 696, 432))

    def test_crop_stays_inside_frame_near_edge(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, box = center_crop(frame, 0.5, center_x=1, center_y=1)
        self.assertEqual(box, (400, 240, 640, 480))


if __name__ == "__main__":
    unittest.main()
