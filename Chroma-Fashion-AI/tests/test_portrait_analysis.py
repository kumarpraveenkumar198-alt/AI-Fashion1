from io import BytesIO
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageCms

from portrait_analysis import (
    MAX_UPLOAD_BYTES,
    PortraitAnalysisError,
    assess_image_quality,
    analyze_portrait,
    classify_skin_color,
    crop_face,
    decode_uploaded_image,
    extract_skin_color,
    select_largest_face,
)


class PortraitAnalysisTests(unittest.TestCase):
    def encode_png(self, image: Image.Image) -> bytes:
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    def test_decode_normalizes_transparent_png_to_rgb(self):
        source = Image.new("RGBA", (160, 140), (125, 96, 78, 255))
        normalized = decode_uploaded_image(self.encode_png(source))
        self.assertEqual(normalized.mode, "RGB")
        self.assertEqual(normalized.size, (160, 140))

    def test_decode_applies_embedded_srgb_profile(self):
        source = Image.new("RGB", (160, 140), (125, 96, 78))
        output = BytesIO()
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        source.save(output, format="PNG", icc_profile=profile.tobytes())
        normalized = decode_uploaded_image(output.getvalue())
        self.assertEqual(normalized.mode, "RGB")
        self.assertEqual(normalized.size, (160, 140))

    def test_decode_corrects_exif_orientation(self):
        source = Image.new("RGB", (160, 140), (125, 96, 78))
        exif = source.getexif()
        exif[274] = 6
        output = BytesIO()
        source.save(output, format="JPEG", exif=exif)
        normalized = decode_uploaded_image(output.getvalue())
        self.assertEqual(normalized.size, (140, 160))

    def test_decode_resizes_large_but_accepted_images(self):
        source = Image.new("RGB", (2500, 1300), (125, 96, 78))
        normalized = decode_uploaded_image(self.encode_png(source))
        self.assertEqual(normalized.size, (2400, 1248))

    def test_decode_rejects_excessive_pixel_dimensions(self):
        source = Image.new("1", (5000, 5000), 0)
        with self.assertRaisesRegex(PortraitAnalysisError, "20 megapixels"):
            decode_uploaded_image(self.encode_png(source))

    def test_decode_rejects_unusable_dimensions(self):
        source = Image.new("RGB", (32, 32), (120, 100, 90))
        with self.assertRaisesRegex(PortraitAnalysisError, "too small"):
            decode_uploaded_image(self.encode_png(source))

    def test_decode_rejects_oversized_file_before_parsing(self):
        oversized = b"x" * (MAX_UPLOAD_BYTES + 1)
        with self.assertRaisesRegex(PortraitAnalysisError, "10 MB"):
            decode_uploaded_image(oversized)

    def test_decode_rejects_corrupted_image(self):
        with self.assertRaisesRegex(PortraitAnalysisError, "could not be decoded"):
            decode_uploaded_image(b"not an image")

    def test_decode_rejects_unsupported_image_format(self):
        source = Image.new("RGB", (160, 140), (120, 100, 90))
        output = BytesIO()
        source.save(output, format="BMP")
        with self.assertRaisesRegex(PortraitAnalysisError, "not a supported"):
            decode_uploaded_image(output.getvalue())

    def test_largest_face_selection_is_deterministic_and_clipped(self):
        faces = [
            (140, 20, 80, 80),
            (10, 20, 80, 80),
            (-10, 5, 40, 50),
        ]
        selected, usable, only_small = select_largest_face(
            faces,
            image_width=300,
            image_height=200,
        )
        self.assertEqual(selected, (10, 20, 80, 80))
        self.assertEqual(len(usable), 2)
        self.assertFalse(only_small)

    def test_small_faces_are_reported_as_unusable(self):
        selected, usable, only_small = select_largest_face(
            [(10, 10, 45, 45)],
            image_width=200,
            image_height=200,
        )
        self.assertIsNone(selected)
        self.assertEqual(usable, [])
        self.assertTrue(only_small)

    def test_invalid_face_crop_is_rejected(self):
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        with self.assertRaisesRegex(PortraitAnalysisError, "outside the image"):
            crop_face(image, (-10, 10, 100, 100))

    def test_skin_color_uses_face_patches_and_returns_measured_sample_counts(self):
        face = np.full((160, 160, 3), (178, 126, 96), dtype=np.uint8)
        color, cluster_pixels, sampled_pixels = extract_skin_color(face)
        self.assertEqual(color, (178, 126, 96))
        self.assertGreater(cluster_pixels, 0)
        self.assertGreaterEqual(sampled_pixels, cluster_pixels)

    def test_classification_returns_estimates_and_lab_measurement(self):
        estimate = classify_skin_color((178, 126, 96))
        self.assertEqual(estimate["tone"], "Medium")
        self.assertIn(estimate["undertone"], {"Warm", "Cool", "Neutral"})
        self.assertEqual(len(estimate["lab"]), 3)

    def test_classification_maps_clear_and_ambiguous_lab_casts(self):
        self.assertEqual(classify_skin_color((178, 126, 96))["undertone"], "Warm")
        self.assertEqual(classify_skin_color((130, 80, 70))["undertone"], "Cool")
        self.assertEqual(classify_skin_color((150, 140, 135))["undertone"], "Neutral")

    def test_analysis_reports_no_face_without_crashing(self):
        blank = np.zeros((320, 320, 3), dtype=np.uint8)
        with self.assertRaisesRegex(PortraitAnalysisError, "No usable face"):
            analyze_portrait(blank)

    def test_analysis_reports_progress_through_each_stage(self):
        image = np.full((240, 240, 3), (150, 120, 100), dtype=np.uint8)
        progress = []
        with (
            patch(
                "portrait_analysis.detect_face",
                return_value=((40, 40, 160, 160), [(40, 40, 160, 160)], False),
            ),
            patch("portrait_analysis.crop_face", return_value=image),
            patch(
                "portrait_analysis.extract_skin_color",
                return_value=((150, 120, 100), 120, 300),
            ),
            patch(
                "portrait_analysis.classify_skin_color",
                return_value={
                    "tone": "Medium",
                    "undertone": "Warm",
                    "lab": (50.0, 12.0, 18.0),
                },
            ),
            patch("portrait_analysis.assess_image_quality", return_value=[]),
        ):
            analyze_portrait(
                image,
                progress_callback=lambda value, message: progress.append(
                    (value, message)
                ),
            )

        self.assertEqual([value for value, _ in progress], [8, 42, 72, 100])
        self.assertIn("face", progress[0][1].lower())
        self.assertIn("Sampling", progress[1][1])
        self.assertIn("undertone", progress[2][1].lower())

    def test_quality_feedback_catches_underexposure_and_blur(self):
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        notes = assess_image_quality(image, (20, 20, 120, 120))
        self.assertTrue(any("dark or shadowed" in note for note in notes))
        self.assertTrue(any("blurry" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
