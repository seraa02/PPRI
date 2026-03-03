#!/usr/bin/env python3
"""
create_mock_pdfs.py
--------------------
Generates realistic mock survey PDFs for testing the pipeline.
Puts 3 PRE PDFs and 2 POST PDFs in the respective folders.

Run with:
    .venv/bin/python create_mock_pdfs.py
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io, sys

try:
    from reportlab.pdfgen import canvas as rl_canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def draw_form_image(title: str, participant: dict, questions: list) -> Image.Image:
    """Draw a survey form as a PIL Image (simulates a scanned form)."""
    W, H = 1700, 2200
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to use a system font; fall back to default
    try:
        font_lg = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        font_md = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 34)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except Exception:
        font_lg = font_md = font_sm = ImageFont.load_default()

    # Header
    draw.rectangle([40, 40, W - 40, 160], outline=(0, 0, 0), width=3)
    draw.text((W // 2, 100), title, fill=(0, 0, 0), font=font_lg, anchor="mm")

    # Personal info section
    y = 200
    draw.text((80, y), f"Name: {participant['name']}", fill=(0, 0, 0), font=font_md)
    draw.text((900, y), f"Date: {participant['date']}", fill=(0, 0, 0), font=font_md)
    y += 60
    draw.text((80, y), f"Participant ID: {participant['id']}", fill=(0, 0, 0), font=font_md)
    draw.text((900, y), f"Center: {participant['center']}", fill=(0, 0, 0), font=font_md)
    draw.line([40, y + 55, W - 40, y + 55], fill=(0, 0, 0), width=2)

    # Questions
    y += 100
    for i, q in enumerate(questions, 1):
        q_text = q["text"]
        q_type = q["type"]
        answer = q["answer"]

        draw.text((80, y), f"Q{i}. {q_text}", fill=(0, 0, 0), font=font_sm)
        y += 45

        if q_type == "checkbox":
            for option in q["options"]:
                box_x, box_y = 120, y
                draw.rectangle([box_x, box_y, box_x + 28, box_y + 28], outline=(0, 0, 0), width=2)
                if option == answer:
                    # Draw a check mark inside the box
                    draw.line([box_x + 4, box_y + 14, box_x + 12, box_y + 24], fill=(0, 0, 0), width=3)
                    draw.line([box_x + 12, box_y + 24, box_x + 26, box_y + 6], fill=(0, 0, 0), width=3)
                draw.text((box_x + 38, box_y), option, fill=(0, 0, 0), font=font_sm)
                y += 42

        elif q_type == "scale":
            draw.text((120, y), "Circle a number:  0  1  2  3  4  5  6  7  8  9  10", fill=(0, 0, 0), font=font_sm)
            # Draw circle around the selected number
            num_positions = {
                "0": 320, "1": 358, "2": 396, "3": 434, "4": 472,
                "5": 510, "6": 548, "7": 586, "8": 624, "9": 662, "10": 700,
            }
            if answer in num_positions:
                cx = num_positions[answer]
                draw.ellipse([cx - 18, y - 5, cx + 18, y + 38], outline=(0, 0, 0), width=3)
            y += 55

        elif q_type == "true_false":
            draw.text((120, y), "True  /  False", fill=(0, 0, 0), font=font_sm)
            if answer.lower() == "true":
                draw.ellipse([118, y - 4, 192, y + 38], outline=(0, 0, 0), width=3)
            else:
                draw.ellipse([220, y - 4, 308, y + 38], outline=(0, 0, 0), width=3)
            y += 55

        elif q_type == "handwritten":
            draw.line([120, y + 40, 1600, y + 40], fill=(180, 180, 180), width=1)
            draw.text((120, y + 8), answer, fill=(30, 30, 30), font=font_sm)
            y += 60

        y += 20  # spacing between questions

    # Footer
    draw.rectangle([40, H - 80, W - 40, H - 40], outline=(200, 200, 200), width=1)
    draw.text((W // 2, H - 60), "MockStudy Survey Form — For Research Use Only",
              fill=(150, 150, 150), font=font_sm, anchor="mm")

    return img


def save_as_pdf(image: Image.Image, path: Path) -> None:
    """Save a PIL Image as a single-page PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path), "PDF", resolution=150)
    print(f"  Created: {path}")


def main():
    print("Generating mock survey PDFs for testing ...\n")

    # ----------------------------------------------------------------
    # PRE survey participants
    # ----------------------------------------------------------------
    pre_data = [
        {
            "participant": {"name": "Maria Garcia", "date": "02/15/2025",
                            "id": "P001", "center": "Austin Senior Center"},
            "questions": [
                {"text": "Do you exercise at least 3 times per week?",
                 "type": "checkbox", "options": ["Yes", "No", "Sometimes"], "answer": "Sometimes"},
                {"text": "How would you rate your balance? (0=poor, 10=excellent)",
                 "type": "scale", "options": [], "answer": "4"},
                {"text": "Have you fallen in the past 6 months?",
                 "type": "true_false", "options": [], "answer": "True"},
                {"text": "What activities do you enjoy most?",
                 "type": "handwritten", "options": [], "answer": "Walking and gardening"},
                {"text": "Do you use a walking aid?",
                 "type": "checkbox", "options": ["Cane", "Walker", "None"], "answer": "None"},
            ],
        },
        {
            "participant": {"name": "James Williams", "date": "02/16/2025",
                            "id": "P002", "center": "Houston Senior Center"},
            "questions": [
                {"text": "Do you exercise at least 3 times per week?",
                 "type": "checkbox", "options": ["Yes", "No", "Sometimes"], "answer": "No"},
                {"text": "How would you rate your balance? (0=poor, 10=excellent)",
                 "type": "scale", "options": [], "answer": "6"},
                {"text": "Have you fallen in the past 6 months?",
                 "type": "true_false", "options": [], "answer": "False"},
                {"text": "What activities do you enjoy most?",
                 "type": "handwritten", "options": [], "answer": "Reading and TV"},
                {"text": "Do you use a walking aid?",
                 "type": "checkbox", "options": ["Cane", "Walker", "None"], "answer": "Cane"},
            ],
        },
        {
            "participant": {"name": "Rosa Martinez", "date": "02/17/2025",
                            "id": "P003", "center": "San Antonio Senior Center"},
            "questions": [
                {"text": "Do you exercise at least 3 times per week?",
                 "type": "checkbox", "options": ["Yes", "No", "Sometimes"], "answer": "Yes"},
                {"text": "How would you rate your balance? (0=poor, 10=excellent)",
                 "type": "scale", "options": [], "answer": "8"},
                {"text": "Have you fallen in the past 6 months?",
                 "type": "true_false", "options": [], "answer": "False"},
                {"text": "What activities do you enjoy most?",
                 "type": "handwritten", "options": [], "answer": "Yoga y caminatas"},
                {"text": "Do you use a walking aid?",
                 "type": "checkbox", "options": ["Cane", "Walker", "None"], "answer": "None"},
            ],
        },
    ]

    for i, data in enumerate(pre_data, 1):
        img = draw_form_image(
            title="MockStudy — PRE Survey",
            participant=data["participant"],
            questions=data["questions"],
        )
        save_as_pdf(img, Path(f"PRE/participant_{i:03d}_pre.pdf"))

    # ----------------------------------------------------------------
    # POST survey participants
    # ----------------------------------------------------------------
    post_data = [
        {
            "participant": {"name": "Maria Garcia", "date": "05/15/2025",
                            "id": "P001", "center": "Austin Senior Center"},
            "questions": [
                {"text": "Do you exercise at least 3 times per week?",
                 "type": "checkbox", "options": ["Yes", "No", "Sometimes"], "answer": "Yes"},
                {"text": "How would you rate your balance now? (0=poor, 10=excellent)",
                 "type": "scale", "options": [], "answer": "7"},
                {"text": "Have you fallen in the past 3 months?",
                 "type": "true_false", "options": [], "answer": "False"},
                {"text": "Did the program help you? Please describe.",
                 "type": "handwritten", "options": [], "answer": "Yes it helped a lot"},
            ],
        },
        {
            "participant": {"name": "James Williams", "date": "05/16/2025",
                            "id": "P002", "center": "Houston Senior Center"},
            "questions": [
                {"text": "Do you exercise at least 3 times per week?",
                 "type": "checkbox", "options": ["Yes", "No", "Sometimes"], "answer": "Sometimes"},
                {"text": "How would you rate your balance now? (0=poor, 10=excellent)",
                 "type": "scale", "options": [], "answer": "8"},
                {"text": "Have you fallen in the past 3 months?",
                 "type": "true_false", "options": [], "answer": "False"},
                {"text": "Did the program help you? Please describe.",
                 "type": "handwritten", "options": [], "answer": "Feeling more confident"},
            ],
        },
    ]

    for i, data in enumerate(post_data, 1):
        img = draw_form_image(
            title="MockStudy — POST Survey",
            participant=data["participant"],
            questions=data["questions"],
        )
        save_as_pdf(img, Path(f"POST/participant_{i:03d}_post.pdf"))

    print(f"\n✓ Created 3 PRE and 2 POST mock survey PDFs.")
    print("  Ready to run: .venv/bin/python main.py")


if __name__ == "__main__":
    main()
