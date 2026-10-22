DISEASE_INFO = {
    "acne": {
        "name": "Acne",
        "description": "Acne is a common skin condition that causes pimples, blackheads, and whiteheads.",
        "advice": "Use gentle skin care and consult a dermatologist if symptoms persist."
    },
    "eczema": {
        "name": "Eczema",
        "description": "Eczema can cause dry, itchy, and inflamed skin.",
        "advice": "Use a fragrance-free moisturizer and seek medical advice if symptoms worsen."
    },
    "psoriasis": {
        "name": "Psoriasis",
        "description": "Psoriasis can cause thick, scaly patches on the skin.",
        "advice": "Consult a dermatologist for appropriate evaluation and treatment."
    },
    "normal": {
        "name": "No recognized condition",
        "description": "The AI did not identify a listed skin condition.",
        "advice": "This does not rule out a skin disease. Seek medical advice if you have concerns."
    }
}


def get_disease_info(disease_name):
    """Return general information for a disease label."""
    key = str(disease_name).strip().lower().replace(" ", "_")

    if key in DISEASE_INFO:
        return DISEASE_INFO[key]

    return {
        "name": str(disease_name),
        "description": "No additional information is available for this label.",
        "advice": "Consult a qualified dermatologist for diagnosis and treatment."
    }