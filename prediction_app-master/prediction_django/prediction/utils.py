import json
import os
import uuid

import numpy as np
from PIL import Image
from django.conf import settings
from paddleocr import PaddleOCR

from .layoutMner import LayoutLMNER
from .serializers import PredictionSerializer


# fonction de scaling des bboxes
def scale_bbox_coordinates(bboxes, image_width, image_height, scaled_min, scaled_max):
    return [
        [
            int((coord / image_width) * scaled_max) if i % 2 == 0 else int((coord / image_height) * scaled_max)
            for i, coord in enumerate(bbox)
        ]
        for bbox in bboxes
    ]


# etape d'OCR
def ocr_and_scale_bboxes(image_path, scaled_min=0, scaled_max=1000):
    img = Image.open(image_path)

    # Obtain OCR results
    ocr = PaddleOCR(lang="fr", use_angle_cls=True, enable_mkldnn=True)
    img_array = np.array(img)
    ocr_result = ocr.ocr(img_array)

    tokens = []
    bboxes = []

    for item in ocr_result:
        for bbox, (text, confidence) in item:
            # Append token and bbox to respective lists
            tokens.append(text)
            # Convert bbox from [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] to [x1, y1, x2, y2]
            flattened_bbox = [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]]
            bboxes.append(flattened_bbox)

    # Scale the bboxes
    image_width, image_height = img.size
    scaled_bboxes = scale_bbox_coordinates(bboxes, image_width, image_height, scaled_min, scaled_max)
    return tokens, scaled_bboxes


# fonction qui retourne le data sous forme de json objet
def format_for_layoutlm(image_data):
    # verifie si l image va s ouvrir
    try:
        # Use PIL to open the image from raw data
        image = Image.open(image_data)
    except Exception as e:
        print(f"Error opening the image: {str(e)}")
        return None

    # Placeholder for ner_tags, replace with actual NER predictions from a pretrained model
    ner_tags = ['O', 'Ref', 'NumFa', 'Fourniss', 'DateFa', 'DateLim', 'TotalHT', 'TVA', 'TotalTTc', 'unitP', 'Qt',
                'TVAP', 'descp']

    # Generate a unique ID for the data
    data_id = uuid.uuid4().hex

    try:
        # enregistre l image
        temp_image_path = os.path.join(settings.MEDIA_ROOT, f'image_{data_id}.png')
        image.save(temp_image_path)

        # Converti les tokens et bboxes
        tokens, bboxes = ocr_and_scale_bboxes(temp_image_path)

        # format correcte des donnees
        formatted_data = {
            'id': data_id,
            'image': temp_image_path,  # Update the image path or use other appropriate naming
            'bboxes': bboxes,
            'ner_tags': ner_tags,
            'tokens': tokens
        }

        print("ocr process done")
        # Retourne les  donnees
        return formatted_data

    except Exception as e:
        print(f"Error processing Json Data: {str(e)}")
        return None


# diviser le json en deux parties
def split_json_data(data):
    # Define the criteria for splitting (e.g., based on the length of words or any other criteria)
    split_point = len(data["tokens"]) // 2  # Splitting based on the number of words

    # Split the data into "top" and "bottom" parts along with their words and bboxes
    top_data = {
        "image": data["image"],
        "id": data["id"],
        "ner_tags": data["ner_tags"],
        "tokens": data["tokens"][:split_point],
        "bboxes": data["bboxes"][:split_point]
    }

    bottom_data = {
        "image": data["image"],
        "id": data["id"],
        "ner_tags": data["ner_tags"],
        "tokens": data["tokens"][split_point:],
        "bboxes": data["bboxes"][split_point:]
    }

    return top_data, bottom_data


# fonction qui traite le json
def post_process(true_predictions, trimmed_list):
    true_confidence_scores = []
    true_predictions_trimmed = true_predictions[1:-1]
    for idx, pred in enumerate(true_predictions_trimmed):
        true_confidence_scores.append((pred, trimmed_list[idx]))

    return true_confidence_scores, true_predictions_trimmed


def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# fonction qui retourne les resultats de prediction
def get_results_json(true_predictions_trimmed_par, true_confidence_scores_par, example_par):
    # Create a list to store dictionaries representing word-label pairs and confidence scores
    word_confidence_list = []

    for idx, (word, prediction) in enumerate(zip(example_par['tokens'], true_predictions_trimmed_par)):
        if word not in word_confidence_list and prediction != 'O':
            if prediction == 'O':
                predicted_label = 'other'
            else:
                predicted_label = prediction

            confidence_score = true_confidence_scores_par[idx][1] if idx < len(true_confidence_scores_par) else 0.0

            # Create a dictionary for each word-label pair
            word_data = {
                'Word': word,
                'Predicted_Label': predicted_label.lower(),
                'Confidence_Score': confidence_score
            }

            word_confidence_list.append(word_data)

    # Filter out labels 'other' and 'o'
    filtered_word_confidence_list = [data for data in word_confidence_list if
                                     data['Predicted_Label'] != 'other' and data['Predicted_Label'] != 'o']

    # Use the serializer to convert the list of dictionaries to JSON
    serializer = PredictionSerializer(filtered_word_confidence_list, many=True)
    serialized_data = serializer.data

    return serialized_data


# delete image after processing
def delete_image(temp_path):
    # Create a temporary file path
    os.remove(temp_path)
    print(f"image deleted at following path: {temp_path}")


model_path = "ineoApp/LayoutLMv3_5_entities_filtred_14"
layout_ner = LayoutLMNER(model_path)
