
import uuid

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

img_path = r"C:\Users\G\Documents\prediction_app_django\prediction_app-master\prediction_django\media\boxy_dom.png"


def scale_bbox_coordinates(bboxes, image_width, image_height, scaled_min, scaled_max):
    return [
        [
            int((coord / image_width) * scaled_max) if i % 2 == 0 else int((coord / image_height) * scaled_max)
            for i, coord in enumerate(bbox)
        ]
        for bbox in bboxes
    ]


def divide_image_into_segments(image, num_segments):
    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Get the shape of the image array
    height, width, _ = img_array.shape

    # Calculate the height of each segment
    segment_height = height // num_segments

    # Initialize a list to store the segmented images
    segments = []

    # Iterate over the number of segments
    for i in range(num_segments):
        # Calculate the start and end indices for the current segment
        start_idx = i * segment_height
        end_idx = (i + 1) * segment_height if i < num_segments - 1 else height

        # Extract the current segment from the image array
        segment = img_array[start_idx:end_idx, :, :]

        # Append the segment to the list
        segments.append(segment)

    return segments


def process_ocr(image_segment):
    # Obtain OCR results
    ocr = PaddleOCR(lang="fr", use_angle_cls=True, enable_mkldnn=True)
    ocr_result = ocr.ocr(image_segment)
    return ocr_result


def perform_ocr():
    ocr_results = []
    image = Image.open(img_path)
    # segments = divide_image_into_segments(image, 5)
    #
    # # Initialize a ThreadPoolExecutor for parallel processing
    # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    #     # Use the executor to process OCR for each image segment in parallel
    #     ocr_results = list(executor.map(process_ocr, segments))

    ocr_res = process_ocr(image)
    print(ocr_res)
    # lists of tokens and bboxes
    # tokens = []
    # bboxes = []
    #
    # for item in ocr_results:
    #     for bbox, (text, confidence) in item:
    #         # Append token and bbox to respective lists
    #         tokens.append(text)
    #         # Convert bbox from [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] to [x1, y1, x2, y2]
    #         flattened_bbox = [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]]
    #         bboxes.append(flattened_bbox)
    #
    # # Scale the bboxes
    # image_width, image_height = image.size
    # scaled_bboxes = scale_bbox_coordinates(bboxes, image_width, image_height, scaled_min=0, scaled_max=1000)
    # print(f"tokens are: {tokens} and bboxes are: {bboxes}")
    # return tokens, scaled_bboxes


# Placeholder for ner_tags, replace with actual NER predictions from a pretrained model
ner_tags = ['O', 'Ref', 'NumFa', 'Fourniss', 'DateFa', 'DateLim', 'TotalHT', 'TVA', 'TotalTTc', 'unitP', 'Qt',
            'TVAP', 'descp']

if __name__ == '__main__':
    data_id = uuid.uuid4().hex
    tokens, bboxes = perform_ocr()
    formatted_data = {
        'id': data_id,
        'image': img_path,  # Update the image path or use other appropriate naming
        'bboxes': bboxes,
        'ner_tags': ner_tags,
        'tokens': tokens
    }
