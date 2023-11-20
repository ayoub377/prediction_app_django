from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import LayoutNerInputSerializer
from .utils import format_for_layoutlm, layout_ner, get_results_json, post_process, split_json_data, delete_image


class LayoutNerAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):

        serializer = LayoutNerInputSerializer(data=request.FILES)

        if serializer.is_valid():
            image = request.FILES['image']
            data = format_for_layoutlm(image)

            data_bottom, data_top = split_json_data(data)

            top_predictions, top_true_boxes, top_trimmed_list = layout_ner.predict(data_top)
            bottom_predictions, bottom_true_boxes, bottom_trimmed_list = layout_ner.predict(data_bottom)

            top_true_confidence_scores, top_true_predictions_trimmed = post_process(top_predictions, top_trimmed_list)
            bottom_true_confidence_scores, bottom_true_predictions_trimmed = post_process(bottom_predictions,
                                                                                          bottom_trimmed_list)

            combined_predictions = bottom_true_predictions_trimmed + top_true_predictions_trimmed
            combined_boxes = bottom_true_boxes + top_true_boxes
            combined_confidence_scores = bottom_true_confidence_scores + top_true_confidence_scores

            combined_example = {
                "id": data_bottom["id"],
                "image": data_bottom["image"],
                "ner_tags": data_bottom["ner_tags"],
                "tokens": data_bottom["tokens"] + data_top["tokens"],
                "bboxes": combined_boxes
            }

            # remove the image from the directory
            delete_image(data["image"])

            # get the result as json
            result_json = get_results_json(combined_predictions, combined_confidence_scores, combined_example)

            return Response(result_json, status=status.HTTP_200_OK, content_type="application/json")

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
