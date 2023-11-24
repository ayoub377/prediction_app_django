from rest_framework import serializers


class LayoutNerInputSerializer(serializers.Serializer):
    image = serializers.ImageField()
    # Add other fields as needed (tokens, bboxes, etc.)


class PredictionSerializer(serializers.Serializer):
    Word = serializers.CharField()
    Predicted_Label = serializers.CharField()
    Confidence_Score = serializers.FloatField()
