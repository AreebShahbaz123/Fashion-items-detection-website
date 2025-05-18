from django.shortcuts import render
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .utils import predict_image
import os

# Create your views here.

def home(request):
    return render(request, 'main/home.html')

def upload_image(request):
    if request.method == 'POST':
        try:
            # Get the uploaded image
            image_file = request.FILES.get('image')
            if not image_file:
                return JsonResponse({'error': 'No image uploaded'}, status=400)

            # Save the image temporarily
            path = default_storage.save('temp/' + image_file.name, ContentFile(image_file.read()))
            temp_path = default_storage.path(path)

            # Get prediction
            predicted_class, confidence = predict_image(temp_path)

            # Clean up the temporary file
            default_storage.delete(path)

            if predicted_class is not None and confidence is not None:
                return JsonResponse({
                    'success': True,
                    'class': predicted_class,
                    'confidence': confidence
                })
            else:
                return JsonResponse({'error': 'Failed to process image'}, status=500)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return render(request, 'main/upload.html')
