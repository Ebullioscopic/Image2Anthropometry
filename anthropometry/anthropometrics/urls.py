from django.urls import path
from .views import analyze_pose

urlpatterns = [
    path('analyze/', analyze_pose, name='analyze_pose'),
]
