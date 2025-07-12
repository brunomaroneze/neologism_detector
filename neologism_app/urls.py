# neologism_app/urls.py

from django.urls import path
from . import views

app_name = 'neologism_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('results/', views.results, name='results'),
    path('validate_neologism/', views.validate_neologism, name='validate_neologism'),
    path('export_csv/', views.export_csv, name='export_csv'),
]