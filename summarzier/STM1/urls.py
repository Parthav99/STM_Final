from . import views
from django.urls import path,include
from django.contrib import admin

urlpatterns = [
  path('', views.home, name='home'),
  path('transcript/',views.transcript,name='transcript'),
  path('summarize/', views.summarize, name='summarize'),
  path('extractive_sum/',views.extractive_sum,name='extractive_sum'),
  path('extract_page/',views.extract_page,name='extract_page'),
  path('audio_page/',views.audio_page,name='audio_page'),
  path('audio_sum/',views.audio_sum,name='audio_sum')

]
