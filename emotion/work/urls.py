from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('login/', views.login),
    path('strescan/<uid>', views.scan),
    path('adminhome/', views.adhome),
    path('addstaff/', views.addstaff),
    path('mstaff/', views.mstaff),
    path('report/<uid>', views.report),
    path('staffdata', views.stafhome),
    path('stanalyse', views.stanalyse),
    path('addtips/', views.addtips),
    path('delstf', views.delstf),
    path('pic', views.pic),
    path('video', views.video),
    path('picanalyse', views.picanalyse),
    path('videoanalyse', views.videoanalyse),
    path('logout', views.home),
    path('stafhome', views.stafhome),
]
