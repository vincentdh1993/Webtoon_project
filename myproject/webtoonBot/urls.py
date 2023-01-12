from django.urls import path
from . import views

urlpatterns = [
    path('', views.index ,name = 'index'),
    # path('ver1', views.ver1, name = 'ver1'),
    # path('ver2', views.ver2, name = 'ver2'),
    # path('ver3', views.ver3, name = 'ver3'),
    # path('ver4', views.ver4, name = 'ver4'),
    # path('ver4_result',views.ver4_result,name='ver4_result'),
]