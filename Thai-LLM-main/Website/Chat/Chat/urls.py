from django.contrib import admin
from django.urls import path,include
# from ChatBot import 
from ChatBot import urls, views

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('', include('Chatbot.urls')),
    # path('chatbot/', include('Chatbot.urls')),
    path('',views.index,name='index'),
    # path('example/', example_view, name='example'),
]