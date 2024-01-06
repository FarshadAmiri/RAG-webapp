from django.urls import path
from .views import *
from rest_framework.authtoken.views import obtain_auth_token
from .views import *

app_name = 'main'

urlpatterns = [
    path('', chat_view, name='chat'),
]