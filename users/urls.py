from django.urls import path
from .views import *
# from .api.views import *
from rest_framework.authtoken.views import obtain_auth_token

app_name = 'users'

urlpatterns = [
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('profile/', profile_view, name='profile'),
    path('api-token-auth/', view=obtain_auth_token),
]