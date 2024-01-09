from django.shortcuts import render
from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.urls import reverse ,reverse_lazy
from django.http import Http404
from django.http import HttpResponseRedirect, HttpResponse
from django.contrib.auth import authenticate, login, logout
from django.views.generic import UpdateView, DetailView, DeleteView
from django.contrib.auth.models import Group
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
# from rest_framework.views import APIView
# from rest_framework.generics import GenericAPIView
# from rest_framework.permissions import IsAuthenticated, IsAuthenticatedOrReadOnly
# from rest_framework.response import Response
from .models import *
from .forms import *

def login_view(request, *kwargs):
    if request.method =='GET':
        login_form = LoginForm(initial={"username": "", "password": ""})
        return render(request, 'Login_page.html', context={"form": login_form})
    elif request.method =='POST':
        form = LoginForm(request.POST)
        print(f"\n\nis form valid: {form.is_valid()}")
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect(reverse('main:main_chat'))
            messages.warning(request, 'Invalid username or password!')
            login_form = LoginForm(initial={"username": username, "password": password})
            return render(request, 'Login_page.html', context={"form": login_form,})
        messages.warning(request, 'Wrong captcha!')
        login_form = LoginForm(initial={"username": "", "password": ""})
        return render(request, 'Login_page.html', context={"form": login_form,})


def logout_view(request):
    logout(request)
    return HttpResponseRedirect(reverse('main:main_chat'))


def profile_view(request):
    if request.method == 'GET':
        if request.user.is_authenticated:
            # user_profile = get_object_or_404(Profile, username=request.user.username)
            return render (request, 'Profile_page.html', {'user': request.user
                                                              # ,'user_profile': user_profile
                                                              })
        return HttpResponse('You Must Log In First')