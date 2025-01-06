"""ProductRecommendation URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from ProductRecommendationtionApp.views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',index),
    path('user/',user),
    path('registration/',registration),
    path('saveUser/',saveUser),
    path('userlogin/',userlogin),
    path('logout/',index),
    path('homepage/',homepage, name="homepage"),
    path('testimage/',home),
    path('adminLogin/',adminLogin),
    path('adminlogin/',adminlogin),
    path('home/',homePage),
    path('addProduct/',addProduct), # type: ignore
    path('getProductInfo/',getProductInfo), # type: ignore
    path('view/',viewProduct),
    path('changeStatus/<int:id>/<int:category>',changeStatus),
    path('delete/<int:id>',deleteNow),
    path('uploadImage/',uploadImage),
    path('testimage/',home),
    path('testagain/',testagain),
    path('orderNow/<int:id>',orderNow),
    path('getProductInfo/',getProductInfo),
    path('updateProduct/',updateProduct),
    path('orderProduct/',orderProduct),
    path('history/',history),
    path('orders/',orders),
    path('capture_image/',capture_image),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
