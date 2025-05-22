from django.urls import path
from . import views
from django.views.decorators.csrf import csrf_exempt

# URLConf
urlpatterns = [
    path('hello/', views.say_hello),
    path('receive_form/', csrf_exempt(views.receive_form)),
    path('receive_rate/', csrf_exempt(views.receive_rate)),
    path('receive_email/', csrf_exempt(views.receive_email)),
    path('load_audios/', csrf_exempt(views.load_audios)),
]