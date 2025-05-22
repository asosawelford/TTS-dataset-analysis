from django.db import models
from django.contrib.auth.models import AbstractBaseUser
from django.contrib.auth.models import PermissionsMixin
from django.contrib.auth.models import BaseUserManager


class StimuliRating(models.Model):
    user_id = models.CharField(primary_key=True, max_length=10)
    stimuli = models.CharField(max_length=256, blank=True, null=True)
    stimuli_rating = models.CharField(max_length=1, blank=True, null=True)

class UserData(models.Model):
    """Database model for user form data"""
    user_id = models.ForeignKey(StimuliRating, models.DO_NOTHING, db_column='user_id')
    user_number = models.IntegerField(unique=True, blank=True, null=True)
    age = models.IntegerField(blank=True, null=True)
    sex = models.CharField(max_length=1, blank=True, null=True)
    tts_familiarity = models.CharField(max_length=45, blank=True, null=True)
    headphone_type = models.CharField(max_length=45, blank=True, null=True)
    nationality_province = models.CharField(max_length=45, blank=True, null=True)


