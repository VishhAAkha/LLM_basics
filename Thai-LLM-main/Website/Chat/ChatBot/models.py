from django.forms import ModelForm
from django.db import models

class Prompt(models.Model):
    input_mssg = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.input_mssg