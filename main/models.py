from django.db import models
from django.contrib.auth import get_user_model


class Document(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    name = models.CharField(max_length=256)
    loc = models.CharField(max_length=512)
    public = models.BooleanField(default=False)
    description = models.TextField(max_length=1024, null=True)


class Vector_db(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    name = models.CharField(max_length=32)
    docs = models.ManyToManyField(Document, related_name='vector_dbs')
    loc = models.CharField(max_length=512)
    public = models.BooleanField(default=False)
    description = models.TextField(max_length=1024, null=True)



