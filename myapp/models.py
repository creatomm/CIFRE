# myapp/models.py
"""from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    summary_supervised = models.TextField(blank=True, null=True)
    summary_unsupervised = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
"""

# myapp/models.py
from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    summary_supervised = models.TextField(blank=True, null=True)
    summary_unsupervised = models.TextField(blank=True, null=True)
    summary_t5 = models.TextField(blank=True, null=True)  # Ajout de cette ligne

    def __str__(self):
        return self.title

