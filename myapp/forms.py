# myapp/forms.py
from django import forms
from .models import Document

class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ['title', 'content']
        labels = {
            'title': 'Titre',
            'content': 'Contenu',
        }

