# myapp/forms.py
from django import forms
from .models import Document

class DocumentForm(forms.ModelForm):
    """
    Formulaire pour créer ou modifier un document.

    Ce formulaire est basé sur le modèle Document et inclut les champs suivants :
    - title : Titre du document
    - content : Contenu du document

    Les étiquettes des champs sont personnalisées pour être en français.
    """

    class Meta:
        model = Document
        fields = ['title', 'content']
        labels = {
            'title': 'Titre',
            'content': 'Contenu',
        }

