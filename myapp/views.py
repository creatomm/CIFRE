from django.shortcuts import render, get_object_or_404, redirect
from .models import Document
from .forms import DocumentForm
from .utils import preprocess_text, generate_summary_supervised, generate_summary_unsupervised, generate_summary_t5, generate_summary_sumy

# Vue pour lister tous les documents
def document_list(request):
    """
    Affiche la liste de tous les documents.
    """
    documents = Document.objects.all()
    return render(request, 'myapp/document_list.html', {'documents': documents})

# Vue pour afficher les détails d'un document spécifique
def document_detail(request, pk):
    """
    Affiche les détails d'un document spécifique.
    
    Args:
        request: La requête HTTP.
        pk: La clé primaire du document à afficher.
    """
    document = get_object_or_404(Document, pk=pk)
    return render(request, 'myapp/document_detail.html', {'document': document})

# Vue pour créer un nouveau document
def document_new(request):
    """
    Permet de créer un nouveau document. 
    Génère des résumés (supervisé, non supervisé, T5 et Sumy) pour le contenu du document.
    
    Args:
        request: La requête HTTP.
    """
    if request.method == "POST":
        form = DocumentForm(request.POST)
        if form.is_valid():
            document = form.save(commit=False)
            text = document.content
            document.summary_supervised = generate_summary_supervised(text)
            document.summary_unsupervised = generate_summary_unsupervised(text)
            document.summary_t5 = generate_summary_t5(text)
            document.summary_sumy = generate_summary_sumy(text)
            document.save()
            return redirect('document_detail', pk=document.pk)
    else:
        form = DocumentForm()
    return render(request, 'myapp/document_edit.html', {'form': form})

# Vue pour modifier un document existant
def document_edit(request, pk):
    """
    Permet de modifier un document existant.
    Régénère les résumés (supervisé, non supervisé, T5 et Sumy) pour le contenu mis à jour du document.
    
    Args:
        request: La requête HTTP.
        pk: La clé primaire du document à modifier.
    """
    document = get_object_or_404(Document, pk=pk)
    if request.method == "POST":
        form = DocumentForm(request.POST, instance=document)
        if form.is_valid():
            document = form.save(commit=False)
            text = document.content
            document.summary_supervised = generate_summary_supervised(text)
            document.summary_unsupervised = generate_summary_unsupervised(text)
            document.summary_t5 = generate_summary_t5(text)
            document.summary_sumy = generate_summary_sumy(text)
            document.save()
            return redirect('document_detail', pk=document.pk)
    else:
        form = DocumentForm(instance=document)
    return render(request, 'myapp/document_edit.html', {'form': form})

# Vue pour supprimer un document
def document_delete(request, pk):
    """
    Permet de supprimer un document existant.
    
    Args:
        request: La requête HTTP.
        pk: La clé primaire du document à supprimer.
    """
    document = get_object_or_404(Document, pk=pk)
    document.delete()
    return redirect('document_list')

