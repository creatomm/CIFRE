# myapp/views.py
from django.shortcuts import render, get_object_or_404, redirect
from .models import Document
from .forms import DocumentForm
from .utils import preprocess_text, generate_summary_supervised, generate_summary_unsupervised, generate_summary_t5

def document_list(request):
    documents = Document.objects.all()
    return render(request, 'myapp/document_list.html', {'documents': documents})

def document_detail(request, pk):
    document = get_object_or_404(Document, pk=pk)
    return render(request, 'myapp/document_detail.html', {'document': document})

def document_new(request):
    if request.method == "POST":
        form = DocumentForm(request.POST)
        if form.is_valid():
            document = form.save(commit=False)
            text = document.content
            document.summary_supervised = generate_summary_supervised(text)
            document.summary_unsupervised = generate_summary_unsupervised(text)
            document.summary_t5 = generate_summary_t5(text)
            document.save()
            return redirect('document_detail', pk=document.pk)
    else:
        form = DocumentForm()
    return render(request, 'myapp/document_edit.html', {'form': form})
