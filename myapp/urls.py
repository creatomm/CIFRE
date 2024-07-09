from django.urls import path
from . import views

# Définition des routes URL de l'application
urlpatterns = [
    # Route pour la liste des documents
    path('', views.document_list, name='document_list'),
    
    # Route pour afficher les détails d'un document spécifique
    path('document/<int:pk>/', views.document_detail, name='document_detail'),
    
    # Route pour créer un nouveau document
    path('document/new/', views.document_new, name='document_new'),
    
    # Route pour éditer un document existant
    path('document/<int:pk>/edit/', views.document_edit, name='document_edit'),
    
    # Route pour supprimer un document existant
    path('document/<int:pk>/delete/', views.document_delete, name='document_delete'),
]

