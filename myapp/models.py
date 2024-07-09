from django.db import models

class Document(models.Model):
    """
    Modèle représentant un document avec son titre, son contenu, et plusieurs types de résumés.

    Attributs :
    - title : Titre du document, de type CharField, limité à 200 caractères.
    - content : Contenu intégral du document, de type TextField.
    - summary_supervised : Résumé généré par un modèle supervisé, de type TextField, optionnel.
    - summary_unsupervised : Résumé généré par un modèle non supervisé, de type TextField, optionnel.
    - summary_t5 : Résumé généré par le modèle T5, de type TextField, optionnel.
    - summary_sumy : Résumé généré par le modèle Sumy, de type TextField, optionnel.
    """
    
    title = models.CharField(max_length=200)
    content = models.TextField()
    summary_supervised = models.TextField(blank=True, null=True)
    summary_unsupervised = models.TextField(blank=True, null=True)
    summary_t5 = models.TextField(blank=True, null=True)
    summary_sumy = models.TextField(blank=True, null=True)

    def __str__(self):
        """
        Retourne une représentation sous forme de chaîne de caractères de l'objet Document.
        Ici, nous choisissons de retourner le titre du document.
        """
        return self.title

