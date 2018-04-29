from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.http import QueryDict

def index(request):
	return render(request, 'index.html')

def splitDoc(request):
	doc = request.GET['data']
	rawParagraphs = doc.split('\n')
	paragraphs = []
	for paragraph in rawParagraphs:
		if paragraph != "":
			paragraphs.append(paragraph)
	response = JsonResponse({'paragraphs': paragraphs})
	return response
