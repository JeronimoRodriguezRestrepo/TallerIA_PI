from django.shortcuts import render
from django.http import HttpResponse
import os

from .models import Movie
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import io
import urllib, base64

def home(request):
    #return HttpResponse('<h1>Welcome to Home Page</h1>')
    #return render(request, 'home.html')
    #return render(request, 'home.html', {'name':'Paola Vallejo'})
    searchTerm = request.GET.get('searchMovie') # GET se usa para solicitar recursos de un servidor
    if searchTerm:
        movies = Movie.objects.filter(title__icontains=searchTerm)
    else:
        movies = Movie.objects.all()
    return render(request, 'home.html', {'searchTerm':searchTerm, 'movies':movies})


def about(request):
    #return HttpResponse('<h1>Welcome to About Page</h1>')
    return render(request, 'about.html')

def signup(request):
    email = request.GET.get('email') 
    return render(request, 'signup.html', {'email':email})


def statistics_view0(request):
    matplotlib.use('Agg')
    # Obtener todas las películas
    all_movies = Movie.objects.all()

    # Crear un diccionario para almacenar la cantidad de películas por año
    movie_counts_by_year = {}

    # Filtrar las películas por año y contar la cantidad de películas por año
    for movie in all_movies:
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    # Ancho de las barras
    bar_width = 0.5
    # Posiciones de las barras
    bar_positions = range(len(movie_counts_by_year))

    # Crear la gráfica de barras
    plt.bar(bar_positions, movie_counts_by_year.values(), width=bar_width, align='center')

    # Personalizar la gráfica
    plt.title('Movies per year')
    plt.xlabel('Year')
    plt.ylabel('Number of movies')
    plt.xticks(bar_positions, movie_counts_by_year.keys(), rotation=90)

    # Ajustar el espaciado entre las barras
    plt.subplots_adjust(bottom=0.3)

    # Guardar la gráfica en un objeto BytesIO
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Convertir la gráfica a base64
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')

    # Renderizar la plantilla statistics.html con la gráfica
    return render(request, 'statistics.html', {'graphic': graphic})

def statistics_view(request):
    matplotlib.use('Agg')
    # Gráfica de películas por año
    all_movies = Movie.objects.all()
    movie_counts_by_year = {}
    for movie in all_movies:
        print(movie.genre)
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    year_graphic = generate_bar_chart(movie_counts_by_year, 'Year', 'Number of movies')

    # Gráfica de películas por género
    movie_counts_by_genre = {}
    for movie in all_movies:
        # Obtener el primer género
        genres = movie.genre.split(',')[0].strip() if movie.genre else "None"
        if genres in movie_counts_by_genre:
            movie_counts_by_genre[genres] += 1
        else:
            movie_counts_by_genre[genres] = 1

    genre_graphic = generate_bar_chart(movie_counts_by_genre, 'Genre', 'Number of movies')

    return render(request, 'statistics.html', {'year_graphic': year_graphic, 'genre_graphic': genre_graphic})


def generate_bar_chart(data, xlabel, ylabel):
    keys = [str(key) for key in data.keys()]
    plt.bar(keys, data.values())
    plt.title('Movies Distribution')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic


def recommend_movie(request):
    search_prompt = request.GET.get('searchPrompt')
    best_movie = None
    best_similarity = None
    error_message = None

    if search_prompt:
        try:
            load_dotenv('key2_1.env')
            api_key = os.environ.get('openai_apikey')
            if not api_key:
                raise ValueError('OpenAI API key was not found in key2_1.env')

            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(
                input=[search_prompt],
                model='text-embedding-3-small'
            )
            prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)

            max_similarity = -1.0
            for movie in Movie.objects.all():
                if not movie.emb:
                    continue

                movie_emb = decode_embedding(movie.emb)
                if movie_emb is None or movie_emb.size != prompt_emb.size:
                    continue

                similarity = cosine_similarity(prompt_emb, movie_emb)
                if np.isnan(similarity):
                    continue

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_movie = movie
                    best_similarity = similarity

            if best_movie is None:
                error_message = 'No se encontro una pelicula con embedding valido para comparar.'

        except Exception as exc:
            error_message = f'Error al generar recomendacion: {exc}'

    return render(
        request,
        'recommendations.html',
        {
            'searchPrompt': search_prompt,
            'best_movie': best_movie,
            'best_similarity': best_similarity,
            'error_message': error_message,
        },
    )


def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def decode_embedding(emb_bytes):
    byte_len = len(emb_bytes)

    if byte_len % 4 == 0:
        emb32 = np.frombuffer(emb_bytes, dtype=np.float32)
        if emb32.size == 1536:
            return emb32

    if byte_len % 8 == 0:
        emb64 = np.frombuffer(emb_bytes, dtype=np.float64)
        return emb64.astype(np.float32)

    if byte_len % 4 == 0:
        return np.frombuffer(emb_bytes, dtype=np.float32)

    return None