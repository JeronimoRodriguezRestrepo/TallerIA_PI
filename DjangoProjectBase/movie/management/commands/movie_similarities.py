import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv


class Command(BaseCommand):
    help = "Compare two movies and optionally a prompt using OpenAI embeddings"

    def add_arguments(self, parser):
        parser.add_argument("--movie1", type=str, help="Title of the first movie")
        parser.add_argument("--movie2", type=str, help="Title of the second movie")
        parser.add_argument(
            "--prompt",
            type=str,
            default="pelicula sobre la Segunda Guerra Mundial",
            help="Prompt text to compare against movie descriptions",
        )

    def handle(self, *args, **kwargs):
        # ✅ Load OpenAI API key
        load_dotenv('key2_1.env')
        client = OpenAI(api_key=os.environ.get('openai_apikey'))

        movie1_title = kwargs.get("movie1")
        movie2_title = kwargs.get("movie2")
        prompt = kwargs.get("prompt")

        movie1, movie2 = self.select_movies(movie1_title, movie2_title)
        if not movie1 or not movie2:
            return

        def get_embedding(text):
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return np.array(response.data[0].embedding, dtype=np.float32)

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # ✅ Generate embeddings of both movies
        emb1 = get_embedding(self.get_text_for_embedding(movie1))
        emb2 = get_embedding(self.get_text_for_embedding(movie2))

        # ✅ Compute similarity between movies
        similarity = cosine_similarity(emb1, emb2)
        self.stdout.write(f"\U0001F3AC Similaridad entre '{movie1.title}' y '{movie2.title}': {similarity:.4f}")

        # ✅ Optional: Compare against a prompt
        prompt_emb = get_embedding(prompt)

        sim_prompt_movie1 = cosine_similarity(prompt_emb, emb1)
        sim_prompt_movie2 = cosine_similarity(prompt_emb, emb2)

        self.stdout.write(f"\U0001F4DD Similitud prompt vs '{movie1.title}': {sim_prompt_movie1:.4f}")
        self.stdout.write(f"\U0001F4DD Similitud prompt vs '{movie2.title}': {sim_prompt_movie2:.4f}")

    def select_movies(self, movie1_title, movie2_title):
        movies = Movie.objects.all()
        if movies.count() < 2:
            self.stderr.write("Se necesitan al menos 2 peliculas en la base de datos.")
            return None, None

        movie1 = None
        movie2 = None

        if movie1_title:
            movie1 = Movie.objects.filter(title__iexact=movie1_title).first()
            if not movie1:
                self.stderr.write(f"No se encontro movie1: {movie1_title}")

        if movie2_title:
            movie2 = Movie.objects.filter(title__iexact=movie2_title).first()
            if not movie2:
                self.stderr.write(f"No se encontro movie2: {movie2_title}")

        if not movie1 or not movie2:
            fallback = list(movies.order_by("id")[:2])
            movie1 = movie1 or fallback[0]
            movie2 = movie2 or fallback[1]
            self.stdout.write(
                self.style.WARNING(
                    f"Usando peliculas por defecto: '{movie1.title}' y '{movie2.title}'"
                )
            )

        if movie1.id == movie2.id:
            alt_movie = movies.exclude(id=movie1.id).order_by("id").first()
            if not alt_movie:
                self.stderr.write("No hay una segunda pelicula distinta para comparar.")
                return None, None
            movie2 = alt_movie

        return movie1, movie2

    def get_text_for_embedding(self, movie):
        text = (movie.description or "").strip()
        if text:
            return text
        return movie.title