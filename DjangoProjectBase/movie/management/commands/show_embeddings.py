import numpy as np
from django.core.management.base import BaseCommand

from movie.models import Movie


class Command(BaseCommand):
    help = "Show stored embeddings from the database"

    def add_arguments(self, parser):
        parser.add_argument("--title", type=str, help="Filter by exact movie title (case-insensitive)")
        parser.add_argument("--limit", type=int, default=5, help="Max number of movies to show")
        parser.add_argument("--values", type=int, default=10, help="How many embedding values to print")

    def handle(self, *args, **kwargs):
        title = kwargs.get("title")
        limit = max(1, kwargs.get("limit", 5))
        values = max(1, kwargs.get("values", 10))

        movies = Movie.objects.all().order_by("id")
        if title:
            movies = movies.filter(title__iexact=title)

        if not movies.exists():
            self.stderr.write("No movies found for the given filters.")
            return

        shown = 0
        iterable = movies if title else movies[:limit]

        for movie in iterable:
            if not movie.emb:
                self.stderr.write(f"No embedding stored for: {movie.title}")
                continue

            try:
                vec, detected_dtype = self.decode_embedding(movie.emb)
            except ValueError as exc:
                self.stderr.write(f"Could not decode embedding for {movie.title}: {exc}")
                continue

            self.stdout.write(f"Title: {movie.title}")
            self.stdout.write(f"Shape: {vec.shape}")
            self.stdout.write(f"Stored dtype: {detected_dtype}")
            self.stdout.write(f"First {values} values: {vec[:values].tolist()}")
            self.stdout.write("-" * 60)
            shown += 1

        self.stdout.write(self.style.SUCCESS(f"Shown embeddings for {shown} movie(s)."))

    def decode_embedding(self, emb_bytes):
        byte_len = len(emb_bytes)

        # Preferred format from OpenAI output stored by movie_embeddings.py.
        if byte_len % 4 == 0:
            vec32 = np.frombuffer(emb_bytes, dtype=np.float32)
            if vec32.size == 1536:
                return vec32, "float32"

        # Fallback for older/default values saved as float64.
        if byte_len % 8 == 0:
            vec64 = np.frombuffer(emb_bytes, dtype=np.float64)
            return vec64, "float64"

        if byte_len % 4 == 0:
            return np.frombuffer(emb_bytes, dtype=np.float32), "float32"

        raise ValueError(f"unsupported byte length: {byte_len}")
