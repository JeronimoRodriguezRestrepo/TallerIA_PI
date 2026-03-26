import os
import unicodedata

from django.core.management.base import BaseCommand

from movie.models import Movie


class Command(BaseCommand):
	help = "Assign movie images from media/movie/images and update DB image paths"

	def handle(self, *args, **kwargs):
		images_folder = "media/movie/images"

		if not os.path.exists(images_folder):
			self.stderr.write(f"Folder not found: {images_folder}")
			return

		image_files = self.get_image_files(images_folder)
		if not image_files:
			self.stderr.write("No image files found in media/movie/images")
			return

		image_index = self.build_image_index(image_files)
		movies = Movie.objects.all()

		updated_count = 0
		not_found_count = 0

		self.stdout.write(f"Found {movies.count()} movies")
		self.stdout.write(f"Found {len(image_files)} image files")

		for movie in movies:
			image_filename = self.find_image_for_title(movie.title, image_index)

			if not image_filename:
				not_found_count += 1
				self.stderr.write(f"Image not found for: {movie.title}")
				continue

			movie.image = os.path.join("movie/images", image_filename).replace("\\", "/")
			movie.save(update_fields=["image"])
			updated_count += 1
			self.stdout.write(self.style.SUCCESS(f"Updated image: {movie.title} -> {image_filename}"))

		self.stdout.write(
			self.style.SUCCESS(
				f"Finished. Updated {updated_count} movies. Missing images for {not_found_count} movies."
			)
		)

	def get_image_files(self, images_folder):
		valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
		files = []
		for name in os.listdir(images_folder):
			full_path = os.path.join(images_folder, name)
			if os.path.isfile(full_path):
				ext = os.path.splitext(name)[1].lower()
				if ext in valid_extensions:
					files.append(name)
		return files

	def build_image_index(self, image_files):
		index = {}
		for filename in image_files:
			stem = os.path.splitext(filename)[0]
			normalized_stem = self.normalize_text(stem)
			index[normalized_stem] = filename

			# Support files named with the "m_" prefix used in this project.
			if stem.startswith("m_"):
				no_prefix = stem[2:]
				index[self.normalize_text(no_prefix)] = filename
		return index

	def find_image_for_title(self, title, image_index):
		normalized_title = self.normalize_text(title)

		direct = image_index.get(normalized_title)
		if direct:
			return direct

		with_prefix = image_index.get(self.normalize_text(f"m_{title}"))
		if with_prefix:
			return with_prefix

		return None

	def normalize_text(self, text):
		text = unicodedata.normalize("NFKD", text)
		text = "".join(ch for ch in text if not unicodedata.combining(ch))
		return text.strip().lower()
