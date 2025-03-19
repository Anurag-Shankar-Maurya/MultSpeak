import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SpeakerDatabase:
    def __init__(self):
        self.speakers = {}  # {speaker_name: [embedding1, embedding2, ...]}

    def add_speaker(self, speaker_name, embedding):
        """Add a new speaker or update an existing one"""
        if speaker_name in self.speakers:
            self.speakers[speaker_name].append(embedding)
        else:
            self.speakers[speaker_name] = [embedding]

    def find_closest_match(self, embedding):
        """Find the closest matching speaker"""
        if not self.speakers:
            return "Unknown", 0.0

        max_similarity = -1
        best_match = "Unknown"

        for speaker_name, embeddings in self.speakers.items():
            for stored_embedding in embeddings:
                # Calculate cosine similarity
                similarity = self._calculate_similarity(embedding, stored_embedding)

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = speaker_name

        return best_match, max_similarity

    def _calculate_similarity(self, embedding1, embedding2):
        """Calculate similarity between two embeddings"""
        # Reshape embeddings for cosine_similarity
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]

        return similarity

    def save_database(self, file_path):
        """Save the database to a file"""
        with open(file_path, 'wb') as file:
            pickle.dump(self.speakers, file)

    def load_database(self, file_path):
        """Load the database from a file"""
        try:
            with open(file_path, 'rb') as file:
                self.speakers = pickle.load(file)
        except FileNotFoundError:
            self.speakers = {}
