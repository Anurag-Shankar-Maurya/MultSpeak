import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SpeakerDatabase:
    def __init__(self):
        self.speakers = {}  # {speaker_name: [embedding1, embedding2, ...]}
        self.mean_embeddings = {}  # {speaker_name: mean_embedding}
        self.raw_audio = {}  # {speaker_name: [(audio1, sr1), (audio2, sr2), ...]}

    def add_speaker(self, speaker_name, embedding, raw_audio=None, sample_rate=None):
        """Add speaker to database and update mean embedding"""
        if speaker_name in self.speakers:
            self.speakers[speaker_name].append(embedding)
        else:
            self.speakers[speaker_name] = [embedding]

        # Store raw audio if provided
        if raw_audio is not None and sample_rate is not None:
            if speaker_name in self.raw_audio:
                self.raw_audio[speaker_name].append((raw_audio, sample_rate))
            else:
                self.raw_audio[speaker_name] = [(raw_audio, sample_rate)]

        # Update the mean embedding
        self.mean_embeddings[speaker_name] = np.mean(self.speakers[speaker_name], axis=0)

    def remove_speaker(self, speaker_name):
        """Remove a speaker from the database"""
        if speaker_name in self.speakers:
            del self.speakers[speaker_name]
            if speaker_name in self.mean_embeddings:
                del self.mean_embeddings[speaker_name]
            if speaker_name in self.raw_audio:
                del self.raw_audio[speaker_name]
            return True
        return False

    def get_speaker_samples_count(self, speaker_name):
        """Get the number of samples for a speaker"""
        if speaker_name in self.speakers:
            return len(self.speakers[speaker_name])
        return 0

    def get_all_raw_audio_samples(self):
        """Get all raw audio samples for training the scaler"""
        all_samples = []
        for speaker_name, samples in self.raw_audio.items():
            all_samples.extend(samples)
        return all_samples

    def find_closest_match(self, embedding, method='combined', threshold=None):
        """
        Find the closest matching speaker using multiple methods

        Parameters:
        - embedding: the test embedding
        - method: 'mean', 'max', or 'combined'
        - threshold: confidence threshold, if None uses adaptive threshold

        Returns:
        - speaker name and confidence score
        """
        if not self.speakers:
            return "Unknown", 0.0

        best_match = "Unknown"
        max_similarity = -1

        # Set adaptive threshold based on number of speakers
        if threshold is None:
            if len(self.speakers) <= 3:
                threshold = 0.45  # Less strict for small number of speakers
            elif len(self.speakers) <= 7:
                threshold = 0.40
            else:
                threshold = 0.35

        for speaker_name, embeddings in self.speakers.items():
            # Calculate similarity based on method
            if method == 'mean':
                # Use pre-calculated mean embedding for efficiency
                mean_embedding = self.mean_embeddings[speaker_name]
                similarity = cosine_similarity(
                    mean_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]

            elif method == 'max':
                # Calculate similarity with each embedding and take max
                similarities = []
                for stored_embedding in embeddings:
                    sim = cosine_similarity(
                        stored_embedding.reshape(1, -1),
                        embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append(sim)
                similarity = max(similarities) if similarities else 0

            else:  # 'combined' - default
                # Use weighted combination of mean and max methods
                mean_embedding = self.mean_embeddings[speaker_name]
                mean_sim = cosine_similarity(
                    mean_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]

                # Also calculate similarities with individual embeddings
                individual_sims = []
                for stored_embedding in embeddings:
                    sim = cosine_similarity(
                        stored_embedding.reshape(1, -1),
                        embedding.reshape(1, -1)
                    )[0][0]
                    individual_sims.append(sim)

                max_sim = max(individual_sims) if individual_sims else 0

                # Weight mean similarity more if we have many samples
                if len(embeddings) >= 5:
                    similarity = 0.8 * mean_sim + 0.2 * max_sim  # Increased weight for mean
                else:
                    similarity = 0.6 * mean_sim + 0.4 * max_sim  # Adjusted weights

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = speaker_name

        if max_similarity < threshold:
            return "Unknown", max_similarity

        return best_match, max_similarity

    def save_database(self, file_path):
        """Save the database to a file"""
        with open(file_path, 'wb') as file:
            pickle.dump({
                'speakers': self.speakers,
                'mean_embeddings': self.mean_embeddings,
                'raw_audio': self.raw_audio
            }, file)

    def load_database(self, file_path):
        """Load the database from a file"""
        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)

                # Handle both old and new format
                if isinstance(data, dict) and 'speakers' in data:
                    self.speakers = data['speakers']

                    # Load mean_embeddings if available, otherwise recalculate
                    if 'mean_embeddings' in data:
                        self.mean_embeddings = data['mean_embeddings']
                    else:
                        self._recalculate_mean_embeddings()

                    # Load raw_audio if available
                    if 'raw_audio' in data:
                        self.raw_audio = data['raw_audio']
                    else:
                        self.raw_audio = {}
                else:
                    # Old format - just the speakers dictionary
                    self.speakers = data
                    self._recalculate_mean_embeddings()
                    self.raw_audio = {}

            return True
        except FileNotFoundError:
            self.speakers = {}
            self.mean_embeddings = {}
            self.raw_audio = {}
            return False
        except Exception as e:
            print(f"Error loading database: {str(e)}")
            self.speakers = {}
            self.mean_embeddings = {}
            self.raw_audio = {}
            return False

    def _recalculate_mean_embeddings(self):
        """Recalculate mean embeddings for all speakers"""
        self.mean_embeddings = {}
        for speaker_name, embeddings in self.speakers.items():
            if embeddings:
                self.mean_embeddings[speaker_name] = np.mean(embeddings, axis=0)
