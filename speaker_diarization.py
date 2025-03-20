import torch
import torchaudio
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import librosa
from speechbrain.inference import SpeakerRecognition


class SpeakerDiarization:
    def __init__(self, voice_processor, speaker_db):
        self.voice_processor = voice_processor
        self.speaker_db = speaker_db
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create cache directory if it doesn't exist
        os.makedirs("pretrained_models", exist_ok=True)

        # Initialize the speaker recognition model
        self.spkrec = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb-verification"
        )

        # Configuration params
        self.window_size = 1.5  # in seconds
        self.step_size = 0.75  # in seconds

    def segment_audio(self, audio, sample_rate):
        """Split audio into overlapping segments for processing"""
        window_samples = int(self.window_size * sample_rate)
        step_samples = int(self.step_size * sample_rate)

        segments = []
        segment_times = []

        # Split audio into overlapping segments
        for start in range(0, len(audio) - window_samples, step_samples):
            end = start + window_samples
            segment = audio[start:end]
            segments.append(segment)
            segment_times.append((start / sample_rate, end / sample_rate))

        return segments, segment_times

    def extract_segment_embeddings(self, segments, sample_rate):
        """Extract embeddings for each segment"""
        embeddings = []

        for segment in segments:
            emb = self.voice_processor.extract_embedding(segment, sample_rate)
            embeddings.append(emb)

        return np.array(embeddings)

    def cluster_embeddings(self, embeddings, min_speakers=1, max_speakers=8):
        """Cluster embeddings to find speaker segments"""
        # Determine optimal number of clusters using silhouette score
        from sklearn.metrics import silhouette_score

        # Limit range based on dataset size
        max_speakers = min(max_speakers, len(embeddings) - 1) if len(embeddings) > 1 else 1

        if len(embeddings) <= 1 or min_speakers >= max_speakers:
            return [0] * len(embeddings)  # Return single cluster if not enough data

        best_score = -1
        best_n_clusters = min_speakers

        for n_clusters in range(min_speakers, min(max_speakers + 1, len(embeddings))):
            if n_clusters >= len(embeddings):
                continue

            # Use cosine affinity for clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='cosine',
                linkage='average'
            ).fit(embeddings)

            if len(set(clustering.labels_)) > 1:  # Ensure we have more than one label
                score = silhouette_score(embeddings, clustering.labels_, metric='cosine')
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters

        # Final clustering
        clustering = AgglomerativeClustering(
            n_clusters=best_n_clusters,
            affinity='cosine',
            linkage='average'
        ).fit(embeddings)

        return clustering.labels_

    def match_clusters_to_speakers(self, embeddings, cluster_labels):
        """Match cluster centroids to known speakers"""
        speakers = list(self.speaker_db.speakers.keys())
        if not speakers:
            return {i: f"Speaker {i + 1}" for i in range(max(cluster_labels) + 1)}

        # Calculate cluster centroids
        unique_labels = np.unique(cluster_labels)
        centroids = {}

        for label in unique_labels:
            indices = np.where(cluster_labels == label)[0]
            centroid = np.mean(embeddings[indices], axis=0)
            centroids[label] = centroid

        # Match centroids to speakers
        matches = {}

        for label, centroid in centroids.items():
            best_speaker = "Unknown"
            best_similarity = 0.3  # Threshold

            for speaker in speakers:
                # Get speaker embeddings and calculate similarity
                speaker_embs = self.speaker_db.speakers[speaker]
                for speaker_emb in speaker_embs:
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity = cosine_similarity(centroid.reshape(1, -1),
                                                   speaker_emb.reshape(1, -1))[0][0]

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_speaker = speaker

            if best_speaker == "Unknown":
                matches[label] = f"Speaker {label + 1}"
            else:
                matches[label] = best_speaker

        return matches

    def diarize(self, audio, sample_rate):
        """Perform diarization on audio segment"""
        # Preprocess audio
        audio, sample_rate = self.voice_processor.preprocess_audio(audio, sample_rate)

        # Check if audio is too short
        if len(audio) / sample_rate < self.window_size:
            # Short audio - try direct matching
            embedding = self.voice_processor.extract_embedding(audio, sample_rate)
            speaker, confidence = self.speaker_db.find_closest_match(embedding)
            return [(0, len(audio) / sample_rate, speaker, confidence)]

        # Segment audio
        segments, segment_times = self.segment_audio(audio, sample_rate)

        # Extract embeddings for each segment
        embeddings = self.extract_segment_embeddings(segments, sample_rate)

        # Cluster embeddings
        cluster_labels = self.cluster_embeddings(embeddings)

        # Match clusters to speakers
        speaker_matches = self.match_clusters_to_speakers(embeddings, cluster_labels)

        # Create diarization results
        results = []

        for i, ((start_time, end_time), label) in enumerate(zip(segment_times, cluster_labels)):
            speaker = speaker_matches[label]

            # Calculate confidence using similarity to known speakers
            emb = embeddings[i]
            if speaker in self.speaker_db.speakers:
                # Use max similarity to speaker's embeddings as confidence
                similarities = []
                for speaker_emb in self.speaker_db.speakers[speaker]:
                    from sklearn.metrics.pairwise import cosine_similarity
                    sim = cosine_similarity(emb.reshape(1, -1), speaker_emb.reshape(1, -1))[0][0]
                    similarities.append(sim)
                confidence = max(similarities) if similarities else 0.5
            else:
                confidence = 0.5  # Default confidence for unknown speakers

            results.append((start_time, end_time, speaker, confidence))

        return results
