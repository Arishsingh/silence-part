from flask import Flask, render_template, request, send_file, jsonify
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tempfile import NamedTemporaryFile
import numpy as np

app = Flask(__name__)
app.secret_key = "supersecretkey"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/auto_trim', methods=['POST'])
def auto_trim():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file uploaded."}), 400

        video_file = request.files['video']

        with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_in:
            video_file.save(temp_in.name)
            clip = VideoFileClip(temp_in.name)

            if clip.audio is None:
                clip.close()
                return jsonify({"error": "No audio track found in this video."}), 400

            original_fps = clip.fps
            original_size = (clip.w, clip.h)

            sample_rate = 0.1
            threshold = 0.01
            silence_min_duration = 1.5  # updated to trim silences >= 1.5s

            is_silent = []
            for t in np.arange(0, clip.duration, sample_rate):
                frame = clip.audio.get_frame(t)
                rms = np.sqrt(np.mean(np.square(frame))) if isinstance(frame, (list, tuple, np.ndarray)) else 0
                is_silent.append(rms < threshold)

            silent_blocks = []
            current_silence_start = None
            times = np.arange(0, clip.duration, sample_rate)

            for idx, silent in enumerate(is_silent):
                t = times[idx]
                if silent:
                    if current_silence_start is None:
                        current_silence_start = t
                else:
                    if current_silence_start is not None:
                        if t - current_silence_start >= silence_min_duration:
                            silent_blocks.append((current_silence_start, t))
                        current_silence_start = None

            if current_silence_start is not None:
                if clip.duration - current_silence_start >= silence_min_duration:
                    silent_blocks.append((current_silence_start, clip.duration))

            keep_ranges = []
            last_end = 0
            for start, end in silent_blocks:
                if start > last_end:
                    keep_ranges.append((last_end, start))
                last_end = end
            if last_end < clip.duration:
                keep_ranges.append((last_end, clip.duration))

            chunks = [clip.subclip(start, end) for start, end in keep_ranges if end > start]

            if not chunks:
                clip.close()
                return jsonify({"error": "No non-silent segments found."}), 400

            final = concatenate_videoclips(chunks)
            final = final.resize(newsize=original_size)

            with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_out:
                final.write_videofile(
                    temp_out.name,
                    codec="libx264",
                    audio_codec="aac",
                    threads=4,
                    preset="veryslow",     # best quality
                    bitrate="20M",         # high bitrate
                    fps=original_fps,
                    ffmpeg_params=["-crf", "18"]  # visually lossless
                )
                clip.close()
                final.close()

                return send_file(temp_out.name, mimetype="video/mp4", as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
