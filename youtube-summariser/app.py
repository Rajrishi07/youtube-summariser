from flask import Flask, render_template, request
from transformers import BartForConditionalGeneration, BartTokenizer
from youtube_transcript_api import YouTubeTranscriptApi


# Load the pre-trained BART model and tokenizer for summarization
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/summary', methods=["POST"])
def transcribe():
    if request.method == 'POST':
        transcribe_text = ""
        link = request.form['video_id']
        print(link)
        response = YouTubeTranscriptApi.get_transcript(extract_video_id(link))
        for n in range(0, len(response)):
            transcribe_text = transcribe_text + response[n]['text'] + " "
        print(transcribe_text)
        summarised = summarise(transcribe_text)
        print(f"summary--+{summarised}")
        return render_template('index.html', summary=summarised)
    else:
        return render_template('index.html')


def extract_video_id(youtube_url):
    # Extract video ID from YouTube URL
    video_id = youtube_url.split("be/")[1]
    return video_id

def summarise(transcript, max_chunk_size=1024):
    # Split the transcript into chunks
    transcript_chunks = [transcript[i:i + max_chunk_size] for i in range(0, len(transcript), max_chunk_size)]

    # Initialize an empty list to store individual summaries
    summaries = []

    # Summarize each chunk
    for chunk in transcript_chunks:
        print(f"{chunk}\n\n\n")
        input_ids = tokenizer.encode(chunk, return_tensors="pt", max_length=1024, truncation=True)

        # Generate summary
        summary_ids = model.generate(input_ids, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode and append the generated summary
        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append("->--"+generated_summary+"\n")

    # Concatenate the individual summaries to get the overall summary
    overall_summary = " ".join(summaries)

    return overall_summary


if __name__ == "__main__":
    app.run(debug=True)

