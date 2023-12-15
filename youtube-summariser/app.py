from flask import Flask, render_template, request
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the T5 model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/summary', methods=["POST"])
def transcribe():
    if request.method == 'POST':
        transcribe_text = ""
        link = request.form['video_id']
        response = YouTubeTranscriptApi.get_transcript(link)
        for n in range(0, len(response)):
            transcribe_text = transcribe_text + response[n]['text'] + " "
        print(transcribe_text)
        summarised = summarise(transcribe_text)
        print(f"summary--+{summarised}")
        return render_template('index.html', summary=summarised)
    else:
        return render_template('index.html')


def summarise(transcript, max_chunk_size=512):

    # Split the transcript into chunks
    transcript_chunks = [transcript[i:i + max_chunk_size] for i in range(0, len(transcript), max_chunk_size)]

    # Initialize an empty list to store individual summaries
    summaries = []

    # Summarize each chunk
    for chunk in transcript_chunks:
        # Add the T5 specific prefix "summarize: "
        input_text = "summarize: " + chunk

        # Tokenize and generate the summary
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                     early_stopping=True)

        # Decode the generated summary and append to the list of summaries
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    # Concatenate the individual summaries to get the overall summary
    overall_summary = " ".join(summaries)

    return overall_summary


if __name__ == "__main__":
    app.run(debug=True)

