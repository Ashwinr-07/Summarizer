import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
from tqdm import tqdm
import re

def transcribe_video(video_path, model_size='tiny'):
    """
    Load the specified Whisper model (saving to a local folder) and transcribe the given video.
    
    Parameters:
        video_path (str): Path to the video file.
        model_size (str): The variant of the model to use ('tiny', 'small', etc.).
        
    Returns:
        str: The transcribed text from the video.
    """
    print(f"Loading Whisper '{model_size}' model (saving to './models/whisper')...")
    # The model will be stored in ./models/whisper; if it already exists, it will be loaded locally.
    model = whisper.load_model(model_size, download_root='./models/whisper')
    print("Whisper model loaded successfully!")
    
    print("Starting transcription. Please wait...")
    result = model.transcribe(video_path)
    
    return result["text"]

def clean_transcript(text):
    """
    Perform basic cleaning on the transcript text.
    """
    return text.strip()

def format_transcript_lines(text):
    """
    Format the transcript so that each sentence appears on a new line.
    """
    # Split the transcript into sentences using regex.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    formatted = "\n".join(sentence.strip() for sentence in sentences if sentence.strip())
    return formatted

def chunk_text(text, max_chunk_length=800):
    """
    Split a long text into smaller chunks (by word boundaries) so that each chunk
    does not exceed max_chunk_length characters.
    """
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        # If adding the next word exceeds the limit, store the current chunk.
        if len(current_chunk) + len(word) + 1 > max_chunk_length:
            chunks.append(current_chunk)
            current_chunk = word
        else:
            current_chunk = (current_chunk + " " + word).strip()
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def summarize_text(text, summarizer, max_length, min_length, prompt_prefix=""):
    """
    Summarize the given text using the Hugging Face summarization pipeline.
    """
    input_text = prompt_prefix + text if prompt_prefix else text
    
    tokenizer = summarizer.tokenizer
    encoded = tokenizer.encode(input_text, return_tensors="pt")

    if encoded.shape[1] > tokenizer.model_max_length:
        print("Input text is too long; truncating to the model's maximum input length...")
        encoded = encoded[:, :tokenizer.model_max_length]
        input_text = tokenizer.decode(encoded[0], skip_special_tokens=True)

    summary = summarizer(
        input_text, 
        max_length=max_length, 
        min_length=min_length, 
        do_sample=False
    )
    return summary[0]['summary_text']

def iterative_refinement(text, summarizer, max_length, min_length, prompt_prefix=""):
    """
    Repeatedly chunk and summarize the text until it fits into the model's max length.
    """
    tokenizer = summarizer.tokenizer
    
    while True:
        encoded = tokenizer.encode(text, return_tensors="pt")
        if encoded.shape[1] <= tokenizer.model_max_length:
            break
        else:
            print("Combined text too long, chunking for iterative refinement...")
            smaller_chunks = chunk_text(text, max_chunk_length=800)
            refined_list = []
            for chunk in tqdm(smaller_chunks, desc="Refining in sub-chunks", unit="chunk"):
                refined_chunk = summarize_text(
                    chunk,
                    summarizer,
                    max_length=max_length,
                    min_length=min_length,
                    prompt_prefix=prompt_prefix
                )
                refined_list.append(refined_chunk)
            text = " ".join(refined_list)
    
    final_summary = summarize_text(text, summarizer, max_length, min_length, prompt_prefix)
    return final_summary

def generate_summary_from_chunks(text, summarizer, max_length, min_length, prompt_prefix=""):
    """
    Chunk the original text, summarize each piece, combine them,
    and then iteratively refine to ensure the final summary does not exceed the model limit.
    """
    chunks = chunk_text(text, max_chunk_length=800)
    print(f"Total chunks to summarize: {len(chunks)}")

    chunk_summaries = []
    for chunk in tqdm(chunks, desc="Summarizing chunks", unit="chunk"):
        summary = summarize_text(
            chunk,
            summarizer,
            max_length,
            min_length,
            prompt_prefix=prompt_prefix
        )
        chunk_summaries.append(summary)

    combined_summary = " ".join(chunk_summaries)
    print("Refining combined summary (iterative)...")
    final_summary = iterative_refinement(
        combined_summary, 
        summarizer, 
        max_length, 
        min_length, 
        prompt_prefix=prompt_prefix
    )
    return final_summary

def load_summarization_pipeline():
    """
    Load the Hugging Face summarization pipeline using local caching.
    """
    model_name = "sshleifer/distilbart-cnn-12-6"
    cache_directory = "./models/transformers"

    # Load the tokenizer and model with caching enabled.
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_directory)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_directory)
    
    # Create the summarization pipeline with the loaded model and tokenizer.
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

def main():
    # Define file paths.
    video_path = "main.mp4"
    transcript_file = "transcript.txt"
    
    print("="*50)
    print("Step 1: Transcription")
    print("="*50)
    
    # Transcribe the video.
    transcript = transcribe_video(video_path, model_size='tiny')
    transcript = clean_transcript(transcript)
    formatted_transcript = format_transcript_lines(transcript)
    
    # Save the formatted transcript.
    with open(transcript_file, "w") as f:
        f.write(formatted_transcript)
    print(f"Transcript saved to '{transcript_file}'.")
    
    # Print the formatted transcript with each sentence on a new line.
    print("\nFormatted Transcript:\n" + "-"*25)
    print(formatted_transcript)
    print("-"*25 + "\n")
    
    print("="*50)
    print("Step 2: Summarization and Explanation")
    print("="*50)
    print("Loading summarization model from local cache (if available)...")
    
    # Load the summarization pipeline.
    summarizer = load_summarization_pipeline()
    print("Summarization model loaded successfully!")
    
    # Generate Small Bullet-Point Summary.
    print("\nGenerating Small Summary (bullet points)...")
    small_summary = generate_summary_from_chunks(
        formatted_transcript,
        summarizer,
        max_length=50,     # Adjust as needed.
        min_length=30,
        prompt_prefix="Provide a concise bullet-point summary:\n- "
    )
    with open("summary_small.txt", "w") as f:
        f.write(small_summary)
    print("Small summary saved to 'summary_small.txt'.")

    # Generate Medium Bullet-Point Summary.
    print("\nGenerating Medium Summary (bullet points)...")
    medium_summary = generate_summary_from_chunks(
        formatted_transcript,
        summarizer,
        max_length=100,    # Adjust as needed.
        min_length=60,
        prompt_prefix="Summarize the following text in bullet points:\n- "
    )
    with open("summary_medium.txt", "w") as f:
        f.write(medium_summary)
    print("Medium summary saved to 'summary_medium.txt'.")

    # Generate Large Bullet-Point Summary.
    print("\nGenerating Large Summary (bullet points)...")
    large_summary = generate_summary_from_chunks(
        formatted_transcript,
        summarizer,
        max_length=200,    # Adjust as needed.
        min_length=120,
        prompt_prefix="Summarize in detailed bullet points:\n- "
    )
    with open("summary_large.txt", "w") as f:
        f.write(large_summary)
    print("Large summary saved to 'summary_large.txt'.")

    # Generate Detailed Explanation (Paragraph style).
    print("\nGenerating a Detailed Explanation (paragraph style)...")
    explanation_prompt = (
        "Provide a detailed, coherent explanation of the main content, "
        "without using bullet points. Write in paragraph form:\n\n"
    )
    explanation = generate_summary_from_chunks(
        formatted_transcript,
        summarizer,
        max_length=300,    # Adjust as needed.
        min_length=150,
        prompt_prefix=explanation_prompt
    )
    with open("explanation.txt", "w") as f:
        f.write(explanation)
    print("Explanation saved to 'explanation.txt'.")

    print("\nAll processing complete!")

if __name__ == "__main__":
    main()
