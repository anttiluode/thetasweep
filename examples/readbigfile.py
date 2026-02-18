import sys
import time
import shutil
from thetasweep import SweepRetriever

def visualize_timeline(hits, total_chunks, width=60):
    """
    Visualizes where hits occur in the document timeline.
    [...*.......*...**.....]
    """
    timeline = ['·'] * width
    
    # Map chunk indices to timeline slots
    for chunk, score in hits:
        # Find index of this chunk in the original text list
        # (Note: In a prod app we'd store indices, but this works for demo)
        try:
            idx = retriever.chunk_texts.index(chunk)
            slot = int((idx / total_chunks) * (width - 1))
            
            # Use different chars for relevance intensity
            if score > 0.25: char = '█'  # Strong hit
            elif score > 0.15: char = '▓'
            else: char = '▒'
            
            timeline[slot] = char
        except ValueError:
            continue
            
    return "".join(timeline)

def main():
    if len(sys.argv) < 2:
        print("Usage: python theta_nav.py <text_file>")
        print("Using internal demo text (Alice in Wonderland Ch.1) for now...\n")
        # Fallback demo text if no file provided
        text = """
        Alice was beginning to get very tired of sitting by her sister on the bank, 
        and of having nothing to do: once or twice she had peeped into the book her sister was reading, 
        but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 
        'without pictures or conversations?' So she was considering in her own mind (as well as she could, 
        for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain 
        would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with 
        pink eyes ran close by her. There was nothing so VERY remarkable in that; nor did Alice think it so 
        VERY much out of the way to hear the Rabbit say to itself, 'Oh dear! Oh dear! I shall be late!' 
        (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, 
        but at the time it all seemed quite natural); but when the Rabbit actually TOOK A WATCH OUT OF ITS 
        WAISTCOAT-POCKET, and looked at it, and then hurried on, Alice started to her feet, for it flashed 
        across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch 
        to take out of it, and burning with curiosity, she ran across the field after it, and fortunately 
        was just in time to see it pop down a large rabbit-hole under the hedge.
        """ * 20 # Repeated to simulate length
    else:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            text = f.read()

    print(f"Initializing ThetaSweep on {len(text):,} characters...")
    
    # 1. Build Index
    # sigma=1.2 gives a good balance of specific phrasing vs context
    global retriever
    retriever = SweepRetriever(stack_size=512, sigma=1.2)
    t0 = time.perf_counter()
    retriever.build_index(text, verbose=True)
    index_time = time.perf_counter() - t0
    
    total_chunks = len(retriever.chunk_texts)
    print(f"\nTime Machine Ready ({index_time:.2f}s). Type 'exit' to quit.\n")

    # 2. Interactive Loop
    while True:
        query = input("\n[Theta-Nav] >> ")
        if query.lower() in ('exit', 'quit'):
            break
            
        t0 = time.perf_counter()
        # Retrieve broad hits to populate the timeline
        hits = retriever.retrieve(query, top_k=max(5, total_chunks // 10))
        query_ms = (time.perf_counter() - t0) * 1000
        
        # Visualize
        timeline_bar = visualize_timeline(hits, total_chunks)
        print(f"\nTimeline: [{timeline_bar}] ({len(hits)} hits in {query_ms:.1f}ms)")
        
        # Show top result detailed
        best_chunk, best_score = hits[0]
        print(f"\nTop Hit ({best_score:.3f}):")
        print("-" * 40)
        print(best_chunk.strip()[:300] + "...")
        print("-" * 40)

if __name__ == "__main__":
    main()