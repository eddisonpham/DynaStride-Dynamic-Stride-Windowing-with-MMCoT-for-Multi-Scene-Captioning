### LLaVA as a baseline
LLaVA is designed a single-image captioning model, so we can apply it to each frame independently for frame-wise captions. However, this introduces several problems or limitations for our scene-captioning method:

  - **temporal difficulties**: captioning frames independently means the model has no context to changes over time; results in a model being able to capture what is happening in a frame, but not actions across them
  - **fragments/overlap**: frame-wise captioning can be redundant or result in a lack of descriptions across multiple frames

I think LLaVA is a viable option as a lower bound or baseline we can start with, but I don't think it can reliably capture temporal relationships across frames.

### Video-LLaMA 2
- **input-output**: takes in a sequence of frames and uses transformer-based temporal and spatial modeling to output a single polished caption for the sequence (can also provide temporally-aligned captions for video segments)
- noted for strong spatial-temporal modeling, outperforming static-image LLMs for video-captioning purposes
- **strengths vs. LLaVA**
    - directly models temporal continuity
    - produces coherent, temporally aware captions
- **limitations**: slightly more complex to fine-tune and deploy; requires well-annotated datasets for best performance
