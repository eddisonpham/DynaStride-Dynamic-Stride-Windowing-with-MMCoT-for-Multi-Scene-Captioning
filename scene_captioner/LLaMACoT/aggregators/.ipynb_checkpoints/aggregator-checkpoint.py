import re
import torch
from transformers import pipeline

# ---------------------------
# Aggregator class
# ---------------------------
class CookingAggregator:
    def __init__(self, model_id="Qwen/Qwen3-4B-Instruct-2507", device_index=0):
        self.device_index = device_index
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            device=device_index
        )

    # ---------------------------
    # Prompt builder
    # ---------------------------
    def build_cooking_summary_prompt(self, captions):
        prompt = (
            "You are given multiple captions from a short cooking clip, in chronological order.\n"
            "Write ONE concise sentence that is both short, and instructional.\n"
            "Use an imperative tone, as if giving instructions for cooking or performing a task.\n"
            "Your response MUST be enclosed between <ANSWER> and </ANSWER>, containing ONLY the final instruction sentence.\n"
        )


        for i, cap in enumerate(captions, 1):
            if cap and cap != "None":
                prompt += f"{i}. {cap}\n"

        prompt += "\nOutput:\n"
        return prompt

    # ---------------------------
    # Extract answer after <ANSWER> tags
    # ---------------------------
    @staticmethod
    def extract_answer(answer_text):
        # Find all <ANSWER>...</ANSWER> blocks
        matches = re.findall(r"<ANSWER>(.*?)</ANSWER>", answer_text, re.DOTALL | re.IGNORECASE)
        
        if not matches:
            return answer_text.strip()  # fallback if no <ANSWER> tags found
    
        # Check the last match
        last_answer = matches[-1].strip().lower()
        if last_answer == "and" and len(matches) > 1:
            # Use the previous one if last is just "and"
            return matches[-2].strip()
        else:
            return matches[-1].strip()


    # ---------------------------
    # Generate summary
    # ---------------------------
    def generate_cooking_summary(self, captions, max_new_tokens=50):
        prompt = self.build_cooking_summary_prompt(captions)
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            return_full_text=False  # ✅ only return the generated text, not the prompt
        )
        print(outputs)
        return self.extract_answer(outputs[0]['generated_text'])
