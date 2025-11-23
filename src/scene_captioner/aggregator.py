import re
from typing import List, Union
from transformers import pipeline


class CookingAggregator:
    """Aggregates multiple frame captions into a single cooking instruction summary."""
    
    def __init__(self, model_id: str = "Qwen/Qwen3-4B-Instruct-2507", device: Union[str, int] = "cuda:0"):
        """
        Initialize the cooking aggregator.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run the model on (cuda:0, cuda:1, or cpu)
        """
        self.device = device
        try:
            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                device=device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CookingAggregator: {e}")

    def build_cooking_summary_prompt(self, captions: List[str]) -> str:
        """
        Build the prompt for generating cooking summary.
        
        Args:
            captions: List of frame captions in chronological order
            
        Returns:
            Formatted prompt string
        """
        filtered_captions = [cap for cap in captions if cap and cap != "None"]
        if not filtered_captions:
            return ""
        
        prompt = (
            "You are given multiple captions from a short cooking clip, in chronological order.\n"
            "Write ONE concise sentence that is both short, and instructional.\n"
            "Use an imperative tone, as if giving instructions for cooking or performing a task.\n"
            "Your response MUST be enclosed between <ANSWER> and </ANSWER>, containing ONLY the final instruction sentence.\n"
            "Captions:\n"
            + "\n".join([f"{i}. {cap}" for i, cap in enumerate(filtered_captions, 1)])
            + "\nOutput:\n"
        )
        return prompt

    @staticmethod
    def extract_answer(answer_text: str) -> str:
        """
        Extract the answer from the model output.
        
        Args:
            answer_text: Raw model output text
            
        Returns:
            Extracted answer string
        """
        if not answer_text:
            return ""
        
        matches = re.findall(r"<ANSWER>(.*?)</ANSWER>", answer_text, re.DOTALL | re.IGNORECASE)
        
        if not matches:
            return answer_text.strip()
    
        last_answer = matches[-1].strip().lower()
        # Handle edge case where last answer is just "and"
        if last_answer == "and" and len(matches) > 1:
            return matches[-2].strip()
        
        return matches[-1].strip()

    def generate_cooking_summary(self, captions: List[str], max_new_tokens: int = 50) -> str:
        """
        Generate a cooking summary from multiple captions.
        
        Args:
            captions: List of frame captions
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Aggregated cooking instruction summary
        """
        if not captions:
            return ""
        
        try:
            prompt = self.build_cooking_summary_prompt(captions)
            if not prompt:
                return ""
            
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                return_full_text=False
            )
            
            if not outputs or len(outputs) == 0:
                return ""
            
            result = self.extract_answer(outputs[0]['generated_text'])
            return result
            
        except Exception as e:
            return ""
