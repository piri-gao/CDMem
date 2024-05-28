import os
import sys
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Optional, List
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


Model = Literal["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-instruct"]
ChatModel = Literal["gpt-4", "gpt-3.5-turbo"]
CompleteModel = Literal["gpt-3.5-turbo-instruct"]

class GPTWrapper:
    def __init__(self, model: Model):
        self.client = OpenAI(
                base_url=os.getenv('OPENAI_API_BASE_URL') if 'OPENAI_API_BASE_URL' in os.environ else None,
                api_key=os.getenv('OPENAI_API_KEY'),
                )
        self.model = model
                    
    def __call__(self, prompt: str, stop: List[str] = ["\n"], mode: str = 'chat'):
        try:
            cur_try = 0
            while cur_try < 6:
                if mode == "chat":
                    text = self.get_chat(prompt=prompt, model=self.model, temperature=cur_try * 0.2, stop_strs=stop)
                elif mode == "complete":
                    text = self.get_completion(prompt=prompt, model=self.model, temperature=cur_try * 0.2, stop_strs=stop)
                else:
                    raise ValueError(f"Invalid mode: {mode}, mode must be 'chat' or 'complete'.")
                # dumb way to do this
                if len(text.strip()) >= 5:
                    return text
                cur_try += 1
            return ""
        except Exception as e:
            print(prompt)
            print(e)
            import sys
            sys.exit(1)
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_chat(self, prompt: str, model: ChatModel, temperature: float = 0.0, max_tokens: int = 256, 
                    stop_strs: Optional[List[str]] = None, is_batched: bool = False) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stop=stop_strs,
            temperature=temperature,
        )
        return response.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_completion(self, prompt: str, model: CompleteModel, temperature: float = 0.0, max_tokens: int = 256, 
                        stop_strs: Optional[List[str]] = None) -> str:
        response = self.client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop_strs,
        )
        return response.choices[0].text
            