import anthropic
import openai

from custom_logging import logger


class Claude:
    def __init__(self, model="claude-3-haiku-20240307", messages=None, prefix_message=None, temperature=0, stream=True):
        #claude-3-opus-20240229
        self.client = anthropic.Anthropic()
        self.model = model
        self.messages = messages or []
        self.prefix_message = prefix_message
        self.temperature = temperature
        self.stream = stream

    def chat(self, prompt, temperature=None, stream=None, **kwargs):
        prompt = prompt.strip()
        content = prompt

        if len(self.messages) > 0 and self.messages[-1]["role"] == "user":
            self.pop()
            
        self.messages.append({"role": "user", "content": self.prefix_message+content})
        stream = stream if stream is not None else self.stream
        if stream:
            ans = ""
            with self.client.messages.stream(model=self.model,
                                             max_tokens=4096,
                                             messages=self.messages,
                                             temperature=temperature or self.temperature,
                                             **kwargs) as strm:
                for text in strm.text_stream:
                    #print(text, end="", flush=True)
                    # TODO: 音声合成を行いながら文章を生成することで高速化
                    ans += text

        else:
            message = self.client.messages.create(model=self.model,
                                                  max_tokens=4096,
                                                  messages=self.messages,
                                                  temperature=temperature or self.temperature,
                                                  **kwargs)
            ans = message.content[0].text
            logger.info(f"claude token usage: {message.usage}")
            
        logger.info(f"claude: {ans}")
        self.messages.append({"role": "assistant", "content": ans})

    def get_messages(self):
        return self.messages

    def pop(self):
        self.messages.pop()
        
    def FIFO(self):
        del self.messages[0]
        

# TODO: テストとデバッグ
class ChatGPT:
    def __init__(self, api_key, model="gpt-4o-mini", messages=None, prefix_message=None, temperature=0, stream=False):
        openai.api_key = api_key
        self.model = model
        self.messages = messages or []
        self.prefix_message = prefix_message
        self.temperature = temperature
        self.stream = stream

    def chat(self, prompt, temperature=None, stream=None, **kwargs):
        prompt = prompt.strip()
        content = prompt
        self.messages.append({"role": "system", "content": "You are a helpful assistant."})
        self.messages.append({"role": "user", "content": self.prefix_message+content})
        stream = stream if stream is not None else self.stream
        if stream:
            ans = ""
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                temperature=temperature or self.temperature,
                stream=True,
                **kwargs
            )
            for chunk in response:
                chunk_message = chunk['choices'][0]['delta']
                if "content" in chunk_message:
                    text = chunk_message["content"]
                    ans += text
        else:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                temperature=temperature or self.temperature,
                **kwargs
            )
            ans = response.choices[0].message.content
            print(ans)
            print(response)
        self.messages.append({"role": "assistant", "content": ans})

    def get_messages(self):
        return self.messages

    def pop(self):
        self.messages.pop()
        
    def FIFO(self):
        del self.messages[0]
        

def query_chatgpt(text: str) -> str:
    response = openai.ChatCompletion.create(
                    model = "gpt-3.5-turbo-16k-0613",
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": text},
                        #{"role":"assistant", "content": ""}
                    ],
                    temperature=0
                )
    return response['choices'][0]['message']['content']


def query_claude(text: str, client) -> str: 
    message = client.messages.create(
        model="claude-3-haiku-20240307", # claude-3-opus-20240229, claude-3-sonnet-20240229
        max_tokens=1000, # 出力上限（4096まで）
        temperature=0.0, # 0.0-1.0
        system="", # 必要ならシステムプロンプトを設定
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
            #{"role":"assistant", "content": ""}
        ]
    )
    return message.content[0].text