from typing import Dict, List
from openai import AzureOpenAI
import logging

logger = logging.getLogger(__name__)


class AzureConfig:
    api_type: str = "azure"
    api_key: str = "<YOUR_API_KEY>"
    api_version: str = "2023-12-01-preview"
    azure_endpoint: str = "<YOUR_ENDPOINT>"
    model: str = "GPT4-Turbo"
    limit: int = 30000
    price: float = 0.01


class ChatAPI:
    def __init__(
        self,
        config: AzureConfig,
    ):
        self.messages: List[Dict[str, str]] = []
        self.history: List[Dict[str, str]] = []

        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.azure_endpoint,
        )

        self.model = config.model
        self.max_limit = min(config.limit - 2000, 20000)
        self.price = config.price
        self.usage_tokens = 0
        self.cost = 0  # USD money cost

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self.history.append(self.messages[-1])

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self.history.append(self.messages[-1])

    def get_system_response(self) -> str:
        try:
            self.do_truncation = False

            response = self.client.chat.completions.create(
                model=self.model, messages=self.messages
            )
            response_message = response.choices[0].message

            usage_tokens = response.usage.total_tokens
            self.cost += usage_tokens * self.price / 1000
            print(
                f"[ChatGPT] current model {self.model}, usage_tokens: {usage_tokens}, "
                f"cost: ${self.cost:.5f}, price: ${self.price:.5f}"
            )
            if usage_tokens > self.max_limit:
                print(
                    f"[ChatGPT] truncate the conversation to avoid token usage limit, save money"
                )
                self.truncate()

            return response_message.content
        except Exception as e:
            logger.warning(f"[ChatGPT] Error: {e}")
            return "Sorry, I am not able to respond to that."

    def get_system_response_stream(self):
        response = self.client.chat.completions.create(
            model=self.model, messages=self.messages, stream=True
        )
        for chuck in response:
            if len(chuck.choices) > 0 and chuck.choices[0].finish_reason != "stop":
                if chuck.choices[0].delta.content is None:
                    continue
                yield chuck.choices[0].delta.content

        # stream mode does not support token usage check, give a rough estimation
        usage_tokens = int(sum([len(item["content"]) for item in self.message]) / 3.5)
        self.usage_tokens = usage_tokens
        self.cost += usage_tokens * self.price / 1000
        logger.info(
            f"[ChatGPT] current model {self.model}, usage_tokens approximation: {usage_tokens},"
            f" cost: ${self.cost:.2f}, price: ${self.price:.2f}"
        )

        if usage_tokens > self.max_limit:
            logger.info(
                f"[ChatGPT] truncate the conversation to avoid token usage limit"
            )
            self.truncate()

    @property
    def message(self):
        return self.messages

    @message.setter
    def message(self, message):
        """
        Usually at the dialog begining
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message_first_turn},
        {"role": "assistant", "content": assistant_message_first_turn},
        """
        self.init_length = len(message)
        self.messages = message
        self.history.extend(self.messages)

    def truncate(self, percentage: int = 3):
        self.do_truncation = True
        usr_idx = [
            idx
            for idx in range(len(self.messages))
            if self.messages[idx]["role"] == "user"
        ]
        middle_idx = usr_idx[len(usr_idx) // percentage]
        logger.info(
            f"\033[33m [ChatGPT] truncate the conversation at index: {middle_idx} from {usr_idx} \033[m"
        )
        self.messages = self.messages[: self.init_length] + self.messages[middle_idx:]

    def clear(self):
        """end the conversation"""
        self.messages = []
        self.history = []



if __name__ == "__main__":
    config = AzureConfig()
    chat_api = ChatAPI(config)
    chat_api.message = [
        {"role": "system", "content": "Are you chatGPT?"},
        {"role": "user", "content": "Answer Yes or No."},
    ]
    print(chat_api.messages)
    user_input = input("User: ")
    chat_api.add_user_message(user_input)
    print(chat_api.messages)
    print(chat_api.cost)
    # while True:
    #     user_input = input("User: ")
    #     if user_input == "exit":
    #         break
    #     chat_api.add_user_message(user_input)
    #     response = chat_api.get_system_response()
    #     print("Assistant:", response)
    #     chat_api.add_assistant_message(response)

